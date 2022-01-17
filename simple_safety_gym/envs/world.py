import collections
import numpy as np
import matplotlib.pyplot as plt


class PolarCoordinate:
    def __init__(self, distance, angle):
        self._distance = distance
        self._angle = angle % (2 * np.pi)

    def zero():
        return PolarCoordinate(0, 0)

    def random_within_radius(max_distance):
        rng = np.random.default_rng()
        return PolarCoordinate(rng.random() * max_distance,
                               rng.random() * 2 * np.pi)

    def to_cartesian(self):
        return (self._distance * np.cos(self._angle),
                self._distance * np.sin(self._angle))

    def from_cartesian(x, y):
        angle = np.arctan(y / x)
        if x < 0.:
            angle += np.pi
        if np.isnan(angle):  # assume x was zero
            angle = np.pi / 2
        return PolarCoordinate(np.sqrt(np.square(x) + np.square(y)), angle)

    def min_distance_from_line(self, line_p1, line_p2):
        (x0, y0) = self.to_cartesian()
        (x1, y1) = line_p1.to_cartesian()
        (x2, y2) = line_p2.to_cartesian()
        numerator = np.abs((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1))
        denominator = np.sqrt(np.square(x2 - x1) + np.square(y2 - y1))
        return numerator / denominator

    def __str__(self):
        return "Polar { " + "distance {}, angle {}".format(
            self._distance, self._angle) + ' }'


class Circle:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius


class FlatWorld:
    def __init__(self, intermediate_zones, arena_radius, zone_radius,
                 num_forbidden):
        """Robot is always created at (0,0)"""
        self._robot = PolarCoordinate.zero()
        self._arena_radius = arena_radius
        self._goal = Circle(PolarCoordinate.random_within_radius(arena_radius),
                            zone_radius)
        self._hazards = [
            Circle(PolarCoordinate.random_within_radius(arena_radius),
                   zone_radius) for _ in range(num_forbidden)
        ]
        self._zone_radius = zone_radius
    
    def move_goal(self, distance=None):
        """
        if not distance:
            distance = self._arena_radius / 2.
        current_cartesian_posn = self._goal.center.to_cartesian()

        while True:
            random_vector = np.random(2)
            random_unit_vector = random_vector / np.linalg.norm(random_vector)

            target_posn = random_unit_vector * distance + current_cartesian_posn
            if np.linalg.norm(target_posn) < self._arena_radius:
                break
        """
        self._goal = Circle(PolarCoordinate.random_within_radius(self._arena_radius),
                            self._zone_radius)


    def update_robot(self, cartesian_delta):
        self._robot = PolarCoordinate.from_cartesian(*(
            self._robot.to_cartesian() + cartesian_delta))

    def vector_to_goal_cartesian(self):
        return np.array(self._goal.center.to_cartesian()) - np.array(
            self._robot.to_cartesian())

    def check_hazard_collision(self):
        colliding = 0
        for hazard in self._hazards:
            distance = np.linalg.norm(
                np.array(self._robot.to_cartesian()) -
                np.array(hazard.center.to_cartesian()))
            if distance < hazard.radius:
                colliding += 1
        return colliding

    def check_out_of_bounds(self):
        return self._robot._distance > self._arena_radius

    def check_goal_collision(self):
        distance = np.linalg.norm(
            np.array(self._robot.to_cartesian()) -
            np.array(self._goal.center.to_cartesian()))
        return distance < self._goal.radius

    def hazard_lidar(self, max_range, num_bins, alias=True):
        return self.obs_lidar_pseudo([h.center for h in self._hazards],
                                     self._robot.to_cartesian(), max_range,
                                     num_bins, alias)

    def goal_lidar(self, max_range, num_bins, alias=True):
        return self.obs_lidar_pseudo([self._goal.center],
                                     self._robot.to_cartesian(), max_range,
                                     num_bins, alias)

    def obs_lidar_pseudo(self,
                         positions,
                         cartesian_ego_position,
                         max_range,
                         num_bins,
                         alias=True):
        obs = np.zeros(num_bins)
        bin_size = (np.pi * 2) / num_bins

        egocentric_hazard_centers = [
            PolarCoordinate.from_cartesian(*(
                np.array(position.to_cartesian()) -
                np.array(cartesian_ego_position))) for position in positions
        ]
        for hazard in egocentric_hazard_centers:
            assert np.isfinite(hazard._angle)
            bin = int(hazard._angle / bin_size)
            bin_angle = bin_size * bin
            sensor = max(0, max_range - hazard._distance) / max_range
            obs[bin] = max(obs[bin], sensor)
            if alias:
                alias = (hazard._angle - bin_angle) / bin_size
                bin_plus = (bin + 1) % num_bins
                bin_minus = (bin - 1) % num_bins
                obs[bin_plus] = max(obs[bin_plus], alias * sensor)
                obs[bin_minus] = max(obs[bin_minus], (1 - alias) * sensor)
        return obs

    def render(self):
        fig, ax = plt.subplots()
        ax.add_patch(
            plt.Circle((0, 0), self._arena_radius, color='black', fill=False))

        ax.plot(*self._robot.to_cartesian(), 'o', color='blue')

        ax.add_patch(
            plt.Circle(self._goal.center.to_cartesian(),
                       self._goal.radius,
                       color='g'))

        for bad in self._hazards:
            ax.add_patch(
                plt.Circle(bad.center.to_cartesian(), bad.radius, color='red'))

        plt.xlim([-self._arena_radius, self._arena_radius])
        plt.ylim([-self._arena_radius, self._arena_radius])
        fig.show()