import gym
from gym import spaces
import numpy as np

from world import FlatWorld
from collections import OrderedDict


class FlatEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(FlatEnv, self).__init__()

        self._config = {
            'action_type': 'cartesian',
            'arena_radius': 20.,
            'zone_radius': 1.,
            'num_hazards': 10,
            'lidar_range': 8.,
            'lidar_num_bins': 12,
            'max_move': 2.,
            'hazard_cost': 1.,
            'goal_reward': 10.,
            'dense_distance_reward': 0.1,
            'max_time': 25,
        }
        self._world = None
        self.action_space = spaces.Box(low=-1,
                                       high=1,
                                       shape=(2, ),
                                       dtype=np.float)
        self.observation_space = self._build_observation_space()
        self.step_num = 0
        self.done = True

    def _build_observation_space(self):
        obs_space_dict = OrderedDict()
        obs_space_dict['goal_lidar'] = gym.spaces.Box(
            0.0, 1.0, (self._config['lidar_num_bins'], ), dtype=np.float32)
        obs_space_dict['hazards_lidar'] = gym.spaces.Box(
            0.0, 1.0, (self._config['lidar_num_bins'], ), dtype=np.float32)
        #obs_space_dict['walls_lidar'] = gym.spaces.Box(
        #    0.0, 1.0, (self._config['lidar_num_bins'], ), dtype=np.float32)
        return gym.spaces.Dict(obs_space_dict)

    def step(self, action):
        info = {'cost': 0.}
        last_distance_to_goal = np.linalg.norm(
            self._world.vector_to_goal_cartesian())

        # update world state
        action = np.array(action, copy=False)  # Cast to ndarray
        assert not self.done, 'Environment must be reset before stepping'

        if self._config['action_type'] == 'polar':
            raise Exception()
        move = (action /
                max(np.linalg.norm(action), 1.)) * self._config['max_move']
        self._world.update_robot(move)

        # set reward and cost
        if self._world.check_hazard_collision():
            info['cost'] += self._config['hazard_cost']
        if self._world.check_out_of_bounds():
            info['cost'] += self._config['oob_cost']
            self.done = True
        if self._world.check_goal_collision():
            reward = self._config['goal_reward']
            self.done = True
        else:
            reward = 0.
        distance_to_goal = np.linalg.norm(
            self._world.vector_to_goal_cartesian())
        reward += self._config['dense_distance_reward'] * (
            last_distance_to_goal - distance_to_goal)

        if self.step_num >= self._config['max_time']:
            done = True
        else:
            self.step_num += 1

        # construct observation
        observation = {}
        observation['hazards_lidar'] = self._world.hazard_lidar(
            self._config['lidar_range'], self._config['lidar_num_bins'])
        observation['goal_lidar'] = self._world.goal_lidar(
            self._config['lidar_range'], self._config['lidar_num_bins'])
        #observation['wall_lidar'] = self._world.wall_lidar(
        #    self._robot_position, self._config['lidar_range'],
        #    self._config['lidar_num_bins'])
        return observation, reward, self.done, info

    def reset(self):
        self._world = FlatWorld(True,
                                arena_radius=self._config['arena_radius'],
                                zone_radius=self._config['zone_radius'],
                                num_forbidden=self._config['num_hazards'])
        observation = None
        self.done = False
        self.step_num = 0
        return observation

    def render(self, mode='human'):
        self._world.render()

    def close(self):
        pass


if __name__ == "__main__":
    env = FlatEnv()
    env.reset()
    while not env.done:
        goal_vec = env._world.vector_to_goal_cartesian()
        print("Vector to goal:", goal_vec)
        print("Distance:", np.linalg.norm(goal_vec))
        print(env.step(goal_vec / 2.))
        env.render()
        input()