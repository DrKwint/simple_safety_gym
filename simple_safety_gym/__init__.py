import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='Safexp-PointSimpleGoal-v0',
    entry_point='simple_safety_gym.envs:FlatEnv',
    timestep_limit=25,
    reward_threshold=10.0,
    nondeterministic=False,
)