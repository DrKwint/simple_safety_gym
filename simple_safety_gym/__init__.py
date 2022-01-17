import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='Safexp-PointSimplegoal-v0',
    entry_point='simple_safety_gym.envs:FlatEnv',
)