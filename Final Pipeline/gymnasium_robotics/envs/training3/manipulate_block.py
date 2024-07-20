import os

import numpy as np
from gymnasium.utils.ezpickle import EzPickle

from gymnasium_robotics.envs.training3 import (
    MujocoManipulateEnv,
)

# Ensure we get the path separator correct on windows
MANIPULATE_BLOCK_XML = os.path.join("IHM_simple_convex", "IHM.xml")


class MujocoHandBlockEnv(MujocoManipulateEnv, EzPickle):
    # noqa: D415
    """
    ## Description

    This environment was introduced in ["Multi-Goal Reinforcement Learning: Challenging Robotics Environments and Request for Research"](https://arxiv.org/abs/1802.09464).

    The environment is based on the same robot hand as in the `HandReach` environment, the [Shadow Dexterous Hand](https://www.shadowrobot.com/). In this task a block is placed on the palm of the hand. The task is to then manipulate the
    block such that a target pose is achieved. The goal is 7-dimensional and includes the target position (in Cartesian coordinates) and target rotation (in quaternions). In addition, variations of this environment can be used with increasing
    levels of difficulty:

    * `HandManipulateBlockRotateZ-v1`: Random target rotation around the *z* axis of the block. No target position.
    * `HandManipulateBlockRotateParallel-v1`: Random target rotation around the *z* axis of the block and axis-aligned target rotations for the *x* and *y* axes. No target position.
    * `HandManipulateBlockRotateXYZ-v1`: Random target rotation for all axes of the block. No target position.
    * `HandManipulateBlockFull-v1`: Random target rotation for all axes of the block. Random target position.

    ## Action Space

    The action space is a `Box(-1.0, 1.0, (2,), float32)`. The control actions are absolute angular positions of the actuated joints (non-coupled). The input of the control actions is set to a range between -1 and 1 by scaling the actual actuator angle ranges.
    The elements of the action array are the following:
    For other parameters
    ------ 1. Check in the actual CAD file to confirm the range for each finger, should be the same, the following is just a rough value
    Vertical = 0 degree, to right is +, to left is -, same for both big and small servo

    | Num | Action                                              | Control Min | Control Max | Angle Min    | Angle Max   | Name (in corresponding XML file) | Joint | Unit        |
    | --- | ----------------------------------------------------| ----------- | ----------- | ------------ | ----------  |--------------------------------- | ----- | ----------- |
    | 0   | Angular position of the right finger                | -1          | 1           | -1.047 (rad) | 1.047 (rad) | robot0:A_WRJ1                    | hinge | angle (rad) |
        # | 1   | Angular position of the left finger                 | -1          | 1           | -1.047 (rad) | 1.047 (rad) | robot0:A_WRJ0                    | hinge | angle (rad) |
    | 2   | Friction States                                     | -1          | 1           | -1.571 (rad) | 1.571 (rad) | robot0:A_FFJ3 & robot0:A_FFJ4    | hinge | angle (rad) |


    ## Observation Space

    The observation is a `goal-aware observation space`. It consists of a dictionary with information about the robot's joint and block states, as well as information about the goal. The dictionary consists of the following 3 keys:

    * `observation`: its value is an `ndarray` of shape `(8,)`. It consists of kinematic information of the block object and finger joints. The elements of the array correspond to the following:


    | Num | Observation                                                       | Min    | Max    | Joint Name (in corresponding XML file) |Joint Type| Unit                     |
    |-----|-------------------------------------------------------------------|--------|--------|----------------------------------------|----------|------------------------- |
    | 0   | Angular position of the right finger joint                        | -Inf   | Inf    | robot0:WRJ1                            | hinge    | angle (rad)              |
        # | 1   | Angular position of the left finger joint                         | -Inf   | Inf    | robot0:WRJ0                            | hinge    | angle (rad)              |
        # | 2   | Angular velocity of the right finger joint                        | -Inf   | Inf    | robot0:FFJ3                            | hinge    | angle (rad)              |
        # | 3   | Angular velocity of the left finger joint                         | -Inf   | Inf    | robot0:FFJ2                            | hinge    | angle (rad)              |
    | 4   | Angular position of the right finger friction servo               | -Inf   | Inf    | robot0:WRJ1                            | hinge    | angle (rad)              |
    | 5   | Angular position of the left finger friction servo                | -Inf   | Inf    | robot0:WRJ0                            | hinge    | angle (rad)              |
        # | 6   | Angular velocity of the right finger friction servo               | -Inf   | Inf    | robot0:FFJ3                            | hinge    | angle (rad)              |
        # | 7   | Angular velocity of the left finger friction servo                | -Inf   | Inf    | robot0:FFJ2                            | hinge    | angle (rad)              |

        # | 8   | Linear velocity of the block in x direction                       | -Inf   | Inf    | object:joint                           | free     | velocity (m/s)           |
        # | 9   | Linear velocity of the block in y direction                       | -Inf   | Inf    | object:joint                           | free     | velocity (m/s)           |
        # | 10  | Linear velocity of the block in z direction                       | -Inf   | Inf    | object:joint                           | free     | velocity (m/s)           |
        # | 11  | Angular velocity of the block in x axis                           | -Inf   | Inf    | object:joint                           | free     | angular velocity (rad/s) |
        # | 12  | Angular velocity of the block in y axis                           | -Inf   | Inf    | object:joint                           | free     | angular velocity (rad/s) |
        # | 13  | Angular velocity of the block in z axis                           | -Inf   | Inf    | object:joint                           | free     | angular velocity (rad/s) |
    | 14  | Position of the block in the x coordinate                         | -Inf   | Inf    | object:joint                           | free     | position (m)             |
    | 15  | Position of the block in the y coordinate                         | -Inf   | Inf    | object:joint                           | free     | position (m)             |
    | 16  | Position of the block in the z coordinate                         | -Inf   | Inf    | object:joint                           | free     | position (m)             |
        # | 17  | w component of the quaternion orientation of the block            | -Inf   | Inf    | object:joint                           | free     | -                        |
        # | 18  | x component of the quaternion orientation of the block            | -Inf   | Inf    | object:joint                           | free     | -                        |
        # | 19  | y component of the quaternion orientation of the block            | -Inf   | Inf    | object:joint                           | free     | -                        |
        # | 20  | z component of the quaternion orientation of the block            | -Inf   | Inf    | object:joint                           | free     | -                        |
    | 21  | Achieved radi between left-contact-point and left motor            | -Inf   | Inf    | object:joint                           | free     | -                        |
    | 22  | Achieved radi between right-contact-point and right motor            | -Inf   | Inf    | object:joint                           | free     | -                        |


    * `desired_goal`: this key represents the final goal to be achieved. In this environment it is a 7-dimensional `ndarray`, `(7,)`, that consists of the pose information of the block. The elements of the array are the following:

    | Num | Observation                                                                                                                           | Min    | Max    | Joint Name (in corresponding XML file) | Joint Type | Unit         |
    |-----|---------------------------------------------------------------------------------------------------------------------------------------|--------|--------|----------------------------------------|------------|--------------|
    | 0   | Target x coordinate of the block                                                                                                      | -Inf   | Inf    | target:joint                           | free       | position (m) |
    | 1   | Target y coordinate of the block                                                                                                      | -Inf   | Inf    | target:joint                           | free       | position (m) |
    | 2   | Target z coordinate of the block                                                                                                      | -Inf   | Inf    | target:joint                           | free       | position (m) |
    | 3   | Target w component of the quaternion orientation of the block                                                                         | -Inf   | Inf    | target:joint                           | free       | -            |
    | 4   | Target x component of the quaternion orientation of the block                                                                         | -Inf   | Inf    | target:joint                           | free       | -            |
    | 5   | Target y component of the quaternion orientation of the block                                                                         | -Inf   | Inf    | target:joint                           | free       | -            |
    | 6   | Target z component of the quaternion orientation of the block                                                                         | -Inf   | Inf    | target:joint                           | free       | -            |
    | 7  | Goal radi between left-contact-point and left motor            | -Inf   | Inf    | object:joint                           | free     | -                        |
    | 8  | Goal radi between right-contact-point and right motor            | -Inf   | Inf    | object:joint                           | free     | -                        |


    * `achieved_goal`: this key represents the current state of the block, as if it would have achieved a goal. This is useful for goal orientated learning algorithms such as those that use [Hindsight Experience Replay](https://arxiv.org/abs/1707.01495) (HER).
    The value is an `ndarray` with shape `(5,)`. The elements of the array are the following:

    | Num | Observation                                                                                                                           | Min    | Max    | Joint Name (in corresponding XML file) | Joint Type | Unit         |
    |-----|---------------------------------------------------------------------------------------------------------------------------------------|--------|--------|----------------------------------------|------------|--------------|
    | 0   | Current x coordinate of the block                                                                                                      | -Inf   | Inf    | object:joint                           | free       | position (m) |
    | 1   | Current y coordinate of the block                                                                                                      | -Inf   | Inf    | object:joint                           | free       | position (m) |
    | 2   | Current z coordinate of the block                                                                                                      | -Inf   | Inf    | object:joint                           | free       | position (m) |
    | 3   | Current w component of the quaternion orientation of the block                                                                         | -Inf   | Inf    | object:joint                           | free       | -            |
    | 4   | Current x component of the quaternion orientation of the block                                                                         | -Inf   | Inf    | object:joint                           | free       | -            |
    | 5   | Current y component of the quaternion orientation of the block                                                                         | -Inf   | Inf    | object:joint                           | free       | -            |
    | 6   | Current z component of the quaternion orientation of the block                                                                         | -Inf   | Inf    | object:joint                           | free       | -            |
    | 7  | Achieved radi between left-contact-point and left motor            | -Inf   | Inf    | object:joint                           | free     | -                        |
    | 8  | Achieved radi between right-contact-point and right motor            | -Inf   | Inf    | object:joint                           | free     | -                        |


    ## Rewards

    The reward can be initialized as `sparse` or `dense`:
    - *sparse*: the returned reward can have two values: `-1` if the block hasn't reached its final target pose, and `0` if the block is in its final target pose. The block is considered to have reached its final goal if the theta angle difference (theta angle of the
    [3D axis angle representation](https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation) is less than 0.1 and if the Euclidean distance to the target position is also less than 0.01 m.
    - *dense*: the returned reward is the negative summation of the Euclidean distance to the block's target and the theta angle difference to the target orientation. The positional distance is multiplied by a factor of 10 to avoid being dominated by the rotational difference.

    To initialize this environment with one of the mentioned reward functions the type of reward must be specified in the id string when the environment is initialized. For `sparse` reward the id is the default of the environment, `HandManipulateBlock-v1`. However, for `dense`
    reward the id must be modified to `HandManipulateBlockDense-v1` and initialized as follows:

    ```python
    import gymnasium as gym

    env = gym.make('HandManipulateBlock-v1')
    ```

    The rest of the id's of the other environment variations follow the same convention to select between a sparse or dense reward function.

    ## Starting State

    When the environment is reset the joints of the hand are initialized to their resting position with a 0 displacement. The blocks position and orientation are randomly selected. The initial position is set to `(x,y,z)=(1, 0.87, 0.2)` and an offset is added to each coordinate
    sampled from a normal distribution with 0 mean and 0.005 standard deviation.
    While the initial orientation is set to `(w,x,y,z)=(1,0,0,0)` and an axis is randomly selected depending on the environment variation to add an angle offset sampled from a uniform distribution with range `[-pi, pi]`.

    The target pose of the block is obtained by adding a random offset to the initial block pose. For the position the offset is sampled from a uniform distribution with range `[(x_min, x_max), (y_min,y_max), (z_min, z_max)] = [(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]`. The orientation
    offset is sampled from a uniform distribution with range `[-pi,pi]` and added to one of the Euler axis depending on the environment variation.


    ## Episode End

    The episode will be `truncated` when the duration reaches a total of `max_episode_steps` which by default is set to 50 timesteps.
    The episode is never `terminated` since the task is continuing with infinite horizon.

    ## Arguments

    To increase/decrease the maximum number of timesteps before the episode is `truncated` the `max_episode_steps` argument can be set at initialization. The default value is 50. For example, to increase the total number of timesteps to 100 make the environment as follows:

    ```python
    import gymnasium as gym

    env = gym.make('HandManipulateBlock-v1', max_episode_steps=100)
    ```

    The same applies for the other environment variations.

    ## Version History

    * v1: the environment depends on the newest [mujoco python bindings](https://mujoco.readthedocs.io/en/latest/python.html) maintained by the MuJoCo team in Deepmind.
    * v0: the environment depends on `mujoco_py` which is no longer maintained.

    """

    def __init__(
        self,
        target_position="random",
        target_rotation="z",
        reward_type="dense",
        randomize_initial_position=False,
        randomize_initial_rotation=False,
        **kwargs,
    ):
        MujocoManipulateEnv.__init__(
            self,
            model_path=MANIPULATE_BLOCK_XML,
            target_position=target_position,
            target_rotation=target_rotation,
            reward_type=reward_type,
            **kwargs,
        )
        EzPickle.__init__(self, target_position, target_rotation, reward_type, **kwargs)
        self.number_of_corners = 4
