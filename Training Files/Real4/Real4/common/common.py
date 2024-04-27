import numpy as np
import time

class COMMON:
    def __init__(self, env):
        self.env = env
        pass

    def pick_up(self, inAir):
        t1 = time.time()
        # pick_up_action = [0, -2, False]
        pick_up_action = [0, 2, False]
        # print("start picking")
        'The position position-controlled finger reaches the middle'
        while True: # for _ in range(105):
            # pick_up_action[0] += 0.01
            pick_up_action[0] = 0.9557
            # print(pick_up_action[0])
            state, reward, _, _, _ = self.env.step(np.array(pick_up_action))
            # while not reward['action_complete']:
            #     state, reward, _, _, _ = self.env.step(np.array(pick_up_action))
            if abs(state["observation"][0] - pick_up_action[0]) < 0.003:
                break

        # print("closing")
        'Wait until the torque-controlled finger reaches the middle'
        for _ in range(50):
            state, reward, _, _, _ = self.env.step(np.array(pick_up_action))
        # print("pick up complete --------")

        'Wait until the finger raised to air'
        lift_action = [0, -3, False]
        if inAir is True:
            print("Lifting the block")
            while True: # for _ in range(120):
                _, reward, _, _, _ = self.env.step(np.array(lift_action))
                if reward["action_complete"]:
                    break
        return reward

    @staticmethod
    def action_preprocess(action):
        friction_state = action[1]
        if friction_state >= 0:
            friction_state = 1
        elif friction_state < 0:
            friction_state = -1
        else:
            print(friction_state)
            assert friction_state == 2  # 随便写个东西来报错
        action[1] = friction_state
        return action

    @staticmethod
    def action_preprocess_control_mode(action):
        # print("original action: ", action)
        control_mode = np.clip(action[1], -1, 1) + 1
        '''
        Action Type	Control Mode       |    (Left, Right)   |   Friction State      |    (Left, Right)   |
        Slide up on right finger	        P, T                1                        H, L
        Slide down on right finger	        T, P                1                        H, L
        Slide up on left finger	            T, P                -1                       L, H
        Slide down on left finger	        P, T                -1                       L, H
        # Rotate clockwise	                T, P                0
        # Rotate anticlockwise	            P, T                0
        '''

        # if 2 / 6 > control_mode > 0:
        #     control_mode_discrete = 0
        #     friction_state = 1
        # elif 2 * 2 / 6 > control_mode > 2 / 6:
        #     control_mode_discrete = 1
        #     friction_state = 1
        # elif 3 * 2 / 6 > control_mode > 2 * 2 / 6:
        #     control_mode_discrete = 2
        #     friction_state = -1
        # elif 4 * 2 / 6 > control_mode > 3 * 2 / 6:
        #     control_mode_discrete = 3
        #     friction_state = -1
        # elif 5 * 2 / 6 > control_mode > 4 * 2 / 6:
        #     control_mode_discrete = 4
        #     friction_state = 0
        # else:
        #     assert control_mode > 5 * 2 / 6
        #     control_mode_discrete = 5
        #     friction_state = 0

        if 2 / 4 > control_mode > 0:
            control_mode_discrete = 0
            friction_state = 1
        elif 2 * 2 / 4 > control_mode > 2 / 4:
            control_mode_discrete = 1
            friction_state = 1
        elif 3 * 2 / 4 > control_mode > 2 * 2 / 4:
            control_mode_discrete = 2
            friction_state = -1
        else:
            assert control_mode > 3 * 2 / 4
            control_mode_discrete = 3
            friction_state = -1

        action[1] = control_mode_discrete
        return action, friction_state

    @staticmethod
    def modify_action(action, state, pos_idx):
        angle = state[pos_idx*2]
        action[0] = angle
        return action

    @staticmethod
    def compute_radi(a):
        # left motor pos: 0.037012 -0.1845 0.002
        # right motor pos: -0.037488 -0.1845 0.002
        a[2] = 0.002

        left_motor = [0.037012, -0.1845, 0.002]
        right_motor = [-0.037488, -0.1845, 0.002]

        assert a.shape[-1] == 7

        # radius_al = np.zeros_like(a[..., 0])
        # radius_ar = np.zeros_like(a[..., 0])

        delta_r_a_left_motor = a[..., :3] - left_motor  # pos of motor left
        delta_r_a_right_motor = a[..., :3] - right_motor  # pos of motor right
        radius_al = np.linalg.norm(delta_r_a_left_motor, axis=-1)
        radius_ar = np.linalg.norm(delta_r_a_right_motor, axis=-1)

        return radius_al, radius_ar
