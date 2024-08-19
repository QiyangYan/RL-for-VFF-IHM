import torch
import numpy as np
from Real4.Real4.agents.simple_nn import SimpleNN

best_model_path = '/Users/qiyangyan/Downloads/diffusion_imitation-master/agents/models_terminate_cube_cylinder_WORK_DONTCHANGE_forReal/SimpleNN/best_model.pth'
agent_control_mode = torch.load(best_model_path)

# state_norm = [-1.00399578,  2.03830909,  1.13698724, -2.03830909,
#               0.        ,  0.        ,  0.        ,  0.        ,
#               1.13500517, -1.17162528,  0.        ,
#               0.29378594,  0.        ,  0.        ,  0.96031258,
#               2.19249811, -2.36672184, -0.83316621]
#
# goal_norm = [-1.68439562, -0.78084566,  0.        ,
#              -0.63880569,  0.        ,  0.        ,  1.4259707 ,
#              2.1154911 , -0.79339538]


state_norm = [-0.68802757, -0.49061039,  0.72458292,  0.49061039,  0.        ,  0.        ,  0.        ,  0.        ,  0.72742832, -1.37406025,  0.        ,  0.75286657,  0.        ,  0.        ,  0.68981531,  1.96711903, -1.82817235,  1.48245994]
goal_norm = [-1.68439562, -0.78084566,  0.        , -0.63880569,  0.        ,  0.        ,  1.4259707 ,  2.10204128, -0.81107925]


robot_qpos = state_norm[0:4]
object_qpos = state_norm[8:15]
# inputs = np.concatenate((robot_qpos, object_qpos, goal_norm[:-2]))
inputs = np.concatenate((robot_qpos, object_qpos, goal_norm))
inputs = torch.from_numpy(inputs).float().unsqueeze(0)

agent_control_mode.eval()
with torch.no_grad():
    outputs_test = agent_control_mode(inputs)
    prediction = torch.argmax(outputs_test, axis=1)

print(prediction)