# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
Import all robosuite environments to ensure they are registered.
"""

# 导入所有自定义环境以确保注册
# try:
from diffusion_policy.env.robomimic.robosuite.nut_assembly import *

from robosuite.environments import ALL_ENVIRONMENTS
if "Square_D2" not in ALL_ENVIRONMENTS:
    ALL_ENVIRONMENTS.append("Square_D2")
    
#     # 添加更多环境导入
# except ImportError as e:
#     print(f"Warning: Could not import custom environments: {e}")

# # 确保环境在 robosuite 注册表中
# import robosuite.environments as suite
# from robosuite.environments.manipulation.single_arm_env import register_env

# # 如果需要手动注册
# try:
#     from diffusion_policy.env.robomimic.robosuite.nut_assembly import Square_D2
#     # 注册环境
#     register_env(Square_D2)
# except ImportError:
#     pass