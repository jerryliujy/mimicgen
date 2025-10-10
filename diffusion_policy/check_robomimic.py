# 创建 check_robomimic_env.py
import robomimic
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
from diffusion_policy.env.robomimic.robomimic_image_wrapper import RobomimicImageWrapper
from mimicgen.envs.robosuite import *

# print(f"RoboMimic version: {robomimic.__version__}")

# # 检查支持的环境
# print("Available environments:")
# from robosuite.environments import ALL_ENVIRONMENTS
# print(ALL_ENVIRONMENTS)

# # 检查 Square 任务是否正确注册
# try:
#     from diffusion_policy.env.robomimic.robosuite.nut_assembly import Square_D2
#     print("✅ Square environment found in robosuite")
# except ImportError as e:
#     print(f"❌ Square environment not found: {e}")

# # 测试直接创建 Square 环境
# try:
#     import robosuite as suite
#     env = suite.make(
#         env_name="Square_D2",
#         robots="Panda",
#         has_renderer=False,
#         has_offscreen_renderer=True,
#         reward_shaping=True,  # 重要：启用奖励塑形
#     )
    
#     obs = env.reset()
#     action = [1, 1, 1, 0, 0, 0, 3, 0]
#     # action = env.action_space.sample()
#     obs, reward, done, info = env.step(action)
    
#     print(f"Direct Square env test:")
#     print(f"  Reward: {reward}")
#     print(f"  Info: {info}")
#     print(f"  Success: {info.get('success', 'No success key')}")
    
#     env.close()
    
# except Exception as e:
#     print(f"Direct Square env test failed: {e}")