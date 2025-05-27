import h5py
import numpy as np
from diffusion_policy.env_runner.robomimic_image_runner import RobomimicImageRunner
import hydra
from omegaconf import OmegaConf

# 检查数据集
dataset_path = "data/robomimic/datasets/square/mimicgen/no_noise.hdf5"
print("=== Dataset Info ===")
with h5py.File(dataset_path, 'r') as f:
    print("Dataset keys:", list(f.keys()))
    demos = list(f['data'].keys())
    print(f"Number of demos: {len(demos)}")
    
    demo_0 = f[f'data/{demos[0]}']
    print(f"Demo 0 keys: {list(demo_0.keys())}")
    
    # 检查动作和状态
    actions = demo_0['actions'][:]
    states = demo_0['states'][:]
    print(f"Demo 0 length: {len(actions)} steps")
    print(f"Action shape: {actions.shape}")
    print(f"State shape: {states.shape}")
    
    # 检查是否有 datagen_info
    if 'datagen_info' in demo_0:
        datagen_info = demo_0['datagen_info']
        print(f"Datagen info keys: {list(datagen_info.keys())}")

print("\n=== Testing Environment Reward Calculation ===")

# 测试环境是否能正确计算奖励
try:
    # 加载配置
    cfg_path = "diffusion_policy/config/eval/eval_square_image.yaml"
    cfg = OmegaConf.load(cfg_path)
    
    # 创建环境运行器
    env_runner = hydra.utils.instantiate(cfg.env_runner)
    
    # 创建环境
    env = env_runner._make_env()
    
    # 重置环境并测试奖励
    obs = env.reset()
    print(f"Environment reset successful")
    
    # 执行几个随机动作测试奖励计算
    for i in range(5):
        # 随机动作
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(f"Step {i}: reward={reward}, done={done}")
        if 'success' in info:
            print(f"  Success: {info['success']}")
        
        if done:
            print("Episode finished early")
            break
    
    env.close()
    
except Exception as e:
    print(f"Environment test failed: {e}")
    import traceback
    traceback.print_exc()