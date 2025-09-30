import hydra
import collections
from omegaconf import OmegaConf
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils

def create_env(env_meta, shape_meta, enable_render=True):
    modality_mapping = collections.defaultdict(list)
    for key, attr in shape_meta['obs'].items():
        modality_mapping[attr.get('type', 'low_dim')].append(key)
    ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False, 
        render_offscreen=enable_render,
        use_image_obs=enable_render, 
    )
    return env

# 检查任务配置
print("=== Task Configuration ===")

# 1. 检查数据集的元信息
dataset_path = "data/robomimic/datasets/square/mimicgen/no_noise.hdf5"
env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
print(f"Environment metadata: {env_meta}")

# 2. 创建环境检查奖励设置
env = create_env(
    env_meta=env_meta,
    shape_meta=FileUtils.get_shape_metadata_from_dataset(dataset_path),
    enable_render=False
)

print(f"Environment type: {type(env)}")
print(f"Environment name: {env.name if hasattr(env, 'name') else 'Unknown'}")

# 测试环境
obs = env.reset()
print(f"Observation keys: {obs.keys()}")

# 检查奖励函数
try:
    # 获取当前状态的奖励（通常初始奖励应该是0）
    current_reward = env._get_reward() if hasattr(env, '_get_reward') else "No _get_reward method"
    print(f"Current reward: {current_reward}")
    
    # 检查是否有成功检查函数
    current_success = env._check_success() if hasattr(env, '_check_success') else "No _check_success method"
    print(f"Current success: {current_success}")
    
except Exception as e:
    print(f"Reward/Success check failed: {e}")

env.close()