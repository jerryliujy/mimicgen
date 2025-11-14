import pathlib
from typing import Dict, List
import torch
import numpy as np
import h5py
from tqdm import tqdm
import zarr
import os
import shutil
import copy
import json
import hashlib
import hydra
from filelock import FileLock
import concurrent.futures
import multiprocessing
from omegaconf import OmegaConf

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, Jpeg2k
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.common.normalize_util import (
    robomimic_abs_action_only_normalizer_from_stat,
    robomimic_abs_action_only_dual_arm_normalizer_from_stat,
    get_range_normalizer_from_stat,
    get_image_range_normalizer,
    get_identity_normalizer_from_stat,
    array_to_stats
)
register_codecs()


def get_cache_path(dataset_path, flow_dataset_path=None):
    """
    Generates cache paths for base data and flow data.
    The base cache hash is based on the main dataset path.
    The flow cache hash is based on both main and flow dataset paths
    to ensure it invalidates when flow data changes.
    """
    if isinstance(dataset_path, str):
        dataset_paths = [dataset_path]
    else:
        dataset_paths = list(dataset_path)
    
    dataset_paths.sort()
    path_str = json.dumps(dataset_paths)
    base_path_hash = hashlib.md5(path_str.encode('utf-8')).hexdigest()
    
    cache_dir = os.path.dirname(dataset_paths[0])
    base_cache_path = os.path.join(cache_dir, f"cache_base_{base_path_hash}.zarr.zip")

    flow_cache_path = None
    if flow_dataset_path:
        if isinstance(flow_dataset_path, str):
            flow_dataset_paths = [flow_dataset_path]
        else:
            flow_dataset_paths = list(flow_dataset_path)
        
        flow_dataset_paths.sort()
        # Hash includes both base and flow paths
        flow_path_str = json.dumps(dataset_paths + flow_dataset_paths)
        flow_path_hash = hashlib.md5(flow_path_str.encode('utf-8')).hexdigest()
        flow_cache_path = os.path.join(cache_dir, f"cache_flow_{flow_path_hash}.zarr.zip")

    return base_cache_path, flow_cache_path


class RobomimicReplayImageFlowDataset(BaseImageDataset):
    def __init__(self,
            shape_meta: dict,
            dataset_path: str,
            horizon=1,
            pad_before=0,
            pad_after=0,
            n_obs_steps=None,
            abs_action=False,
            rotation_rep='rotation_6d',
            use_cache=False,
            seed=42,
            val_ratio=0.0
        ):
        super().__init__()
        rotation_transformer = RotationTransformer(
            from_rep='axis_angle', to_rep=rotation_rep)

        # infer flow dataset path
        flow_dataset_paths = None
        if isinstance(dataset_path, str):
            dp = pathlib.Path(dataset_path)
            flow_dataset_paths = str(dp.parent.joinpath(dp.stem + '_flow' + dp.suffix))
        else:
            flow_dataset_paths = [str(pathlib.Path(p).parent.joinpath(pathlib.Path(p).stem + '_flow' + pathlib.Path(p).suffix)) for p in dataset_path]

        if use_cache:
            base_cache_path, flow_cache_path = get_cache_path(dataset_path, flow_dataset_paths)
            
            # Load or create base cache (obs, action)
            base_cache_lock_path = base_cache_path + '.lock'
            print('Acquiring lock on base cache.')
            with FileLock(base_cache_lock_path):
                if not os.path.exists(base_cache_path):
                    print('Base cache does not exist. Creating!')
                    replay_buffer = _convert_base_to_replay(
                        shape_meta=shape_meta,
                        dataset_path=dataset_path,
                        abs_action=abs_action,
                        rotation_transformer=rotation_transformer
                    )
                    print('Saving base cache to disk.')
                    with zarr.ZipStore(base_cache_path, mode='w') as zip_store:
                        replay_buffer.save_to_store(store=zip_store)
                else:
                    print('Loading cached base ReplayBuffer from Disk.')
                    with zarr.ZipStore(base_cache_path, mode='r') as zip_store:
                        replay_buffer = ReplayBuffer.copy_from_store(src_store=zip_store, store=zarr.MemoryStore())
                    print('Loaded base cache!')

            # Load or create flow cache
            if flow_cache_path and shape_meta.get('flow') is not None:
                flow_cache_lock_path = flow_cache_path + '.lock'
                print('Acquiring lock on flow cache.')
                with FileLock(flow_cache_lock_path):
                    # Check if flow data is already in the buffer
                    flow_keys_in_buffer = [f'flow_{key}' for key in shape_meta.get('flow', {}).keys()]
                    is_flow_loaded = all(key in replay_buffer.keys() for key in flow_keys_in_buffer)

                    if not is_flow_loaded:
                        if not os.path.exists(flow_cache_path):
                            print('Flow cache does not exist. Creating!')
                            # Create a temporary buffer for flow data
                            flow_replay_buffer = _convert_flow_to_replay(
                                shape_meta=shape_meta,
                                flow_dataset_path=flow_dataset_paths
                            )
                            print('Saving flow cache to disk.')
                            with zarr.ZipStore(flow_cache_path, mode='w') as zip_store:
                                flow_replay_buffer.save_to_store(store=zip_store)
                            # Merge flow data into the main replay buffer
                            for key, value in flow_replay_buffer.items():
                                replay_buffer.data.update({key: value})
                        else:
                            print('Loading cached flow data from Disk.')
                            with zarr.ZipStore(flow_cache_path, mode='r') as zip_store:
                                temp_rb = ReplayBuffer.copy_from_store(src_store=zip_store, store=zarr.MemoryStore())
                                for key, value in temp_rb.items():
                                    replay_buffer.data.update({key: value})
                            print('Loaded flow cache!')
        else:
            # Fallback to old behavior if not using cache
            replay_buffer = _convert_base_to_replay(
                shape_meta=shape_meta,
                dataset_path=dataset_path,
                abs_action=abs_action,
                rotation_transformer=rotation_transformer
            )
            if shape_meta.get('flow') is not None:
                flow_replay_buffer = _convert_flow_to_replay(
                    shape_meta=shape_meta,
                    flow_dataset_path=flow_dataset_paths
                )
                for key, value in flow_replay_buffer.items():
                    replay_buffer.data.update({key: value})

        rgb_keys = [key for key, attr in shape_meta['obs'].items() if attr.get('type', 'low_dim') == 'rgb']
        lowdim_keys = [key for key, attr in shape_meta['obs'].items() if attr.get('type', 'low_dim') == 'low_dim']
        flow_keys = [key for key, attr in shape_meta.get('flow', {}).items() if attr.get('type', 'rgb') == 'rgb']
        
        key_first_k = dict()
        if n_obs_steps is not None:
            for key in rgb_keys + lowdim_keys:
                key_first_k[key] = n_obs_steps

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        sampler = SequenceSampler(
            replay_buffer=replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k)
        
        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.flow_keys = flow_keys
        self.abs_action = abs_action
        self.n_obs_steps = n_obs_steps
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # action
        stat = array_to_stats(self.replay_buffer['action'])
        if self.abs_action:
            if stat['mean'].shape[-1] > 10: # dual arm
                this_normalizer = robomimic_abs_action_only_dual_arm_normalizer_from_stat(stat)
            else:
                this_normalizer = robomimic_abs_action_only_normalizer_from_stat(stat)
        else: # rel action
            this_normalizer = get_identity_normalizer_from_stat(stat)
        normalizer['action'] = this_normalizer

        # obs
        for key in self.lowdim_keys:
            stat = array_to_stats(self.replay_buffer[key])
            if key.endswith('pos'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('quat'):
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith('qpos'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            else:
                raise RuntimeError('unsupported')
            normalizer[key] = this_normalizer

        # image
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
            
        # flow
        for key in self.flow_keys:
            normalizer[f'flow_{key}'] = get_image_range_normalizer()
        
        return normalizer

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = self.sampler.sample_sequence(idx)
        to_slice = slice(self.n_obs_steps)

        obs_dict = dict()
        for key in self.rgb_keys:
            obs_dict[key] = np.moveaxis(data[key][to_slice],-1,1).astype(np.float32) / 255.
            del data[key]
        for key in self.lowdim_keys:
            obs_dict[key] = data[key][to_slice].astype(np.float32)
            del data[key]
        
        flow_dict = dict()
        for key in self.flow_keys:
            flow_key_name = f'flow_{key}'
            if flow_key_name in data:
                flow_dict[key] = np.moveaxis(data[flow_key_name][to_slice], -1, 1).astype(np.float32)
                del data[flow_key_name]
        
        torch_data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(data['action'].astype(np.float32))
        }
        if len(flow_dict) > 0:
            torch_data['flow'] = dict_apply(flow_dict, torch.from_numpy)
            
        return torch_data

# ==================== Helper Functions ====================

def _convert_actions(raw_actions, abs_action, rotation_transformer):
    actions = raw_actions
    if abs_action:
        is_dual_arm = False
        if raw_actions.shape[-1] == 14: # dual arm
            raw_actions = raw_actions.reshape(-1,2,7)
            is_dual_arm = True

        pos = raw_actions[...,:3]
        rot = raw_actions[...,3:6]
        gripper = raw_actions[...,6:]
        rot = rotation_transformer.forward(rot)
        raw_actions = np.concatenate([pos, rot, gripper], axis=-1).astype(np.float32)
    
        if is_dual_arm:
            raw_actions = raw_actions.reshape(-1,20)
        actions = raw_actions
    return actions

def _get_episode_ends(dataset_paths, load_flow=False, flow_key=None):
    episode_starts = []
    episode_ends = []
    n_steps = 0
    print(f"Scanning {len(dataset_paths)} files for metadata...")
    for path in tqdm(dataset_paths, desc="Scanning metadata"):
        try:
            with h5py.File(path, 'r') as file:
                if 'data' not in file: 
                    continue
                for i in range(len(file['data'])):
                    demo_key = f'demo_{i}'
                    if demo_key not in file['data']: 
                        continue
                    if load_flow:
                        episode_length = file[f'data/{demo_key}/flow/{flow_key}'].shape[0]
                    else:
                        episode_length = file[f'data/{demo_key}/actions'].shape[0]
                    episode_starts.append(n_steps)
                    n_steps += episode_length
                    episode_ends.append(n_steps)
        except Exception as e:
            print(f"Could not read file {path}: {e}")
    return np.array(episode_starts, dtype=np.int64), np.array(episode_ends, dtype=np.int64), n_steps

def _parallel_load_images(
    data_group,
    episode_starts,
    n_steps, 
    dataset_paths, 
    keys, 
    data_type, 
    shape_meta, 
    n_workers, 
    max_inflight_tasks
):
    def img_copy(zarr_arr, zarr_idx, file_path, hdf5_key, hdf5_idx):
        try:
            with h5py.File(file_path, 'r') as file:
                zarr_arr[zarr_idx] = file[hdf5_key][hdf5_idx]
                _ = zarr_arr[zarr_idx] # Verify decoding
            return True, None
        except Exception:
            return False
            # import traceback
            # err = traceback.format_exc()
            # return False, f"[{data_type}] file={file_path}, dataset={hdf5_key}, hdf5_idx={hdf5_idx}, zarr_idx={zarr_idx}\n{err}"
        
    print(f"Steps: {n_steps}, keys: {keys}")
    with tqdm(total=n_steps * len(keys), desc=f"Loading {data_type} data", mininterval=1.0) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = set()
            for key in keys:
                zarr_key = f'flow_{key}' if data_type == 'flow' else key
                shape = tuple(shape_meta[data_type][key]['shape'])
                c, h, w = shape
                compressor = Jpeg2k(level=50)
                img_arr = data_group.require_dataset(
                    name=zarr_key,
                    shape=(n_steps,h,w,c),
                    chunks=(1,h,w,c),
                    compressor=compressor,
                    dtype=np.uint8
                )
                
                demo_idx_offset = 0
                for path in dataset_paths:
                    if not os.path.exists(path): 
                        continue
                    with h5py.File(path, 'r') as file:
                        if 'data' not in file: 
                            continue
                        
                        demos = file['data']
                        for i in range(len(demos)):
                            demo_key = f'demo_{i}'
                            if demo_key not in demos: 
                                continue
                            hdf5_key = f"data/{demo_key}/{data_type}/{key}"
                            if hdf5_key not in file: 
                                continue
                            
                            global_demo_idx = demo_idx_offset + i
                            demo_len = file[hdf5_key].shape[0]
                            
                            for hdf5_idx in range(demo_len):
                                if len(futures) >= max_inflight_tasks:
                                    completed, futures = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
                                    for f in completed:
                                        # ok, err = f.result()
                                        # if not ok:
                                        #     raise RuntimeError(err)
                                        if not f.result(): 
                                            raise RuntimeError(f'Failed to encode {data_type} data!')
                                    pbar.update(len(completed))
                                
                                zarr_idx = episode_starts[global_demo_idx] + hdf5_idx
                                futures.add(executor.submit(img_copy, img_arr, zarr_idx, path, hdf5_key, hdf5_idx))
                        demo_idx_offset += len(demos)
            
            completed, futures = concurrent.futures.wait(futures)
            for f in completed:
                if not f.result(): 
                    raise RuntimeError(f'Failed to encode {data_type} data!')
            pbar.update(len(completed))

def _convert_base_to_replay(
    shape_meta, 
    dataset_path, 
    abs_action, 
    rotation_transformer, 
    n_workers=None, 
    max_inflight_tasks=None
):
    if n_workers is None: 
        n_workers = multiprocessing.cpu_count()
    if max_inflight_tasks is None: 
        max_inflight_tasks = n_workers * 5

    if isinstance(dataset_path, str): 
        dataset_paths = [dataset_path]
    else: 
        dataset_paths = list(dataset_path)
        
    rgb_keys = [key for key, attr in shape_meta['obs'].items() if attr.get('type', 'low_dim') == 'rgb']
    lowdim_keys = [key for key, attr in shape_meta['obs'].items() if attr.get('type', 'low_dim') == 'low_dim']

    episode_starts, episode_ends, n_steps = _get_episode_ends(dataset_paths)
    
    store = zarr.MemoryStore()
    root = zarr.group(store=store)
    data_group = root.require_group('data', overwrite=True)
    meta_group = root.require_group('meta', overwrite=True)
    _ = meta_group.array('episode_ends', episode_ends, 
        dtype=np.int64, compressor=None, overwrite=True)
    
    for key in tqdm(lowdim_keys + ['action'], desc="Loading lowdim data"):
        data_key = 'obs/' + key if key != 'action' else 'actions'
        shape = tuple(shape_meta['action']['shape']) if key == 'action' else tuple(shape_meta['obs'][key]['shape'])
        this_data_arr = data_group.empty(
            name=key,
            shape=(n_steps,) + shape,
            chunks=(n_steps,) + shape, # single chunk for faster access
            compressor=None,
            dtype=np.float32
        )
        
        demo_idx_offset = 0
        for path in tqdm(dataset_paths, desc=f"Loading {key}", leave=False):
            with h5py.File(path, 'r') as file:
                if 'data' not in file: 
                    continue
                num_demos_in_file = len(file['data'])
                for i in range(num_demos_in_file):
                    demo_key = f'demo_{i}'
                    if demo_key not in file['data']: 
                        continue
                    
                    data_chunk = file[f'data/{demo_key}/{data_key}'][:].astype(np.float32)
                    global_demo_idx = demo_idx_offset + i
                    start_idx = episode_starts[global_demo_idx]
                    end_idx = episode_ends[global_demo_idx]
                    
                    if key == 'action':
                        data_chunk = _convert_actions(data_chunk, abs_action, rotation_transformer)
                    this_data_arr[start_idx:end_idx] = data_chunk
                    
                demo_idx_offset += num_demos_in_file

    _parallel_load_images(data_group, episode_starts, n_steps, dataset_paths, rgb_keys, 'obs', shape_meta, n_workers, max_inflight_tasks)
    
    replay_buffer = ReplayBuffer(root)
    return replay_buffer

def _convert_flow_to_replay(
    shape_meta, 
    flow_dataset_path, 
    n_workers=None, 
    max_inflight_tasks=None
):
    if n_workers is None: 
        n_workers = multiprocessing.cpu_count()
    if max_inflight_tasks is None: 
        max_inflight_tasks = n_workers * 5
        
    episode_starts, episode_ends, n_steps = _get_episode_ends(flow_dataset_paths, load_flow=True, flow_key=flow_keys[0])

    store = zarr.MemoryStore()
    root = zarr.group(store=store)
    data_group = root.require_group('data', overwrite=True)
    meta_group = root.require_group('meta', overwrite=True)
    _ = meta_group.array('episode_ends', episode_ends, 
        dtype=np.int64, compressor=None, overwrite=True)

    if isinstance(flow_dataset_path, str):
        flow_dataset_paths = [flow_dataset_path]
    else: 
        flow_dataset_paths = list(flow_dataset_path)

    flow_keys = [key for key, attr in shape_meta.get('flow', {}).items() if attr.get('type', 'rgb') == 'rgb']

    
    _parallel_load_images(data_group, episode_starts, n_steps, flow_dataset_paths, flow_keys, 'flow', shape_meta, n_workers, max_inflight_tasks)
    
    # Return a buffer-like object (a dict is fine) to be merged
    return ReplayBuffer(root)
