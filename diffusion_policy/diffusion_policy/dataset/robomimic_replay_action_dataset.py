from typing import Dict, List
import torch
import numpy as np
import h5py
from tqdm import tqdm
import os
import hydra
from filelock import FileLock
import multiprocessing

from torch.utils.data import Dataset
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.common.sampler import ActionSequenceSampler, get_val_mask
from diffusion_policy.common.normalize_util import (
    robomimic_abs_action_only_normalizer_from_stat,
    get_identity_normalizer_from_stat,
    array_to_stats
)

def _convert_actions(raw_actions, abs_action, rotation_transformer):
    """Helper function to process raw actions from robomimic."""
    actions = raw_actions
    if abs_action:
        is_dual_arm = False
        if raw_actions.shape[-1] == 14:
            raw_actions = raw_actions.reshape(-1, 2, 7)
            is_dual_arm = True

        pos = raw_actions[..., :3]
        rot = raw_actions[..., 3:6]
        gripper = raw_actions[..., 6:]
        rot = rotation_transformer.forward(rot)
        raw_actions = np.concatenate([pos, rot, gripper], axis=-1).astype(np.float32)
    
        if is_dual_arm:
            raw_actions = raw_actions.reshape(-1, 20)
        actions = raw_actions
    return actions

class RobomimicReplayActionDataset(Dataset):
    """
    A lightweight dataset that loads only action sequences from Robomimic datasets
    into memory for fast VAE training.
    """
    def __init__(self,
            shape_meta: dict,
            dataset_path: str,
            horizon=1,
            pad_before=0,
            pad_after=0,
            abs_action=False,
            rotation_rep='rotation_6d',
            seed=42,
            val_ratio=0.0
        ):
        super().__init__()

        # ========= 1. Load all actions into memory =========
        rotation_transformer = RotationTransformer(from_rep='axis_angle', to_rep=rotation_rep)
        
        if isinstance(dataset_path, str):
            dataset_paths = [dataset_path]
        else:
            dataset_paths = list(dataset_path)

        resolved_dataset_paths = []
        try:
            original_cwd = hydra.utils.get_original_cwd()
            for path in dataset_paths:
                resolved_dataset_paths.append(os.path.join(original_cwd, path))
        except (ValueError, ImportError):
            resolved_dataset_paths = dataset_paths

        all_actions = []
        episode_ends = []
        n_steps = 0
        
        print(f"Loading actions from {len(resolved_dataset_paths)} files into memory...")
        for path in tqdm(resolved_dataset_paths, desc="Loading Actions"):
            with h5py.File(path, 'r') as file:
                if 'data' not in file:
                    continue
                demos = file['data']
                for i in range(len(demos)):
                    demo_key = f'demo_{i}'
                    if demo_key not in demos:
                        continue
                    
                    actions_chunk = demos[demo_key]['actions'][:]
                    converted_actions = _convert_actions(actions_chunk, abs_action, rotation_transformer)
                    all_actions.append(torch.from_numpy(converted_actions).float())
                    
                    episode_length = len(actions_chunk)
                    episode_ends.append(n_steps + episode_length)
                    n_steps += episode_length
        
        self.actions_in_memory = torch.cat(all_actions, dim=0)
        print(f"Loaded {self.actions_in_memory.shape[0]} total action steps.")

        # ========= 2. Create a sampler =========
        val_mask = get_val_mask(
            n_episodes=len(episode_ends), 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask

        self.sampler = ActionSequenceSampler(
            episode_ends=np.array(episode_ends), 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask
        )

        # ========= 3. Store metadata =========
        self.shape_meta = shape_meta
        self.horizon = horizon
        self.abs_action = abs_action
        self.train_mask = train_mask

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Sampler returns episode_idx and local_idx
        episode_idx, local_idx = self.sampler.idx_to_local_idx(idx)
        
        # Calculate start and end index in the flat actions_in_memory tensor
        start_idx = self.sampler.episode_starts[episode_idx] + local_idx
        end_idx = start_idx + self.horizon
        
        # Slice directly from the in-memory tensor
        action_sequence = self.actions_in_memory[start_idx:end_idx]
        
        return {'action': action_sequence.float()}

    def get_validation_dataset(self):
        val_set = self
        val_set.sampler = ActionSequenceSampler(
            episode_ends=self.sampler.episode_ends, 
            sequence_length=self.horizon,
            pad_before=self.sampler.pad_before, 
            pad_after=self.sampler.pad_after,
            episode_mask=~self.train_mask
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self) -> LinearNormalizer:
        # This dataset is specialized for VAE training, where normalization
        # might be handled differently or not needed if actions are in a known range.
        # We return a simple identity normalizer.
        normalizer = LinearNormalizer()
        stat = array_to_stats(self.actions_in_memory.numpy())
        
        if self.abs_action:
            this_normalizer = robomimic_abs_action_only_normalizer_from_stat(stat)
        else:
            this_normalizer = get_identity_normalizer_from_stat(stat)
        normalizer['action'] = this_normalizer
        return normalizer

