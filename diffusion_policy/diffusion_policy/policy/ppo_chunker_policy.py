import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List
import numpy as np

from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.common.rl_chunker_net import ActionChunkerNet

class PPOChunkerPolicy(BaseImagePolicy):
    """
    A wrapper policy for PPO training of the ActionChunkerNet.
    """
    def __init__(self, 
                 action_generator: BaseImagePolicy, 
                 action_chunker_net: ActionChunkerNet,
                 clip_param=0.2,
                 ppo_epoch=4,
                 mini_batch_size=64,
                 value_loss_coef=0.5,
                 entropy_coef=0.01):
        super().__init__()
        self.action_generator = action_generator
        self.model = action_chunker_net
        
        # PPO Hyperparameters
        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.mini_batch_size = mini_batch_size
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        
        # Add a Value Head to the Chunker Net (Critic)
        # We assume action_chunker_net has an obs_encoder we can reuse or share features with.
        # For simplicity, let's add a separate value head here or modify ChunkerNet.
        # Let's add it here as a separate module sharing the encoder if possible, 
        # but ChunkerNet encapsulates the encoder.
        # So we should probably add the value head to ChunkerNet or access its encoder.
        
        obs_feature_dim = self.model.obs_encoder.output_dim()
        self.value_head = nn.Sequential(
            nn.Linear(obs_feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Storage for PPO rollouts
        self.rollout_buffer = {
            'obs': [],
            'actions': [],
            'log_probs': [],
            'values': [],
            'rewards': [],
            'dones': []
        }
        
        # We need to store rewards temporarily during rollout
        self.rewards = [] 

    def predict_action(self, obs_dict):
        # 1. Get full action sequence from frozen generator
        self.action_generator.eval()
        with torch.no_grad():
            generator_output = self.action_generator.predict_action(obs_dict)
        full_action_chunk = generator_output['action']

        # 2. Get Chunker decision
        self.model.train() # Keep in train mode for exploration (dropout etc if any)
        
        # We need value estimate for PPO
        # We need to run forward pass to get features first?
        # ChunkerNet.forward calls obs_encoder.
        # We need to access features to run value_head.
        # Let's modify ChunkerNet or just run encoder twice (inefficient but simple).
        # Or better: ChunkerNet exposes a method to get features?
        # For now, let's assume we can run encoder.
        
        obs_features = self.model.obs_encoder(obs_dict)
        logits = self.model.decision_head(obs_features)
        value = self.value_head(obs_features)
        
        dist = F.softmax(logits, dim=-1)
        cat_dist = torch.distributions.Categorical(dist)
        
        step_choice_idx = cat_dist.sample()
        log_prob = cat_dist.log_prob(step_choice_idx)
        
        # Store rollout data
        # We need to store obs_dict, but it's a dict of tensors.
        # Storing it might consume memory.
        # For PPO, we need to re-evaluate action probabilities on the batch.
        # So we need the observations.
        
        # Store data for PPO update
        # Note: This is called per step (or per batch of envs).
        # If obs_dict is a batch, we append the batch.
        self.rollout_buffer['obs'].append(dict_apply(obs_dict, lambda x: x.detach().cpu()))
        self.rollout_buffer['actions'].append(step_choice_idx.detach().cpu())
        self.rollout_buffer['log_probs'].append(log_prob.detach().cpu())
        self.rollout_buffer['values'].append(value.detach().cpu())
        
        # Construct execution action
        execution_steps_batch = [self.model.get_execution_step_count(idx) for idx in step_choice_idx]
        max_steps = max(execution_steps_batch)
        
        batch_size = full_action_chunk.shape[0]
        action_dim = full_action_chunk.shape[-1]
        
        action_to_execute = torch.zeros((batch_size, max_steps, action_dim), device=self.device, dtype=self.dtype)
        
        for i in range(batch_size):
            steps = execution_steps_batch[i]
            action_to_execute[i, :steps, :] = full_action_chunk[i, :steps, :]
            if steps < max_steps:
                action_to_execute[i, steps:, :] = full_action_chunk[i, steps-1, :].unsqueeze(0)

        return {'action': action_to_execute}

    def update(self, optimizer, gamma):
        # PPO Update Logic
        # 1. Process rewards and compute advantages
        # self.rewards contains lists of rewards [tensor(B), tensor(B), ...]
        # We need to flatten/stack them.
        
        if not self.rewards:
            return 0.0
            
        rewards = torch.stack(self.rewards).to(self.device) # (T, B)
        values = torch.stack(self.rollout_buffer['values']).to(self.device).squeeze(-1) # (T, B)
        log_probs = torch.stack(self.rollout_buffer['log_probs']).to(self.device) # (T, B)
        actions = torch.stack(self.rollout_buffer['actions']).to(self.device) # (T, B)
        
        # Compute Returns and Advantages (GAE)
        # For simplicity, let's use simple Monte Carlo returns first or GAE if we had next_value.
        # We don't have next_value easily here without modifying runner loop.
        # Let's use simple discounted returns.
        
        returns = torch.zeros_like(rewards)
        R = 0
        for t in reversed(range(rewards.shape[0])):
            R = rewards[t] + gamma * R
            returns[t] = R
            
        advantages = returns - values
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Flatten for batch processing
        # (T, B, ...) -> (T*B, ...)
        # But obs is a list of dicts. We need to collate them.
        
        # PPO Epochs
        total_loss = 0
        
        # Flatten data
        # obs is a list of dicts. We need to stack them.
        # Assuming obs_dict structure is consistent.
        # We need a helper to stack list of dicts of tensors into dict of tensors.
        
        # Helper to stack list of dicts
        def stack_dict_list(dict_list):
            result = dict()
            for k in dict_list[0].keys():
                if isinstance(dict_list[0][k], dict):
                    result[k] = stack_dict_list([d[k] for d in dict_list])
                else:
                    result[k] = torch.cat([d[k] for d in dict_list], dim=0)
            return result

        # Flatten obs: (T, B, ...) -> (T*B, ...)
        # Our buffer stores list of (B, ...) dicts.
        # stack_dict_list will concat them along dim 0, effectively flattening T and B.
        flat_obs = stack_dict_list(self.rollout_buffer['obs'])
        # Move to device
        flat_obs = dict_apply(flat_obs, lambda x: x.to(self.device))
        
        flat_actions = actions.view(-1)
        flat_log_probs = log_probs.view(-1)
        flat_advantages = advantages.view(-1)
        flat_returns = returns.view(-1)
        
        dataset_size = flat_actions.shape[0]
        indices = np.arange(dataset_size)
        
        for _ in range(self.ppo_epoch):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, self.mini_batch_size):
                end = start + self.mini_batch_size
                mb_indices = indices[start:end]
                
                mb_obs = dict_apply(flat_obs, lambda x: x[mb_indices])
                mb_actions = flat_actions[mb_indices]
                mb_old_log_probs = flat_log_probs[mb_indices]
                mb_advantages = flat_advantages[mb_indices]
                mb_returns = flat_returns[mb_indices]
                
                # Forward pass
                obs_features = self.model.obs_encoder(mb_obs)
                logits = self.model.decision_head(obs_features)
                new_values = self.value_head(obs_features).squeeze(-1)
                
                dist = F.softmax(logits, dim=-1)
                cat_dist = torch.distributions.Categorical(dist)
                new_log_probs = cat_dist.log_prob(mb_actions)
                entropy = cat_dist.entropy().mean()
                
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * mb_advantages
                action_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = (mb_returns - new_values).pow(2).mean()
                
                loss = action_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()

        # Clear buffers
        self.rollout_buffer = {k: [] for k in self.rollout_buffer}
        self.rewards = []
        
        return total_loss / (self.ppo_epoch * (dataset_size // self.mini_batch_size + 1))

    def set_normalizer(self, normalizer):
        self.action_generator.set_normalizer(normalizer)
        
def dict_apply(d, func):
    result = dict()
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = dict_apply(v, func)
        else:
            result[k] = func(v)
    return result
