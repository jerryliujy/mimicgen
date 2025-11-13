import torch
import torch.nn as nn
from typing import Dict, List

from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.common.rl_chunker_net import ActionChunkerNet

class RLDPPolicy(BaseImagePolicy):
    """
    A wrapper policy for RL training of the ActionChunkerNet.
    It uses a pre-trained, frozen policy to generate actions and
    an ActionChunkerNet to decide the execution length.
    """
    def __init__(self, action_generator: BaseImagePolicy, action_chunker_net: ActionChunkerNet):
        super().__init__()
        self.action_generator = action_generator
        self.model = action_chunker_net # The network to be trained
        
        # Buffers for REINFORCE
        self.saved_log_probs = []
        self.rewards = []

    def predict_action(self, obs_dict):
        # 1. Get full action sequence from the frozen generator
        self.action_generator.eval()
        with torch.no_grad():
            generator_output = self.action_generator.predict_action(obs_dict)
        full_action_chunk = generator_output['action']

        # 2. Sample execution length from the chunker network
        self.model.train() # Ensure chunker is in train mode for sampling
        step_choice_idx, log_prob, _ = self.model.get_action(obs_dict)
        self.saved_log_probs.append(log_prob)
        
        execution_steps = self.model.get_execution_step_count(step_choice_idx)
        action_to_execute = full_action_chunk[:, :execution_steps, :]

        return {'action': action_to_execute}

    def update(self, optimizer, gamma):
        """Performs the REINFORCE policy gradient update."""
        if not self.saved_log_probs:
            return 0.0

        R = 0
        policy_loss = []
        returns = []
        for r in self.rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns, device=self.model.device)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-6)
        
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        
        optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        optimizer.step()
        
        loss_val = policy_loss.item()
        del self.rewards[:]
        del self.saved_log_probs[:]
        return loss_val

    def set_normalizer(self, normalizer):
        self.action_generator.set_normalizer(normalizer)