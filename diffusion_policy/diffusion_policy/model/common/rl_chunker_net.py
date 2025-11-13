import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class ActionChunkerNet(nn.Module):
    def __init__(self, 
                 obs_encoder: nn.Module,
                 n_action_steps: int):
        """
        A network that decides how many steps of an action chunk to execute.

        Args:
            obs_encoder (nn.Module): A pre-built observation encoder.
            n_action_steps (int): The maximum number of steps to choose from.
                                  Choices will be [1, 2, ..., n_action_steps].
        """
        super().__init__()
        self.obs_encoder = obs_encoder
        
        # Get the feature dimension from the observation encoder
        obs_feature_dim = obs_encoder.output_dim()

        # Create the list of possible execution steps
        execution_steps = list(range(1, n_action_steps + 1))

        # Simple MLP head to decide on the number of steps
        self.decision_head = nn.Sequential(
            nn.Linear(obs_feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, len(execution_steps))
        )
        
        self.execution_steps = execution_steps
        self.num_choices = len(execution_steps)

    def forward(self, obs_dict):
        """
        Forward pass to get the logits for each execution step choice.

        Args:
            obs_dict (dict): A dictionary of observations.

        Returns:
            torch.Tensor: Logits for each choice in execution_steps.
        """
        # Encode the observation
        obs_features = self.obs_encoder(obs_dict)
        
        # Get logits from the decision head
        logits = self.decision_head(obs_features)
        return logits

    def get_action(self, obs_dict):
        """
        Selects an execution step count based on the observation.

        Args:
            obs_dict (dict): A dictionary of observations.

        Returns:
            tuple: (
                step_choice_idx (torch.Tensor): Index of the chosen step count.
                log_prob (torch.Tensor): Log probability of the chosen action.
                dist (torch.distributions.Categorical): The full probability distribution.
            )
        """
        logits = self.forward(obs_dict)
        dist = F.softmax(logits, dim=-1)
        cat_dist = torch.distributions.Categorical(dist)
        
        step_choice_idx = cat_dist.sample()
        log_prob = cat_dist.log_prob(step_choice_idx)
        
        return step_choice_idx, log_prob, cat_dist

    def get_execution_step_count(self, step_choice_idx):
        """
        Converts a choice index to the actual number of steps.

        Args:
            step_choice_idx (int or torch.Tensor): The index of the choice.

        Returns:
            int: The number of steps to execute.
        """
        if isinstance(step_choice_idx, torch.Tensor):
            step_choice_idx = step_choice_idx.item()
        return self.execution_steps[step_choice_idx]

