if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import tqdm
import numpy as np
import random

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.common.rl_chunker_net import ActionChunkerNet
from diffusion_policy.policy.rl_dp_policy import RLDPPolicy

OmegaConf.register_new_resolver("eval", eval, replace=True)


class TrainRLWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # 1. Load pre-trained, frozen action generator
        print("Loading pre-trained action generator policy...")
        self.action_generator: BaseImagePolicy = hydra.utils.instantiate(cfg.action_generator_policy)
        # Load checkpoint
        ckpt_path = cfg.training.generator_ckpt_path
        print(f"Loading checkpoint from {ckpt_path}")
        payload = torch.load(open(ckpt_path, 'rb'), map_location='cpu')
        self.action_generator.load_state_dict(payload['model'])
        # Freeze the generator
        for param in self.action_generator.parameters():
            param.requires_grad = False

        # 2. Configure the model to be trained (ActionChunkerNet)
        # It reuses the obs_encoder from the frozen generator
        action_chunker_net = ActionChunkerNet(
            obs_encoder=self.action_generator.model.obs_encoder,
            n_action_steps=self.action_generator.n_action_steps
        )
        
        # 3. Create the wrapper policy for RL
        self.model = RLDPPolicy(
            action_generator=self.action_generator,
            action_chunker_net=action_chunker_net
        )

        # 4. Configure optimizer for the chunker network parameters
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.model.parameters())

        # configure training state
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = self.cfg

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        self.action_generator.to(device)
        optimizer_to(self.optimizer, device)

        # Get normalizer from dataset
        dataset: BaseImageDataset = hydra.utils.instantiate(cfg.task.dataset)
        normalizer = dataset.get_normalizer()
        self.model.set_normalizer(normalizer)

        # configure env runner
        env_runner: BaseImageRunner = hydra.utils.instantiate(
            cfg.task.env_runner, output_dir=self.output_dir)

        # configure logging
        wandb_run = None
        if self.is_master:
            import wandb
            wandb_run = wandb.init(
                dir=str(self.output_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
                **cfg.logging
            )

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for self.epoch in tqdm.tqdm(range(cfg.training.num_epochs), desc="Training Epochs"):
                
                # --- Main RL Interaction and Update Step ---
                # The runner will interact with the env and populate policy buffers
                runner_log = env_runner.run(self.model)
                
                # Perform the REINFORCE update
                policy_loss = self.model.update(self.optimizer, cfg.training.gamma)
                
                # --- Logging ---
                step_log = {
                    'policy_loss': policy_loss,
                    'epoch': self.epoch,
                }
                step_log.update(runner_log)
                
                if self.is_master:
                    wandb_run.log(step_log)
                    json_logger.log(step_log)

                # checkpoint
                if cfg.checkpoint.save_ckpt:
                    if hasattr(cfg.checkpoint, 'save_specific_steps') and \
                        self.epoch in cfg.checkpoint.save_specific_steps:
                        self.save_checkpoint(tag=f"step_{self.epoch}")
                    elif (self.epoch % cfg.training.checkpoint_every) == 0 and cfg.checkpoint.save_ckpt:
                        # checkpointing
                        if cfg.checkpoint.save_last_ckpt:
                            self.save_checkpoint()
                        if cfg.checkpoint.save_last_snapshot:
                            self.save_snapshot()

                        # sanitize metric names
                        metric_dict = dict()
                        for key, value in step_log.items():
                            new_key = key.replace('/', '_')
                            metric_dict[new_key] = value
                            
                # end of epoch
                # log of last step is combined with validation and rollout
                if self.is_master:
                    wandb_run.log(step_log, step=self.global_step)
                    json_logger.log(step_log)
                self.epoch += 1

        print("RL training finished.")


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name="train_rl_workspace.yaml") # A new config file for this workspace
def main(cfg):
    workspace = TrainRLWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()