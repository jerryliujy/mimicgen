"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf
import pathlib
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.dataset.robomimic_replay_image_dataset import RobomimicReplayImageDataset

import os
import torch
import torch.distributed as dist

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy','config'))
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    
    if dist.is_initialized():
        print(f"[Rank {local_rank}] Process group successfully initialized in train.py!")
    else:
        print(f"[Rank {local_rank}] FATAL: Process group NOT initialized in train.py!")
        # Exit early if initialization failed
        return

    sys.path.append(get_original_cwd())
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(
        cfg=cfg,
        local_rank=local_rank
    )
    # workspace: BaseWorkspace = hydra.utils.instantiate(
    #     cfg._target_,
    #     cfg=cfg,
    #     local_rank=local_rank  
    # )
    workspace.run()

    dist.destroy_process_group()
    
if __name__ == "__main__":
    main()
