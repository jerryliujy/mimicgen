import os
import torch
import torch.distributed as dist

def main():
    """
    A minimal script to test torch.distributed (DDP) initialization.
    """
    try:
        # torchrun sets these environment variables
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        
        print(f"[Rank {local_rank}] Starting DDP check. World size: {world_size}")

        # This is the critical step for multi-GPU training on NVIDIA hardware
        dist.init_process_group(backend='nccl', rank=local_rank, world_size=world_size)
        
        print(f"[Rank {local_rank}] Successfully initialized process group with backend 'nccl'.")

        # Check if CUDA is available for this process
        if torch.cuda.is_available():
            print(f"[Rank {local_rank}] CUDA is available. Device: {torch.cuda.get_device_name(local_rank)}")
        else:
            print(f"[Rank {local_rank}] CUDA is NOT available.")

        dist.destroy_process_group()
        print(f"[Rank {local_rank}] Cleanly destroyed process group.")

    except Exception as e:
        print(f"[Rank {local_rank}] FAILED with exception: {e}")
        # Re-raise to ensure the process exits with an error code
        raise

if __name__ == "__main__":
    main()
