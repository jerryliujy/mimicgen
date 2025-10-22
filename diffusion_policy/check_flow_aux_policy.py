
import torch
import hydra
import os
from omegaconf import OmegaConf
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.dataset.base_dataset import BaseImageDataset

# supress hydra's future warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="hydra")

@hydra.main(
    version_base=None,
    config_path=os.path.join(os.path.dirname(__file__), 'diffusion_policy/config'), 
    config_name='train_diffusion_unet_flowaux_ddim_hybrid_workspace.yaml'
)
def main(cfg: OmegaConf):
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg=cfg)
    
    dataset: BaseImageDataset
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    normalizer = dataset.get_normalizer()
    
    # Get the policy from the workspace
    policy = workspace.model
    print("Policy instantiated successfully:", type(policy).__name__)
    policy.set_normalizer(normalizer)

    # --- Create a dummy batch of data ---
    # These dimensions are based on the 'can_image' task config
    batch_size = 2
    horizon = policy.horizon
    obs_horizon = policy.n_obs_steps
    action_dim = policy.action_dim
    
    # Dummy observations
    obs_dict = {
        'agentview_image': torch.randn(batch_size, obs_horizon, 3, 84, 84, device=policy.device),
        'robot0_eye_in_hand_image': torch.randn(batch_size, obs_horizon, 3, 84, 84, device=policy.device),
        'robot0_eef_pos': torch.randn(batch_size, obs_horizon, 3, device=policy.device),
        'robot0_eef_quat': torch.randn(batch_size, obs_horizon, 4, device=policy.device),
        'robot0_gripper_qpos': torch.randn(batch_size, obs_horizon, 2, device=policy.device)
    }
    
    # Dummy ground truth actions and flow
    gt_action = torch.randn(batch_size, horizon, action_dim, device=policy.device)
    flow = torch.randn(batch_size, horizon, 2, 84, 84, device=policy.device)

    batch = {
        'obs': obs_dict,
        'action': gt_action,
        'flow': flow
    }
    
    print("\nCreated a dummy batch with the following shapes:")
    print(f"agentview_image: {batch['obs']['agentview_image'].shape}")
    print(f"action: {batch['action'].shape}")
    print(f"flow: {batch['flow'].shape}")

    # --- Run a single forward and backward pass ---
    try:
        print("\nAttempting to run compute_loss...")
        loss_dict = policy.compute_loss(batch)
        loss = loss_dict['total_loss']
        
        print("compute_loss() successful!")
        print("Loss dict:", {k: v.item() for k, v in loss_dict.items()})
        
        print("\nAttempting to run backward pass...")
        loss.backward()
        print("Backward pass successful!")
        
        # Check if gradients are present in a sample parameter
        sample_param = next(policy.model.parameters())
        if sample_param.grad is not None:
            print("Gradients are present in model parameters.")
        else:
            print("Warning: Gradients are None after backward pass.")

    except Exception as e:
        print(f"\nAn error occurred during the test run: {e}")
        import traceback
        traceback.print_exc()

    print("\nTest script finished.")


if __name__ == '__main__':
    main()
