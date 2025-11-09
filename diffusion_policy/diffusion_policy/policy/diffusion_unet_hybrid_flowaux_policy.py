from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.base_nets as rmbn
import diffusion_policy.model.vision.crop_randomizer as dmvc
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules


class DiffusionUnetHybridFlowauxPolicy(BaseImagePolicy):
    """
        This is a diffusion UNet policy that predicts both action and flow
        The gt_action is passed through VAE to get latent distribution,
        then diffusion model samples latent representation, denoises it, and reconstructs action and flow as auxiliary task.
    """
    
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            flow_decoder,
            action_vae,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            action_emb_dim=256,  # the latent dim of action encoder
            num_inference_steps=None,
            obs_as_global_cond=False,
            crop_shape=(76, 76),
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            obs_encoder_group_norm=False,
            eval_fixed_crop=False,
            flow_weight=0.1,
            action_weight=0.1,
            # parameters passed to step
            **kwargs):
        super().__init__()

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta['obs']
        obs_config = {
            'low_dim': [],
            'rgb': [],
            'depth': [],
            'scan': []
        }
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)

            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                obs_config['rgb'].append(key)
            elif type == 'low_dim':
                obs_config['low_dim'].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        # get raw robomimic config
        config = get_robomimic_config(
            algo_name='bc_rnn',
            hdf5_type='image',
            task_name='square',
            dataset_type='ph')
        
        with config.unlocked():
            # set config with shape_meta
            config.observation.modalities.obs = obs_config

            if crop_shape is None:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality['obs_randomizer_class'] = None
            else:
                # set random crop parameter
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw

        # init global state
        ObsUtils.initialize_obs_utils_with_config(config)

        # load model
        policy: PolicyAlgo = algo_factory(
                algo_name=config.algo_name,
                config=config,
                obs_key_shapes=obs_key_shapes,
                ac_dim=action_dim,
                device='cpu',
            )

        obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']
        
        if obs_encoder_group_norm:
            # replace batch norm with group norm
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features//16, 
                    num_channels=x.num_features)
            )
            # obs_encoder.obs_nets['agentview_image'].nets[0].nets
        
        # obs_encoder.obs_randomizers['agentview_image']
        if eval_fixed_crop:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
                func=lambda x: dmvc.CropRandomizer(
                    input_shape=x.input_shape,
                    crop_height=x.crop_height,
                    crop_width=x.crop_width,
                    num_crops=x.num_crops,
                    pos_enc=x.pos_enc
                )
            )

        # create diffusion model
        obs_feature_dim = obs_encoder.output_shape()[0]
        input_dim = action_emb_dim
        global_cond_dim = None
        if obs_as_global_cond:
            global_cond_dim = obs_feature_dim * n_obs_steps
        else:
            input_dim += obs_feature_dim

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )
        
        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.flow_decoder = flow_decoder
        self.action_vae = action_vae
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_emb_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_emb_dim = action_emb_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.action_weight = action_weight
        self.flow_weight = flow_weight
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        print("Diffusion params: %e" % sum(p.numel() for p in self.model.parameters()))
        print("Vision params: %e" % sum(p.numel() for p in self.obs_encoder.parameters()))
        print("Flow decoder params: %e" % sum(p.numel() for p in self.flow_decoder.parameters()))
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict 
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Do = self.obs_feature_dim
        action_emb_dim = self.action_emb_dim

        device = self.device
        dtype = self.dtype

        global_cond = None
        if self.obs_as_global_cond:
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            global_cond = nobs_features.reshape(B, -1)
            
            noisy_z = torch.randn((B, T, action_emb_dim), device=device, dtype=dtype)
            
            self.noise_scheduler.set_timesteps(self.num_inference_steps)
            for t in self.noise_scheduler.timesteps:
                model_output = self.model(noisy_z, t, global_cond=global_cond)
                noisy_z = self.noise_scheduler.step(model_output, t, noisy_z, **self.kwargs).prev_sample
            predicted_z = noisy_z
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            nobs_features = nobs_features.reshape(B, To, -1)

            cond_data = torch.zeros(size=(B, T, action_emb_dim + Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            
            cond_data[:,:To,action_emb_dim:] = nobs_features
            cond_mask[:,:To,action_emb_dim:] = True

            predicted_trajectory = self.conditional_sample(
                cond_data, 
                cond_mask,
                global_cond=None,
                **self.kwargs)
            
            predicted_z = predicted_trajectory[..., :action_emb_dim]

        with torch.no_grad():
            naction_pred = self.action_vae.decode(predicted_z)
        
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        nflow = self.normalizer['flow'].normalize(batch['flow']) if batch.get('flow', None) is not None else None
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]
        n_obs_steps = self.n_obs_steps
        
        # encode action to latent space
        with torch.no_grad():
            z = self.action_vae.encode(nactions.reshape(batch_size * horizon, -1))
        z = z.reshape(batch_size, horizon, -1)

        global_cond = None
        trajectory = z
        cond_data = trajectory
        if self.obs_as_global_cond:
            # global cond uses film conditioning
            this_nobs = dict_apply(nobs, lambda x: x[:,:n_obs_steps,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            # co-denoise action embedding and obs features
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([z, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        condition_mask = self.mask_generator(trajectory.shape)
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        
        timesteps = torch.randint(0, 
            self.noise_scheduler.config.num_train_timesteps, 
            (batch_size,), 
            device=trajectory.device
        ).long()
        
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)
        
        noisy_trajectory[condition_mask] = cond_data[condition_mask]
        
        pred = self.model(noisy_trajectory, timesteps, global_cond=global_cond)

        loss_mask = ~condition_mask
        diffusion_loss = F.mse_loss(pred[loss_mask], noise[loss_mask], reduction='mean')
        
        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            predicted_trajectory = trajectory + pred
        elif pred_type == 'sample':
            predicted_trajectory = pred
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        if self.obs_as_global_cond:
            predicted_z = predicted_trajectory
            predicted_obs_feat = predicted_z
        else:
            action_emb_dim = z.shape[-1]
            predicted_z = predicted_trajectory[..., :action_emb_dim]
            predicted_obs_feat = predicted_trajectory[..., action_emb_dim:]
            
        # action loss
        with torch.no_grad():
            recon_action = self.action_vae.decode(predicted_z)
        recon_a_loss = F.mse_loss(recon_action, nactions, reduction='mean')
        
        # flow loss 
        recon_flow = self.flow_decoder(predicted_obs_feat)
        recon_flow = recon_flow[:, :n_obs_steps]
        recon_f_loss = F.l1_loss(recon_flow, nflow, reduction='mean')
        
        total_loss = diffusion_loss + self.action_weight * recon_a_loss + self.flow_weight * recon_f_loss
        
        loss_dict = {
            'diffusion_loss': diffusion_loss,
            'recon_action_loss': recon_a_loss,
            'recon_flow_loss': recon_f_loss,
            'total_loss': total_loss
        }
        return loss_dict