import numpy as np

import sys
import os
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import cv2


class InputPadder:
    """Pad images so that dimensions are divisible by 8."""
    def __init__(self, shape):
        ht, wd = shape[0], shape[1]
        pad_ht = (8 - ht % 8) % 8
        pad_wd = (8 - wd % 8) % 8
        self.pad = (pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2)

    def pad_img(self, img: torch.Tensor) -> torch.Tensor:
        return F.pad(img, self.pad, mode='replicate')

    def unpad(self, flow: torch.Tensor) -> torch.Tensor:
        x1, x2, y1, y2 = self.pad
        # Remove padding to restore original dimensions
        _, _, h, w = flow.shape
        return flow[..., y1:h - y2, x1:w - x2]


class FlowPredictor:
    def __init__(self, cfg):
        self.ckpt_path, self.config, self.device = (
            cfg.flow_predictor.ckpt_path, cfg.flow_predictor.config, cfg.device
        )
        if self.device.startswith('cuda') and not torch.cuda.is_available():
            self.device = 'cpu'

    def predict(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        return self.wrapper.predict(img1, img2)

    def predict_video(self, video_path: str) -> list:
        return self.wrapper.predict_video(video_path)


class FlowFormerWrapper(FlowPredictor):
    """Wrapper for FlowFormer++ model to predict optical flow."""
    def __init__(self, cfg):
        super().__init__(cfg)
         
        # sys.path.insert(0, str(base_dir))
        try:
            from FlowFormerPlusPlus.core.FlowFormer import build_flowformer
            from FlowFormerPlusPlus.configs.things import get_cfg 
        except Exception as e:
            raise ImportError('FlowFormer++ not available.')

        flow_cfg = get_cfg() if self.config == 'things' else None
        if flow_cfg is None:
            raise ValueError(f'Unsupported config: {self.config}')
        model = build_flowformer(flow_cfg)
        model = torch.nn.DataParallel(model).to(self.device)
        if not os.path.isfile(self.ckpt_path):
            raise FileNotFoundError(f'Checkpoint not found: {self.ckpt_path}')
        state = torch.load(self.ckpt_path, map_location=self.device)
        model.load_state_dict(state)
        model.eval()
        self.model = model

    @torch.no_grad()
    def predict(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        # Convert BGR to RGB, HWC to CHW, to tensor
        # Make sure array is contiguous (avoid negative strides)
        arr1 = img1[..., ::-1].transpose(2,0,1)
        t1 = torch.from_numpy(np.ascontiguousarray(arr1)).float()[None].to(self.device)
        arr2 = img2[..., ::-1].transpose(2,0,1)
        t2 = torch.from_numpy(np.ascontiguousarray(arr2)).float()[None].to(self.device)
        padder = InputPadder(img1.shape[:2])
        t1 = padder.pad_img(t1)
        t2 = padder.pad_img(t2)
        outs = self.model(t1, t2)
        # If model returns multiple predictions, take the last one
        if isinstance(outs, (list, tuple)):
            flow_up = outs[-1]
        else:
            flow_up = outs
        # Remove padding to original padded resolution
        flow = padder.unpad(flow_up)
        # Upsample flow to original image resolution
        orig_h, orig_w = img1.shape[:2]
        # flow shape: [1, 2, h_coarse, w_coarse]
        flow = F.interpolate(flow, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
        # Scale flow values according to upsampling factor
        factor_h = orig_h / flow_up.shape[-2]
        factor_w = orig_w / flow_up.shape[-1]
        flow = flow * torch.tensor([factor_w, factor_h], device=flow.device)[None, :, None, None]
        # Convert CHW to HWC and numpy
        return flow[0].permute(1,2,0).cpu().numpy()

    def predict_video(self, video_path: str) -> list:
        cap = cv2.VideoCapture(video_path)
        flows = []
        ret, prev = cap.read()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            flow = self.predict(prev, frame)
            flows.append(flow)
            prev = frame
        cap.release()
        return flows


def get_flow_predictor(cfg) -> FlowPredictor:
    if cfg.flow_predictor.name == 'flowformer++':
        # Use the FlowFormerWrapper to properly initialize model
        return FlowFormerWrapper(cfg)
    else:
        raise ValueError(f"Unsupported flow predictor name: {cfg.flow_predictor.name}")


def generate_flow_from_frames(frames: np.ndarray, flow_estimator, interval: int = 1) -> np.ndarray:
    T, C, H, W = frames.shape
    flows = []
    for t in range(T):
        t2 = min(t + interval, T - 1)
        f1 = frames[t].transpose(1, 2, 0)
        f2 = frames[t2].transpose(1, 2, 0)
        uv = flow_estimator.predict(f1, f2)
        flows.append(uv.transpose(2, 0, 1)) 
    return np.stack(flows, axis=0)



