import numpy as np

import sys
import os
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import cv2

class InputPadder:
    """Pads images so that dimensions are divisible by a factor."""
    def __init__(self, dims, factor=8):
        self.ht, self.wd = dims[-2:]
        pad_ht = (factor - self.ht % factor) % factor
        pad_wd = (factor - self.wd % factor) % factor
        # (pad_left, pad_right, pad_top, pad_bottom)
        self._pad = (pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2)

    def pad(self, x: torch.Tensor) -> torch.Tensor:
        return F.pad(x, self._pad, mode='replicate')

    def unpad(self, x: torch.Tensor) -> torch.Tensor:
        ht, wd = x.shape[-2:]
        c = (self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1])
        return x[..., c[0]:c[1], c[2]:c[3]]


class FlowPredictor:
    def __init__(self):
        self.ckpt_path = '/home/workspace/diffusion_policy/ckpts/things_288960.pth'
        self.device = 'cuda:0'


    def predict(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        return self.wrapper.predict(img1, img2)

    def predict_video(self, video_path: str) -> list:
        return self.wrapper.predict_video(video_path)


class FlowFormerWrapper(FlowPredictor):
    """Wrapper for FlowFormer++ model to predict optical flow."""
    def __init__(self):
        super().__init__()
        
        flowformer_dir = '/home/workspace/FlowFormerPlusPlus'
        sys.path.append(flowformer_dir)
        try:
            from core.FlowFormer import build_flowformer
            from configs.things import get_cfg   # we use things as default
        except Exception as e:
            raise ImportError('FlowFormer++ not available.')

        flow_cfg = get_cfg()
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
        t1 = padder.pad(t1)
        t2 = padder.pad(t2)
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


def get_flow_predictor() -> FlowPredictor:
    return FlowFormerWrapper()


def generate_flow_from_frames(frames: np.ndarray, flow_estimator, interval: int = 1, target_size: int = 224) -> np.ndarray:
    T, H, W, C = frames.shape
    flows = []
    for t in range(T):
        t2 = min(t + interval, T - 1)
        f1 = frames[t]
        f2 = frames[t2]
        f1 = cv2.resize(f1, dsize=(target_size, target_size), interpolation=cv2.INTER_CUBIC)
        f2 = cv2.resize(f2, dsize=(target_size, target_size), interpolation=cv2.INTER_CUBIC)
        uv = flow_estimator.predict(f1, f2)
        uv = cv2.resize(uv, dsize=(H, W), interpolation=cv2.INTER_CUBIC)
        uv = np.stack([uv[..., 0], uv[..., 1], (uv[..., 0] + uv[..., 1]) / 2], axis=-1)  # add the third channel
        flows.append(uv)
    return np.stack(flows, axis=0)



