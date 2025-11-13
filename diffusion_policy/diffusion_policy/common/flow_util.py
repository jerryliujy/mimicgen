import numpy as np

import sys
import os
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from tqdm import tqdm

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
        self.ckpt_path = '/workspace/mimicgen/diffusion_policy/data/checkpoints/things_288960.pth'
        self.device = 'cuda:0'


    def predict(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        return self.wrapper.predict(img1, img2)

    def predict_video(self, video_path: str) -> list:
        return self.wrapper.predict_video(video_path)


class FlowFormerWrapper(FlowPredictor):
    """Wrapper for FlowFormer++ model to predict optical flow."""
    def __init__(self):
        super().__init__()
        
        flowformer_dir = '/workspace/mimicgen/FlowFormerPlusPlus'
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


def generate_flow_from_frames(frames, flow_estimator, interval=12, target_size=224, progress_desc=None):
    num_frames = len(frames)
    
    h, w = frames.shape[1], frames.shape[2]
    if target_size is not None:
        if w < h:
            new_w = target_size
            new_h = int(h * (new_w / w))
        else:
            new_h = target_size
            new_w = int(w * (new_h / h))
        th, tw = new_h, new_w
    else:
        th, tw = h, w

    res_frames = np.array([cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA) for img in frames])
    flow_arr = np.zeros((num_frames, h, w, 2), dtype=np.float32)
    
    # Create an iterable with a progress bar if requested
    frame_indices = range(num_frames - interval)
    frame_indices = tqdm(frame_indices, desc=progress_desc or "Generating Flow", leave=False)

    for i in frame_indices:
        frame1 = res_frames[i]
        frame2 = res_frames[i+interval]
        flow = flow_estimator.predict(frame1, frame2)
        # resize
        flow = cv2.resize(flow, (w,h), interpolation=cv2.INTER_NEAREST)
        flow_arr[i] = flow
    return flow_arr