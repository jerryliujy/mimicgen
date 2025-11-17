#!/usr/bin/env python
"""Visualize Action VQ-VAE latent / codebook trajectories over dataset demos."""

import argparse
import copy
import os
import pathlib
import sys
from typing import List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm

# Ensure repo root is on sys.path so hydra target strings work when script is run directly
ROOT_DIR = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
os.chdir(ROOT_DIR)

import hydra
from diffusion_policy.dataset.robomimic_replay_action_dataset import (
    RobomimicReplayActionDataset,
)
from diffusion_policy.model.action.action_vq_vae import ActionVqVae


def _resolve_device(device_str: str) -> torch.device:
    if device_str.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_str)
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available. Falling back to CPU.")
    return torch.device("cpu")


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device) -> Tuple[ActionVqVae, OmegaConf]:
    payload = torch.load(checkpoint_path, map_location=device)
    if "cfg" not in payload:
        raise RuntimeError("Checkpoint payload missing 'cfg'.")
    cfg = payload["cfg"]

    model: ActionVqVae = hydra.utils.instantiate(cfg.policy)
    state_dicts = payload.get("state_dicts", {})
    if "model" not in state_dicts:
        raise RuntimeError("Checkpoint missing model state dict.")
    model.load_state_dict(state_dicts["model"])
    model.to(device)
    model.eval()
    return model, cfg


def build_action_dataset(cfg: OmegaConf, horizon: int) -> RobomimicReplayActionDataset:
    dataset_cfg = cfg.task.dataset
    dataset_paths = list(dataset_cfg.dataset_path)
    rotation_rep = getattr(dataset_cfg, "rotation_rep", "rotation_6d")
    val_ratio = getattr(dataset_cfg, "val_ratio", 0.0)
    seed = getattr(dataset_cfg, "seed", 42)

    action_dataset = RobomimicReplayActionDataset(
        shape_meta=copy.deepcopy(cfg.task.shape_meta),
        dataset_path=dataset_paths,
        horizon=horizon,
        pad_before=int(dataset_cfg.pad_before),
        pad_after=int(dataset_cfg.pad_after),
        abs_action=bool(dataset_cfg.abs_action),
        rotation_rep=rotation_rep,
        seed=int(seed),
        val_ratio=float(val_ratio),
    )
    return action_dataset


def extract_demo_sequences(dataset: RobomimicReplayActionDataset, demo_indices: Sequence[int]) -> List[torch.Tensor]:
    actions = dataset.actions_in_memory # (total_steps, action_dim)
    episode_starts = dataset.sampler.episode_starts
    episode_ends = dataset.sampler.episode_ends
    sequences = []
    for episode_idx in demo_indices:
        start_idx = int(episode_starts[episode_idx])
        end_idx = int(episode_ends[episode_idx])
        sequences.append(actions[start_idx:end_idx].clone())
    return sequences


def sliding_window_latents(
    model: ActionVqVae,
    normalizer,
    sequences: Sequence[torch.Tensor],
    horizon: int,
    stride: int,
    device: torch.device,
    latent_aggregation: str,
    max_windows: int = None,
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    latents = []
    metadata = []  # (demo_idx, window_start)
    total_windows = 0

    action_normalizer = normalizer["action"].to(device)
    iterator = enumerate(tqdm(sequences, desc="Encoding demos")) if len(sequences) > 1 else enumerate(sequences)
    for demo_idx, seq in iterator:
        seq_len = seq.shape[0]
        if seq_len < horizon:
            continue
        for start in range(0, seq_len - horizon + 1, stride):
            window = seq[start:start + horizon].unsqueeze(0).to(device)  # get action of horizon length
            norm_window = action_normalizer.normalize(window)
            with torch.no_grad():
                latent = model.encode(norm_window).detach()
            if latent.ndim == 3:
                if latent_aggregation == "mean":
                    latent = latent.mean(dim=-1)
                elif latent_aggregation == "max":
                    latent = latent.max(dim=-1).values
                else:  # flatten
                    latent = latent.reshape(latent.shape[0], -1)
            latents.append(latent.squeeze(0).cpu().numpy())
            metadata.append((demo_idx, start))
            total_windows += 1
            if max_windows is not None and total_windows >= max_windows:
                return np.stack(latents), metadata
    if len(latents) == 0:
        raise RuntimeError("No windows collected. Check stride/horizon/demo selection.")
    return np.stack(latents), metadata


def collect_codebook_vectors(model: ActionVqVae) -> np.ndarray:
    with torch.no_grad():
        codebooks = model.vq_layer.codebooks  # (num_quantizers, codebook_size, dim)
        flat = codebooks.reshape(-1, codebooks.shape[-1]).cpu().numpy()
    return flat


def reduce_vectors(vectors: np.ndarray, method: str, random_state: int = 0) -> np.ndarray:
    if vectors.shape[0] < 2:
        raise RuntimeError("Need at least two points for dimensionality reduction.")
    if method == "pca":
        reducer = PCA(n_components=2)
    elif method == "tsne":
        reducer = TSNE(n_components=2, init="pca", learning_rate="auto", random_state=random_state)
    else:
        raise ValueError(f"Unsupported reducer {method}")
    return reducer.fit_transform(vectors)


def plot_latents(
    latent_2d: np.ndarray,
    metadata: List[Tuple[int, int]],
    codebook_2d: np.ndarray,
    output_path: pathlib.Path,
    color_mode: str,
    connect: bool,
):
    plt.figure(figsize=(8, 6))
    latent_count = latent_2d.shape[0]

    if color_mode == "time":
        colors = np.linspace(0, 1, latent_count)
        scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=colors, cmap="plasma", s=12, alpha=0.8, label="Encoded actions")
        cbar = plt.colorbar(scatter)
        cbar.set_label("Temporal progression")
    else:  # per demo
        demo_ids = np.array([m[0] for m in metadata])
        scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=demo_ids, cmap="tab20", s=12, alpha=0.8, label="Encoded actions")
        cbar = plt.colorbar(scatter)
        cbar.set_label("Demo index")

    if connect:
        demo_to_points = {}
        for idx, (demo_idx, _) in enumerate(metadata):
            demo_to_points.setdefault(demo_idx, []).append(idx)
        for demo_idx, point_indices in demo_to_points.items():
            ordered = sorted(point_indices, key=lambda i: metadata[i][1])
            plt.plot(
                latent_2d[ordered, 0],
                latent_2d[ordered, 1],
                linewidth=0.7,
                alpha=0.4,
            )

    if codebook_2d is not None and len(codebook_2d) > 0:
        plt.scatter(
            codebook_2d[:, 0],
            codebook_2d[:, 1],
            c="black",
            marker="x",
            s=40,
            label="Codebook entries",
            alpha=0.7,
        )

    plt.title("Action VQ-VAE latent trajectory")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

"""
python diffusion_policy/scripts/visualize_action_vq_vae.py \
  --checkpoint data/outputs/.../checkpoints/latest.ckpt \
  --output media/latents/coffee_demo0.png \
  --demo-idx 0 \
  --stride 2 \
  --reducer tsne \
  --latent-aggregation flatten \
  --connect-trajectories
"""
def main():
    parser = argparse.ArgumentParser(description="Visualize Action VQ-VAE latent space over dataset demos.")
    parser.add_argument("--checkpoint", required=True, help="Path to workspace checkpoint (.ckpt)")
    parser.add_argument("--output", default="latent_trajectory.png", help="Path to save the plot")
    parser.add_argument("--device", default="cuda", help="Device to run encoding on (e.g., cuda, cuda:0, cpu)")
    parser.add_argument("--demo-idx", type=int, default=None, help="Specific demo index to visualize. Defaults to all demos.")
    parser.add_argument("--max-demos", type=int, default=None, help="Maximum number of demos to include.")
    parser.add_argument("--stride", type=int, default=1, help="Sliding window stride over actions.")
    parser.add_argument("--max-windows", type=int, default=20000, help="Maximum number of windows to encode (to limit runtime).")
    parser.add_argument("--reducer", choices=["pca", "tsne"], default="pca", help="Dimensionality reduction method.")
    parser.add_argument("--latent-aggregation", choices=["flatten", "mean", "max"], default="flatten", help="How to collapse non-MLP latents with temporal dimension.")
    parser.add_argument("--color-mode", choices=["time", "demo"], default="time", help="Coloring scheme for scatter plot.")
    parser.add_argument("--connect-trajectories", action="store_true", help="Draw lines connecting latent points per demo.")
    parser.add_argument("--random-seed", type=int, default=0, help="Random seed for dimensionality reduction (TSNE).")
    args = parser.parse_args()

    device = _resolve_device(args.device)

    print(f"Loading checkpoint from {args.checkpoint}")
    model, cfg = load_model_from_checkpoint(args.checkpoint, device)

    print("Building lightweight action dataset to stream demos...")
    dataset = build_action_dataset(cfg, horizon=model.input_dim_h)
    normalizer = dataset.get_normalizer()
    normalizer.to(device)

    n_episodes = len(dataset.sampler.episode_ends)
    if args.demo_idx is not None:
        demo_indices = [args.demo_idx]
    else:
        demo_indices = list(range(n_episodes))
        if args.max_demos is not None:
            demo_indices = demo_indices[:args.max_demos]
    print(f"Using {len(demo_indices)} demos (out of {n_episodes}).")

    sequences = extract_demo_sequences(dataset, demo_indices)

    print("Encoding sliding windows...")
    latents, metadata = sliding_window_latents(
        model=model,
        normalizer=normalizer,
        sequences=sequences,
        horizon=model.input_dim_h,
        stride=args.stride,
        device=device,
        latent_aggregation=args.latent_aggregation,
        max_windows=args.max_windows,
    )
    print(f"Collected {latents.shape[0]} latent points")

    print("Collecting codebook vectors...")
    codebook_vectors = collect_codebook_vectors(model)

    print(f"Running {args.reducer.upper()} for dimensionality reduction...")
    combined = np.concatenate([latents, codebook_vectors], axis=0)
    reduced = reduce_vectors(combined, method=args.reducer, random_state=args.random_seed)
    latent_2d = reduced[: latents.shape[0]]
    codebook_2d = reduced[latents.shape[0] :]

    output_path = pathlib.Path(args.output)
    print(f"Saving plot to {output_path}")
    plot_latents(
        latent_2d=latent_2d,
        metadata=metadata,
        codebook_2d=codebook_2d,
        output_path=output_path,
        color_mode=args.color_mode,
        connect=args.connect_trajectories,
    )

    print("Done.")


if __name__ == "__main__":
    main()
