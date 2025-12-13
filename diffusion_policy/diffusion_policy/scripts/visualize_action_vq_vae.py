#!/usr/bin/env python
"""Visualize Action VQ-VAE latent / codebook trajectories over dataset demos."""
if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)

import argparse
import copy
import os
import pathlib
import sys
from typing import List, Sequence, Tuple, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm
import imageio
from PIL import Image


import hydra
from diffusion_policy.dataset.robomimic_replay_image_flow_dataset import RobomimicReplayImageFlowDataset
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


def build_action_image_dataset(cfg: OmegaConf, horizon: int) -> RobomimicReplayImageFlowDataset:
    # dataset_cfg = cfg.task.dataset
    # dataset_paths_cfg = dataset_cfg.dataset_path
    # if isinstance(dataset_paths_cfg, str):
    #     dataset_paths = [dataset_paths_cfg]
    # else:
    #     dataset_paths = list(dataset_paths_cfg)
    dataset_paths = ["data/core/coffee_d0.hdf5"]

    rotation_rep = "rotation_6d"
    seed = 42

    n_obs_steps = 2

    pad_before = 0
    pad_after = 0
    abs_action = False
    action_dataset = RobomimicReplayImageFlowDataset(
        shape_meta=copy.deepcopy(cfg.task.shape_meta),
        dataset_path=dataset_paths,
        horizon=horizon,
        n_action_primitives=8,
        pad_before=pad_before,
        pad_after=pad_after,
        n_obs_steps=n_obs_steps,
        abs_action=abs_action,
        rotation_rep=rotation_rep,
        use_cache=True,
        seed=int(seed),
        action_pre_encode=False,
        max_demos=10
    )
    return action_dataset


def extract_demo_sequences(
    dataset: RobomimicReplayImageFlowDataset,
    demo_indices: Sequence[int],
    image_key: Optional[str] = None,
) -> Tuple[List[torch.Tensor], List[Optional[np.ndarray]]]:
    sequences: List[torch.Tensor] = []
    image_sequences: List[Optional[np.ndarray]] = []

    for episode_idx in demo_indices:
        episode_data = dataset.replay_buffer.get_episode(episode_idx, copy=True)
        actions_np = episode_data["action"].astype(np.float32)
        sequences.append(torch.from_numpy(actions_np))

        if image_key is not None and image_key in episode_data:
            image_sequences.append(episode_data[image_key])
        else:
            image_sequences.append(None)

    return sequences, image_sequences


def sliding_window_latents(
    model: ActionVqVae,
    normalizer,
    sequences: Sequence[torch.Tensor],
    horizon: int,
    stride: int,
    device: torch.device,
    max_windows: int = None,
    demo_ids: Sequence[int] = None,
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    print("Horizon:", horizon)
    latents = []
    metadata = []  # (demo_idx, window_start)
    total_windows = 0

    action_normalizer = normalizer["action"].to(device)
    iterator = enumerate(tqdm(sequences, desc="Encoding demos")) if len(sequences) > 1 else enumerate(sequences)
    for demo_idx, seq in iterator:
        resolved_demo = demo_ids[demo_idx]
        seq_len = seq.shape[0]
        if seq_len < horizon:
            continue
        for start in range(0, seq_len - horizon + 1, stride):
            window = seq[start:start + horizon].unsqueeze(0).to(device)  # get action of horizon length
            norm_window = action_normalizer.normalize(window)
            # print(f"norm_window: {norm_window.squeeze(0).detach().cpu().numpy()}")
            # import time
            # time.sleep(1)
            with torch.no_grad():
                latent = model.encode(norm_window).detach()
            latents.append(latent.squeeze(0).cpu().numpy())
            metadata.append((resolved_demo, start))
            total_windows += 1
            if max_windows is not None and total_windows >= max_windows:
                return np.stack(latents), metadata
    if len(latents) == 0:
        raise RuntimeError("No windows collected. Check stride/horizon/demo selection.")
    return np.stack(latents), metadata


def collect_codebook_vectors(model: ActionVqVae) -> np.ndarray:
    with torch.no_grad():
        codebooks = model.vq_layer.codebooks  # (num_quantizers, codebook_size, dim)
        return [codebooks[i].cpu().numpy() for i in range(codebooks.shape[0])]


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
    codebook_2d_sets: List[np.ndarray],
    output_path: pathlib.Path,
    color_mode: str,
    connect: bool,
):
    plt.figure(figsize=(10, 8))
    latent_count = latent_2d.shape[0]

    # draw connecting trajectories first so points stay on top
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

    if color_mode == "time":
        colors = np.linspace(0, 1, latent_count)
        scatter = plt.scatter(
            latent_2d[:, 0],
            latent_2d[:, 1],
            c=colors,
            cmap="plasma",
            s=12,
            alpha=0.8,
            label="Encoded actions",
            zorder=3,
        )
        cbar = plt.colorbar(scatter)
        cbar.set_label("Temporal progression")
    else:
        demo_ids = np.array([m[0] for m in metadata])
        scatter = plt.scatter(
            latent_2d[:, 0],
            latent_2d[:, 1],
            c=demo_ids,
            cmap="tab20",
            s=12,
            alpha=0.8,
            label="Encoded actions",
            zorder=3,
        )
        cbar = plt.colorbar(scatter)
        cbar.set_label("Demo index")

    if codebook_2d_sets is not None and len(codebook_2d_sets) > 0:
        colors = plt.cm.get_cmap("Set1", len(codebook_2d_sets))
        markers = ["x", "+", "1", "2"]
        sizes = [80, 60, 40, 30]
        for i, codebook_set in enumerate(codebook_2d_sets):
            plt.scatter(
                codebook_set[:, 0],
                codebook_set[:, 1],
                c=[colors(i)],
                marker=markers[i % len(markers)],
                s=sizes[i % len(sizes)],
                label=f"Codebook Layer {i+1}",
                alpha=0.9,
                linewidths=1.5,
            )

    plt.title("Action VQ-VAE latent trajectory")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    
def _render_image(ax, image: np.ndarray, title: Optional[str] = None):
    ax.axis("off")
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    img = np.asarray(image)
    if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
        # convert channel-first tensors to channel-last for Matplotlib
        img = np.moveaxis(img, 0, -1)

    if img.dtype == np.uint8:
        disp_img = img
    else:
        max_val = np.max(img)
        if max_val > 1.0:
            disp_img = np.clip(img, 0.0, 255.0) / 255.0
        else:
            disp_img = np.clip(img, 0.0, 1.0)

    ax.imshow(disp_img)
    if title:
        ax.set_title(title)


def render_action_latent_video(
    images: np.ndarray,
    horizon: int,
    latent_coords: np.ndarray,
    latent_steps: List[int],
    output_path: pathlib.Path,
    fps: int = 6,
):
    if len(latent_coords) == 0:
        raise RuntimeError("No latent coordinates available for the requested demo.")
    if images is None:
        raise RuntimeError("Image data required for video rendering.")

    images_np = np.asarray(images)
    total_steps = images_np.shape[0]
    if total_steps == 0:
        raise RuntimeError("Empty image sequence provided for video rendering.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(output_path, fps=fps) as writer:
        for idx, (coord, step) in enumerate(zip(latent_coords, latent_steps)):
            future_idx = step + horizon
            if future_idx >= total_steps:
                continue

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            _render_image(axes[0], images_np[step], title=f"Obs t={step}")

            axes[1].scatter(latent_coords[:, 0], latent_coords[:, 1], c="lightgray", s=10, alpha=0.4, label="trajectory")
            if idx > 0:
                axes[1].plot(latent_coords[: idx + 1, 0], latent_coords[: idx + 1, 1], color="#1f77b4", linewidth=1.0, alpha=0.7)
            axes[1].scatter(coord[0], coord[1], c="crimson", s=40, label="current")
            axes[1].set_title("Latent position")
            axes[1].set_xlabel("Comp 1")
            axes[1].set_ylabel("Comp 2")
            axes[1].set_aspect("equal")

            _render_image(axes[2], images_np[future_idx], title=f"Obs t+H={future_idx}")

            plt.suptitle(f"Frame {idx} | step {step}")
            fig.tight_layout()
            fig.canvas.draw()
            width, height = fig.canvas.get_width_height()
            # Use buffer_rgba for compatibility with newer Matplotlib versions
            buf = fig.canvas.buffer_rgba()
            img = Image.frombuffer("RGBA", (width, height), buf)
            img = img.convert("RGB")
            frame = np.array(img) 
            writer.append_data(frame)
            plt.close(fig)

"""
python diffusion_policy/scripts/visualize_action_vq_vae.py \
  --checkpoint data/checkpoints/mlp2/latest.ckpt \
  --output data/media/latents/coffee_demo0.png \
  --demo-idx 0 \
  --stride 1 \
  --reducer tsne \
  --video-output data/media/latents/coffee_demo0_mlp2.mp4 
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
    parser.add_argument("--color-mode", choices=["time", "demo"], default="time", help="Coloring scheme for scatter plot.")
    parser.add_argument("--connect-trajectories", action="store_true", help="Draw lines connecting latent points per demo.")
    parser.add_argument("--random-seed", type=int, default=0, help="Random seed for dimensionality reduction (TSNE).")
    parser.add_argument("--video-output", type=str, default=None, help="Optional path to save latent-action video animation.")
    parser.add_argument("--video-demo-idx", type=int, default=None, help="Demo index to render video for (defaults to --demo-idx).")
    parser.add_argument("--video-fps", type=int, default=6, help="Frames per second for latent-action video.")
    args = parser.parse_args()

    device = _resolve_device(args.device)

    print(f"Loading checkpoint from {args.checkpoint}")
    model, cfg = load_model_from_checkpoint(args.checkpoint, device)

    print("Building lightweight action dataset to stream demos...")
    dataset = build_action_image_dataset(cfg, horizon=model.input_dim_h)
    normalizer = dataset.get_normalizer()
    normalizer.to(device)

    n_episodes = dataset.replay_buffer.n_episodes
    if args.demo_idx is not None:
        demo_indices = [args.demo_idx]
    else:
        demo_indices = list(range(n_episodes))
        if args.max_demos is not None:
            demo_indices = demo_indices[:args.max_demos]
    print(f"Using {len(demo_indices)} demos (out of {n_episodes}).")

    primary_rgb_key = dataset.rgb_keys[0] if len(dataset.rgb_keys) > 0 else None
    action_sequences, image_sequences = extract_demo_sequences(
        dataset,
        demo_indices,
        image_key=primary_rgb_key,
    )
    demo_records = {
        idx: {"actions": action_seq, "images": image_seq}
        for idx, action_seq, image_seq in zip(demo_indices, action_sequences, image_sequences)
    }

    print("Encoding sliding windows...")
    latents, metadata = sliding_window_latents(
        model=model,
        normalizer=normalizer,
        sequences=action_sequences,
        horizon=model.input_dim_h,
        stride=args.stride,
        device=device,
        max_windows=args.max_windows,
        demo_ids=demo_indices,
    )
    print(f"Collected {latents.shape[0]} latent points")
    dump_path = pathlib.Path("data/media/latents/latent_points.txt")
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    latents_flat = latents.reshape(latents.shape[0], -1)
    np.savetxt(dump_path, latents_flat, fmt="%.6f")
    print(f"Saved latent tensor with shape {latents_flat.shape} to {dump_path}")
    
    print("Collecting codebook vectors...")
    codebook_vectors_sets = collect_codebook_vectors(model)
    codebook_vectors_flat = [cv.reshape(cv.shape[0], -1) for cv in codebook_vectors_sets]
    vector_counts = [cv.shape[0] for cv in codebook_vectors_flat]
    print(f"Collected {len(vector_counts)} codebook layers with counts: {vector_counts}")

    print(f"Running {args.reducer.upper()} for dimensionality reduction...")
    all_vectors_flat = np.concatenate([latents_flat] + codebook_vectors_flat, axis=0)
    reduced = reduce_vectors(
        all_vectors_flat, method=args.reducer, random_state=args.random_seed
    )

    latent_2d = reduced[: latents_flat.shape[0]]
    codebook_2d_sets = []
    current_idx = latents_flat.shape[0]
    for count in vector_counts:
        codebook_2d_sets.append(reduced[current_idx:current_idx+count])
        current_idx += count

    output_path = pathlib.Path(args.output)
    print(f"Saving plot to {output_path}")
    # plot_latents(
    #     latent_2d=latent_2d,
    #     metadata=metadata,
    #     codebook_2d_sets=codebook_2d_sets,
    #     output_path=output_path,
    #     color_mode=args.color_mode,
    #     connect=args.connect_trajectories,
    # )

    if args.video_output is not None:
        video_demo_idx = args.video_demo_idx if args.video_demo_idx is not None else args.demo_idx
        if video_demo_idx is None:
            raise ValueError("Video rendering requires --video-demo-idx or --demo-idx to be specified.")
        if video_demo_idx not in demo_records:
            raise ValueError(f"Demo index {video_demo_idx} was not part of the processed set {demo_indices}.")

        demo_latent_indices = [i for i, (demo_id, _) in enumerate(metadata) if demo_id == video_demo_idx]
        if len(demo_latent_indices) == 0:
            raise RuntimeError(f"No latent windows recorded for demo {video_demo_idx}.")

        demo_latent_coords = latent_2d[demo_latent_indices]
        demo_latent_steps = [metadata[i][1] for i in demo_latent_indices]
        video_path = pathlib.Path(args.video_output)
        print(f"Rendering latent-action video for demo {video_demo_idx} to {video_path}")
        demo_record = demo_records[video_demo_idx]
        if demo_record["images"] is None:
            raise RuntimeError("Video rendering requested but no RGB observations are available in the dataset.")
        render_action_latent_video(
            images=demo_record["images"],
            horizon=model.input_dim_h,
            latent_coords=demo_latent_coords,
            latent_steps=demo_latent_steps,
            output_path=video_path,
            fps=args.video_fps,
        )

    print("Done.")


if __name__ == "__main__":
    main()
