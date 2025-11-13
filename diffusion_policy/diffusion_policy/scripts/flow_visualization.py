if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)

import click
import pathlib
import h5py
import numpy as np
import cv2
from tqdm import tqdm
from diffusion_policy.common.flow_viz import flow_to_image


@click.command()
@click.option('-i', '--input', required=True, help='Input HDF5 file path for observation data.')
@click.option('-o', '--output', required=False, help='Output video file path (e.g., video.mp4). If not provided, a default name will be generated.')
@click.option('-d', '--demo_key', required=True, help='Demonstration key to visualize (e.g., demo_0).')
@click.option('-v', '--view', default='agentview', help='Camera view to process.')
@click.option('--fps', default=30, type=int, help='Frames per second for the output video.')
@click.option('--save-frames', is_flag=True, help='Save individual frames as images.')
def main(input, output, demo_key, view, fps, save_frames):
    """
    Generates a video visualizing an image sequence and its corresponding optical flow.
    The left side of the video shows the original image, and the right side shows the flow.
    """
    input_path = pathlib.Path(input).expanduser()
    # Infer flow file path from the input observation file path
    flow_input_path = input_path.parent.joinpath(input_path.stem + '_flow' + input_path.suffix)
    
    if output is None:
        # Generate a default output path if not provided
        output_path = input_path.parent.joinpath(f"{input_path.stem}_{demo_key}_flow_vis.mp4")
    else:
        output_path = pathlib.Path(output).expanduser()

    if not input_path.is_file():
        print(f"Error: Input observation file not found at {input_path}")
        return
    if not flow_input_path.is_file():
        print(f"Error: Inferred flow file not found at {flow_input_path}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame_output_path = output_path.parent.joinpath("image")
    if save_frames:
        frame_output_path.mkdir(parents=True, exist_ok=True)

    print(f"Reading observation data for {demo_key} from {input_path}")
    print(f"Reading flow data for {demo_key} from {flow_input_path}")
    
    # Read observation data from the original file
    with h5py.File(input_path, 'r') as f:
        try:
            # Load original images (T, H, W, C)
            obs_arr = f[f'data/{demo_key}/obs/{view}'][:]
        except KeyError as e:
            print(f"Error: Observation data not found in {input_path}. Missing key: {e}")
            return

    # Read flow data from the separate flow file
    with h5py.File(flow_input_path, 'r') as f:
        try:
            # Load flow data (T, 2, H, W)
            flow_arr = f[f'data/{demo_key}/flow/{view}'][:]
        except KeyError as e:
            print(f"Error: Flow data not found in {flow_input_path}. Missing key: {e}")
            return

    num_frames, height, width, _ = obs_arr.shape
    
    flow_arr = flow_arr[..., :2]  # only use the first two channels

    # Setup video writer
    # The output video width will be doubled to accommodate both images
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width * 2, height))

    print(f"Generating video {output_path}...")
    for i in tqdm(range(num_frames)):
        # Original image (convert RGB to BGR for OpenCV)
        original_img_bgr = cv2.cvtColor(obs_arr[i], cv2.COLOR_RGB2BGR)
        
        # Convert flow to a visual representation
        flow_vis_img = flow_to_image(flow_arr[i])  
        
        # Combine images side-by-side
        combined_frame = np.concatenate((original_img_bgr, flow_vis_img), axis=1)
        
        if save_frames:
            frame_output_path_each = frame_output_path.joinpath(f"{demo_key}_frame_{i:04d}.png")
            cv2.imwrite(str(frame_output_path_each), original_img_bgr)

        # Write frame to video
        video_writer.write(combined_frame)

    # Release the video writer
    video_writer.release()
    print(f"Done! Video saved to {output_path}")
    if save_frames:
        print(f"Individual frames saved to {frame_output_path}")

if __name__ == "__main__":
    main()