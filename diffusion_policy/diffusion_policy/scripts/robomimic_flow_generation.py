if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)

import multiprocessing as mp
import os
import shutil
import click
import pathlib
import h5py
from tqdm import tqdm
import collections
import pickle
import traceback
from diffusion_policy.common.flow_util import get_flow_predictor, generate_flow_from_frames


def worker(args):
    input_path, demo_key, view, interval, target_size = args
    try:
        # get_flow_predictor is initialized in each worker
        flow_predictor = get_flow_predictor()

        with h5py.File(input_path, 'r') as f:
            obs_arr = f[f'data/{demo_key}/obs/{view}'][:]
            
        if obs_arr.shape[0] <= interval:
            return (demo_key, False, "Not enough frames.")

        flow_arr = generate_flow_from_frames(
            obs_arr, 
            flow_predictor, 
            interval=interval, 
            target_size=target_size,
            progress_desc=f"Flow for {demo_key}"
        )

        # Return the result instead of writing to file
        return (demo_key, True, flow_arr)
    except Exception as e:
        # Return the error message
        error_trace = traceback.format_exc()
        return (demo_key, False, error_trace)
        # return (demo_key, False, str(e))
    

@click.command()
@click.option('-i', '--input', required=True, help='input hdf5 path')
@click.option('-v', '--view', required=True, help='camera view to process')
@click.option('--interval', default=12, type=int, help='frame interval for flow computation')
@click.option('-n', '--num_workers', default=4, type=int)
@click.option('-t', '--target_size', default=224, type=int, help='resize the shorter side of the image to this size before flow computation')
def main(input, view, interval, num_workers, target_size):
    # process inputs
    input_path = pathlib.Path(input).expanduser()
    assert input_path.is_file()

    output_path = input_path.parent.joinpath(input_path.stem + '_flow' + input_path.suffix)
    print(f"Input file: {input_path}")
    print(f"Output file for flow data: {output_path}")
        
    if num_workers is None:
        num_workers = mp.cpu_count()
        
    with h5py.File(input_path, 'r') as f:
        demo_keys = sorted(list(f['data'].keys()))

    print(f"Processing {len(demo_keys)} demos with {num_workers} workers...")
    
    # run
    tasks = [(input_path, key, view, interval, target_size) for key in demo_keys]
    results = list()
    # Use imap_unordered for a live progress bar, as map blocks until all results are ready.
    with mp.Pool(num_workers) as pool:
        with tqdm(total=len(tasks), desc="Processing Demos") as pbar:
            for result in pool.imap_unordered(worker, tasks):
                results.append(result)
                pbar.update()
                
    # Sort results back into the original order of demo_keys
    results_dict = {key: (success, data) for key, success, data in results}
    results = [(key, results_dict[key][0], results_dict[key][1]) for key in demo_keys if key in results_dict]
                
    # Write results in the main process to a new file
    success_count = 0
    fail_count = 0
    print(f"Writing {len(results)} results to {output_path}...")
    with h5py.File(output_path, 'w') as f:
        for demo_key, success, result_data in tqdm(results):
            if success:
                flow_arr = result_data
                if f'data/{demo_key}/flow' not in f:
                    f.create_group(f'data/{demo_key}/flow')
                
                dset_path = f'data/{demo_key}/flow/{view}'
                if dset_path in f:
                    del f[dset_path]
                
                f.create_dataset(
                    name=dset_path,
                    data=flow_arr,
                    shape=flow_arr.shape,
                    dtype=flow_arr.dtype,
                    compression='gzip'
                )
                success_count += 1
            else:
                error_msg = result_data
                fail_count += 1
                print(f"Fail to process: {demo_key}, Error: {error_msg}")
            
    print(f"Done! Success: {success_count}, Fail: {fail_count}")
                
            
if __name__ == "__main__":
    main()



