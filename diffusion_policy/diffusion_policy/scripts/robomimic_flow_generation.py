if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)

import multiprocessing
import os
import shutil
import click
import pathlib
import h5py
from tqdm import tqdm
import collections
import pickle
from diffusion_policy.common.flow_util import get_flow_predictor, generate_flow_from_frames


@click.command()
@click.option('-i', '--input', required=True, help='input hdf5 path')
@click.option('-o', '--output', required=False, help='output hdf5 path. Default to input path with _flow suffix')
@click.option('-v', '--view', required=True, help='camera view to process')
@click.option('--interval', default=12, type=int, help='frame interval for flow computation')
@click.option('-n', '--num_workers', default=None, type=int)
def main(input, output, view, interval, num_workers):
    # process inputs
    input = pathlib.Path(input).expanduser()
    assert input.is_file()
    output = pathlib.Path(output).expanduser()
    assert output.parent.is_dir()
    assert not output.is_dir()
    # converter = RobomimicAbsoluteActionConverter(input)

    # run
    # with multiprocessing.Pool(num_workers) as pool:
    #     results = pool.map(worker, [(input, i, do_eval) for i in range(len(converter))])
    
    flow_predictor = get_flow_predictor()
    
    
    with h5py.File(input) as file:
        # count total steps
        demos = file['data']
        for i in range(len(demos)):
            demo = demos[f'demo_{i}']
            obs_arr = demo['obs'][view]
            flow_arr = generate_flow_from_frames(obs_arr, flow_predictor, interval=interval)
            demo['flow'][view] = flow_arr
                
            
if __name__ == "__main__":
    main()



