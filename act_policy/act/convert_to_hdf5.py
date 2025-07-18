
import os
import cv2
import h5py
import argparse
import json
from tqdm import tqdm 
import time 
import numpy as np

TASK_CONFIGS = {
    'cam_width': 640,
    'cam_height': 480,
    'camera_names': ['wrist', 'wrist_back']
    }
cfg = TASK_CONFIGS


def load_image(episode, time_step, image_folder):
    img_path = os.path.join(image_folder, f"{episode}_{time_step}.png")
    # print(f'Loading image: {img_path}')
    img = cv2.imread(img_path)
    if img is None:
        raise IOError(f"Cannot read image: {img_path}")
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return image

def sample_last_frames(max_timesteps, episode_len):
    start_idx = max(0, max_timesteps - episode_len)  # Get the last 'episode_len' indices
    sampled_indices = list(range(start_idx, max_timesteps))

    # If there are fewer than 'episode_len' frames, repeat the last frame
    while len(sampled_indices) < episode_len:
        sampled_indices.append(sampled_indices[-1])  # Duplicate the last frame
    
    # print(f'start_idx: {start_idx}')
    # print(f'sampled_indices: {sampled_indices}')

    return sampled_indices

def main(args):
    task = args.task_name
    dataset_dir = args.dataset_dir
    camera_names = args.camera_names
    episode_len = args.episode_len  
    start_idx = args.start_idx
    
    json_file = os.path.join(dataset_dir, 'data.json')
    img_folder = os.path.join(dataset_dir, 'visual_observations', 'realsensecameracolorimage_raw')

    with open(json_file, 'r') as f:
        json_data = json.load(f)

    # Extract and pop ft_bias if it exists
    ft_bias = None
    if 'ft_bias' in json_data:
        ft_bias = json_data.pop('ft_bias')
        print(f"Found ft_bias: {ft_bias}")


    # Iterate through each episode
    for eps, eps_data in json_data.items():
        # Create a dictionary to store the data for each episode
        data_dict = {
            '/observations/qpos': [],
            '/observations/qvel': [],
            '/action': [],
        }


        ## Indexing
        eps_idx = int(eps)+start_idx # Start from the specified index if defined
        print(f'Processing episode {eps_idx}')
        print(f'Number of timesteps: {len(eps_data)}')
        
        # Sample the last frames (for now)
        selected_indices = sample_last_frames(len(eps_data), episode_len) # Sample the frames        
        timesteps_list = list(eps_data.keys()) # Convert the timesteps to a list to index


        ## Store data in the data_dict
        for idx in selected_indices:
            timestep = timesteps_list[idx]  # Get the timestep from the list of timesteps
            # print(f'Processing idx {idx} timestep {timestep}')
            time_data = eps_data[timestep]
            obs_replay = time_data['robot_data']['observations']
            action_replay = time_data['robot_data']['action']

            # Create a dictionary to store the data for each timestep
            data_dict['/observations/qpos'].append(obs_replay['obs_pos'])
            data_dict['/observations/qvel'].append(obs_replay['obs_vel'])  
            data_dict['/action'].append(action_replay)

            # Load the image data for each camera
            image = load_image(eps, timestep, img_folder)

            for cam_name in camera_names:
                key = f'/observations/images/{cam_name}'

            if key not in data_dict:
                data_dict[key] = []  

            data_dict[key].append(image)  
            


        # Determine the file path for saving the dataset
        t0 = time.time()
        max_timesteps = episode_len
        data_dir = os.path.join(dataset_dir, task)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)  # Create the directory if it doesn't exist

        # Count existing files to index the new dataset
        idx = len([name for name in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, name))])
        dataset_path = os.path.join(data_dir, f'episode_{idx}')


        ## Save the data to an HDF5 file, organizing datasets into groups
        t0 = time.time()
        dataset_path = os.path.join(data_dir, f'episode_{eps_idx}')
        with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs['sim'] = True

            # If ft_bias data was collected
            if ft_bias is not None:
                root.create_dataset('ft_bias', data=ft_bias)

            obs = root.create_group('observations')
            image = obs.create_group('images')
            for cam_name in camera_names:
                _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3),
                                        dtype='uint8', chunks=(1, 480, 640, 3))
            qpos = obs.create_dataset('qpos', (max_timesteps, 6))
            qvel = obs.create_dataset('qvel', (max_timesteps, 6))
            action = root.create_dataset('action', (max_timesteps, 6))

            for name, array in data_dict.items():
                root[name][...] = array
        print(f'Saving: {time.time() - t0:.1f} secs\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', action='store', type=str, default='real_peg_insertion', help = 'default: real_peg_insertion')
    parser.add_argument('--dataset_dir', action='store', type=str, help='REQUIRED: abs path', required=True)
    parser.add_argument('--camera_names', type=str, default={'wrist', 'wrist_back'}, help = 'default: wrist')
    parser.add_argument('--episode_len', type=int, default=400, help = 'default: 400')
    parser.add_argument('--start_idx', type=int, default=0, help = 'default: 0')
    args = parser.parse_args() 
    main(args)