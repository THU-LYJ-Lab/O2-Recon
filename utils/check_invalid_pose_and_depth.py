import os
import cv2
import numpy as np
from tqdm import tqdm

for scene_name in ['scene0005_00', 'scene0008_00', 'scene0050_00', 'scene0461_00', 'scene0549_00', 'scene0616_00']:
    scene_pose_path = f'/data/huyb/nips-2023/ys-NeuRIS/dataset/indoor/{scene_name}_scannet/pose'
    
    scene_depth_path = f'/data/huyb/nips-2023/ys-NeuRIS/dataset/indoor/{scene_name}_scannet/depth'
    
    for name in tqdm(os.listdir(scene_pose_path)):
        pose = np.loadtxt(os.path.join(scene_pose_path, name))
        if np.isnan(pose).any():
            print(scene_name, 'pose', 'nan', name)
        
        if np.isinf(pose).any():
            print(scene_name, 'pose', 'inf', name)
        
        # check inf values 
        depth = cv2.imread(os.path.join(scene_depth_path, name.replace('pose', 'depth')).replace('txt', 'png'), -1)
        
        if np.isinf(depth).any():
            print(scene_name, 'depth', 'inf', name)
            
        if np.isnan(depth).any():
            print(scene_name, 'depth', 'nan', name)