from plyfile import PlyData
import os
import numpy as np
import argparse
import json
import open3d as o3d
from tqdm import tqdm

def read_mesh_vertices(filename):
    assert os.path.isfile(filename)
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        # print(plydata['vertex'].data[1])
        vertices = np.zeros(shape=[num_verts, 3], dtype=np.float32)
        vertices[:,0] = plydata['vertex'].data['x']
        vertices[:,1] = plydata['vertex'].data['y']
        vertices[:,2] = plydata['vertex'].data['z']
    return vertices

def write_pcd(save_path, points_array):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_array)
    o3d.io.write_point_cloud(save_path, pcd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--scannet_root', type=str, default=None)
    parser.add_argument('--save_gt_dir', type=str, default=None)
    args = parser.parse_args()

    for scene_name in ['scene0008_00', 'scene0005_00', 'scene0050_00', 'scene0461_00', 'scene0549_00', 'scene0616_00']:
        ids = [int(x.split('_')[-1]) for x in os.listdir(os.path.join(args.scannet_root, 'object_original_with_clip')) if scene_name in x]
        
        ids = [x-1 for x in ids]
        
        print(scene_name)
        print(ids)
        
        save_dir = os.path.join(args.save_gt_dir, scene_name, "gt")

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        for id in ids:
            scannet_path = args.scannet_root
            scan_path = os.path.join(scannet_path, scene_name+'_scannet', f"{scene_name}_vh_clean_2.ply")
            vertices_data = read_mesh_vertices(scan_path)
            
            instance_path = os.path.join(scannet_path, scene_name+'_scannet', f"{scene_name}_vh_clean.aggregation.json")
            with open(instance_path, "r", encoding="utf-8") as f:
                inst_data = json.load(f)
            # print(inst_data["segGroups"][0])
            inst_idx = inst_data["segGroups"][id]['segments']
            
            if id == 7 and scene_name == 'scene0580_00':
                inst_idx = inst_idx + inst_data["segGroups"][8]['segments']
                inst_idx = inst_idx + inst_data["segGroups"][17]['segments']
            # import pdb; pdb.set_trace()

            seg_path = os.path.join(scannet_path, scene_name+'_scannet', f"{scene_name}_vh_clean_2.0.010000.segs.json")
            with open(seg_path, "r", encoding="utf-8") as f:
                seg_data = json.load(f)
            segIndices = seg_data["segIndices"]
            
            vertices_idx = []
            for idx in tqdm(range(len(segIndices))):
                for object_seg_id in inst_idx:
                    if segIndices[idx] == object_seg_id:
                        vertices_idx.append(idx)

            print(len(vertices_idx))
            segment_object_vertices = vertices_data[vertices_idx]
            write_pcd(os.path.join(save_dir, f"obj_{id}.ply"), segment_object_vertices)