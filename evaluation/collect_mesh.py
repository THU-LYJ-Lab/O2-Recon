import os
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--exp_root', type=str, default=None)

args = parser.parse_args()
    
exp_root = args.exp_root

save_dir = exp_root.replace('neus', 'collected_meshes_per_scene')

method = 'inpainted'

# for scene_name in ['scene0008_00', 'scene0005_00', 'scene0050_00', 'scene0461_00', 'scene0549_00', 'scene0616_00']:
for scene_name in ['scene0008_00']:
    obj_names = []
    
    for name in os.listdir(exp_root):
        if scene_name in name and name.endswith(method):
            obj_names.append(name)
    
    for obj_name in obj_names:
        obj_id = int(obj_name.split('_')[4])

        if not os.path.exists(os.path.join(save_dir, method, scene_name)):
            os.makedirs(os.path.join(save_dir, method, scene_name))
        
        exp_dir = os.path.join(exp_root, '{:s}_scannet_obj_{:d}_{:s}'.format(scene_name, obj_id, method))
        
        pcd_path = os.path.join(exp_dir, 'exp_{:s}_scannet_obj_{:d}_{:s}_default/'.format(scene_name, obj_id, method), 'meshes', '00050000_reso512_{:s}_scannet_obj_{:d}_{:s}_world_cleaned.ply'.format(scene_name, obj_id, method))
        
        mesh_path = os.path.join(exp_dir, 'exp_{:s}_scannet_obj_{:d}_{:s}_default/'.format(scene_name, obj_id, method), 'meshes', '00050000_reso512_{:s}_scannet_obj_{:d}_{:s}_world_cleaned_mesh.ply'.format(scene_name, obj_id, method))
        
        full_mesh_path = os.path.join(exp_dir, 'exp_{:s}_scannet_obj_{:d}_{:s}_default/'.format(scene_name, obj_id, method), 'meshes', '00050000_reso512_{:s}_scannet_obj_{:d}_{:s}_world.ply'.format(scene_name, obj_id, method))
        
        target_pcd_path = os.path.join(save_dir, method, scene_name, '{:s}_scannet_obj_{:d}_{:s}_pointcloud.ply'.format(scene_name, obj_id, method))
        
        target_mesh_path = os.path.join(save_dir, method, scene_name, '{:s}_scannet_obj_{:d}_{:s}_mesh.ply'.format(scene_name, obj_id, method))
        
        target_full_mesh_path = os.path.join(save_dir, method, scene_name, '{:s}_scannet_obj_{:d}_{:s}_full_mesh.ply'.format(scene_name, obj_id, method))
        
        try:
            shutil.copy(pcd_path, target_pcd_path)
            shutil.copy(mesh_path, target_mesh_path)
            shutil.copy(full_mesh_path, target_full_mesh_path)
        except:
            import pdb; pdb.set_trace()

    