import numpy as np
import open3d as o3d
import sklearn.neighbors as skln
import argparse
from glob import glob
import os

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--scannet_gt_dir', type=str, default='/data/huyb/aaai2024/O2-Recon/evaluation/scannet_gt', help='scannet gt data')
    parser.add_argument('--scan2cad_gt_dir', type=str, default='/data/huyb/aaai2024/O2-Recon/evaluation/meshes-scan2cad-cleaned', help='scan2cad gt data')
    parser.add_argument('--max_dist', type=float, default=10, help='ignore the irrelevant points')
    args = parser.parse_args()

    for method in ['frodo', 'monosdf', 'scan2cad-prediction', 'vmap', 'newfilled_5', 'selected_5']:
        if method in ['frodo', 'monosdf', 'scan2cad-prediction', 'vmap']:
            data_dir = os.path.join('/data/huyb/aaai2024/O2-Recon/results/baselines', method)
        else:
            data_dir = os.path.join('/data/huyb/aaai2024/O2-Recon/results/ours', method)
        
        vis_out_dir = os.path.join(data_dir, 'metrics-new')
        if not os.path.exists(vis_out_dir):
            os.mkdir(vis_out_dir)
        
        for scene_name in ['scene0008_00', 'scene0005_00', 'scene0050_00', 'scene0461_00', 'scene0549_00', 'scene0616_00']:
            scene_dir = os.path.join(data_dir, scene_name)
            
            for eval_mode in ['scannet', 'scan2cad']:
                if eval_mode == 'scannet':
                    ## evaluate on scannet gt
                    gt_dir = os.path.join(args.scannet_gt_dir, scene_name, 'gt')
                    
                    if os.path.exists(os.path.join(vis_out_dir, f"metric_{scene_name}.txt")):
                        print(f"File %s already exists."%(os.path.join(vis_out_dir, f"metric_{scene_name}.txt")))
                        continue
                    else:
                        file = open(os.path.join(vis_out_dir, f"metric_{scene_name}.txt"), 'w')
                    if method in ['frodo', 'monosdf', 'scan2cad-prediction', 'vmap']:
                        input_data_list = sorted(glob(os.path.join(scene_dir, '*_pointcloud.ply')))
                        ## remove the *_full_pointcloud.ply
                        input_data_list = [x for x in input_data_list if 'full' not in x]
                    elif method in ['newfilled_5', 'selected_5']:
                        input_data_list = sorted(glob(os.path.join(scene_dir, '*_pointcloud.ply')))
                
                elif eval_mode == 'scan2cad':
                    ## evaluate on scan2cad gt
                    gt_dir = os.path.join(args.scan2cad_gt_dir, scene_name)
                    
                    if os.path.exists(os.path.join(vis_out_dir, f"metric_{scene_name}_scan2cad.txt")):
                        print(f"File %s already exists."%(os.path.join(vis_out_dir, f"metric_{scene_name}_scan2cad.txt")))
                        continue
                    else:
                        file = open(os.path.join(vis_out_dir, f"metric_{scene_name}_scan2cad.txt"), 'w')
                    
                    if method == 'monosdf':
                        input_data_list = sorted(glob(os.path.join(scene_dir, '*_mesh.ply')))
                    elif method in ['frodo', 'scan2cad-prediction', 'vmap']:
                        input_data_list = sorted(glob(os.path.join(scene_dir, '*_full_pointcloud.ply')))
                    elif method in ['newfilled_5', 'selected_5']:
                        input_data_list = sorted(glob(os.path.join(scene_dir, '*_mesh.ply')))
            
            
                accuracy_list = []
                completion_list = []
                chamfer_distance_list = []
                precision_1_list = []
                recall_1_list = []
                fscore_1_list = []
                fscore_2_list = []
                fscore_3_list = []

                for input_data in input_data_list:   
                    obj_id = int(os.path.basename(input_data).split('_')[4]) - 1
                    
                    # if scene_name != 'scene0616_00':
                    #     obj_id = int(obj_id) - 1
                        
                    print(obj_id)

                    if eval_mode == 'scannet':
                        gt_data = os.path.join(gt_dir, f"obj_{obj_id}.ply")
                           
                    elif eval_mode == 'scan2cad':                        
                        gt_data = os.path.join(gt_dir, f"scannet_{obj_id}_pointcloud.ply")

                        ## some objects are not in the scan2cad gt data
                        if not os.path.exists(gt_data):
                            continue
                    
                    print(input_data)
                    print(gt_data)
                    
                    dist_thred3 = 0.10
                    dist_thred1 = 0.05  # 5cm
                    dist_thred2 = 0.02  # 2cm

                    print('read data pcd ...')
                    data_pcd_o3d = o3d.io.read_point_cloud(input_data)
                    data_pcd = np.asarray(data_pcd_o3d.points)
                    
                    print('random shuffle pcd index ...')
                    shuffle_rng = np.random.default_rng()
                    shuffle_rng.shuffle(data_pcd, axis=0)

                    print('read GT pcd ...')
                    stl_pcd = o3d.io.read_point_cloud(gt_data)
                    stl = np.asarray(stl_pcd.points)

                    print('compute data2gt ...')
                    nn_engine = skln.NearestNeighbors(n_neighbors=1, radius=1.0, algorithm='kd_tree', n_jobs=-1)
                    nn_engine.fit(stl)
                    dist_d2s, idx_d2s = nn_engine.kneighbors(data_pcd, n_neighbors=1, return_distance=True)
                    max_dist = args.max_dist
                    mean_d2s = dist_d2s[dist_d2s < max_dist].mean()
                    precision_1 = len(dist_d2s[dist_d2s < dist_thred1]) / len(dist_d2s)
                    precision_2 = len(dist_d2s[dist_d2s < dist_thred2]) / len(dist_d2s)
                    precision_3 = len(dist_d2s[dist_d2s < dist_thred3]) / len(dist_d2s)

                    print('compute gt2data ...')
                    nn_engine.fit(data_pcd)
                    dist_s2d, idx_s2d = nn_engine.kneighbors(stl, n_neighbors=1, return_distance=True)
                    mean_s2d = dist_s2d[dist_s2d < max_dist].mean()

                    recall_1 = len(dist_s2d[dist_s2d < dist_thred1]) / len(dist_s2d)
                    recall_2 = len(dist_s2d[dist_s2d < dist_thred2]) / len(dist_s2d)
                    recall_3 = len(dist_s2d[dist_s2d < dist_thred3]) / len(dist_s2d)

                    over_all = (mean_d2s + mean_s2d) / 2
                    fscore_1 = 2 * precision_1 * recall_1 / (precision_1 + recall_1 + 1e-8)
                    fscore_2 = 2 * precision_2 * recall_2 / (precision_2 + recall_2 + 1e-8)
                    fscore_3 = 2 * precision_3 * recall_3 / (precision_3 + recall_3 + 1e-8)

                    accuracy_list.append(mean_d2s)
                    completion_list.append(mean_s2d)
                    chamfer_distance_list.append(over_all)
                    fscore_1_list.append(fscore_1)
                    precision_1_list.append(precision_1)
                    recall_1_list.append(recall_1)
                    fscore_2_list.append(fscore_2)
                    fscore_3_list.append(fscore_3)

                    print(f'chamfer_distance_1m: {over_all}; mean_d2gt_1m: {mean_d2s}; mean_gt2d_1m: {mean_s2d}.')
                    print(f'precision_10cm: {precision_3};  recall_10cm: {recall_3};  fscore_10cm: {fscore_3}')
                    print(f'precision_5cm: {precision_1};  recall_5cm: {recall_1};  fscore_5cm: {fscore_1}')
                    print(f'precision_2cm: {precision_2};  recall_2cm: {recall_2};  fscore_2cm: {fscore_2}')

                    file.write(f'{obj_id}' + '\n')
                    file.write(f'chamfer_distance_1m: {over_all}; mean_d2gt_1m: {mean_d2s}; mean_gt2d_1m: {mean_s2d}.' + '\n')
                    file.write(f'precision_10cm: {precision_3};  recall_10cm: {recall_3};  fscore_10cm: {fscore_3}' + '\n')
                    file.write(f'precision_5cm: {precision_1};  recall_5cm: {recall_1};  fscore_5cm: {fscore_1}' + '\n')
                    file.write(f'precision_2cm: {precision_2};  recall_2cm: {recall_2};  fscore_2cm: {fscore_2}' + '\n')
                    file.write('\n')

                accuracy_avg = sum(accuracy_list) / len(accuracy_list)
                completion_avg = sum(completion_list) / len(completion_list)
                chamfer_distance_avg = sum(chamfer_distance_list) / len(chamfer_distance_list)
                f_score1_avg = sum(fscore_1_list) / len(fscore_1_list)
                precision_1_avg = sum(precision_1_list) / len(precision_1_list)
                recall_1_avg = sum(recall_1_list) / len(recall_1_list)
                f_score2_avg = sum(fscore_2_list) / len(fscore_2_list)
                f_score3_avg = sum(fscore_3_list) / len(fscore_3_list)

                file.write(f'accuracy_avg (1m): {accuracy_avg};' + '\n')
                file.write(f'completion_avg (1m): {completion_avg};' + '\n')
                file.write(f'chamfer_distance_avg (1m): {chamfer_distance_avg};' + '\n')
                file.write(f'fscore_3_avg (10cm): {f_score3_avg};' + '\n')
                file.write(f'fscore_1_avg (5cm): {f_score1_avg};' + '\n')
                file.write(f'precision_1_avg (5cm): {precision_1_avg};' + '\n')
                file.write(f'recall_1_avg (5cm): {recall_1_avg};' + '\n')
                file.write(f'fscore_2_avg (2cm): {f_score2_avg};' + '\n')

                file.close()