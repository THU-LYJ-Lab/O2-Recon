import os, argparse, logging
import numpy as np

import preprocess.neuris_data  as neuris_data


from confs.path import lis_name_scenes

import preprocess.sd_sam_inpaint as sd_sam_inpaint

if __name__ == '__main__':
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)

    FORMAT = "[%(filename)s:%(lineno)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', type=str, default='scannet')
    parser.add_argument('--scannet_root', type=str, default=None)
    parser.add_argument('--neus_root', type=str, default=None)
    parser.add_argument('--dir_snu_code', type=str, default=None)
    
    
    args = parser.parse_args()
    

    dataset_type = args.data_type
             
    dir_root_scannet = os.path.join(args.scannet_root, 'object_original_with_clip')
    dir_root_neus = args.neus_root
    dir_snu_code = args.dir_snu_code
        
    if dataset_type == 'scannet-with-inpaint':
        # for scene_name in ['scene0008_00', 'scene0005_00', 'scene0050_00', 'scene0461_00', 'scene0549_00', 'scene0616_00']:
        for scene_name in ['scene0616_00']:
            object_ids = ['9']
            # object_ids = [x.split('_')[-1] for x in os.listdir(dir_root_scannet) if scene_name in x]

            lis_name_scenes = [f'{scene_name}_scannet_obj_{id}' for id in object_ids]

            for scene_name in lis_name_scenes:
                dir_scan = f'{dir_root_scannet}/{scene_name}'
                dir_neus = f'{dir_root_neus}/{scene_name}_inpainted'
                
                sd_sam_inpaint.generate_new_image_and_mask(dir_scan)
                
                neuris_data.prepare_neuris_data_from_scannet(dir_scan, dir_neus, dir_snu_code, sample_interval=1, 
                                                    b_sample = True, 
                                                    b_generate_neus_data = True,
                                                    b_pred_normal = True, 
                                                    b_detect_planes = False,
                                                    b_with_obj_mask= True,
                                                    b_with_inpaint = True,
                                                    is_object = True,
                                                    b_pred_normal_full = True,
                                                    ) 
                
                sd_sam_inpaint.generate_new_normal(root_dir=dir_scan, save_dir=dir_neus)

        
    if dataset_type == 'scannet-selected':
        
        for scene_name in ['scene0580_00']:
            object_ids = [x.split('_')[-1] for x in os.listdir(dir_root_scannet) if scene_name in x]
            object_ids = [1,8]

            lis_name_scenes = [f'{scene_name}_scannet_obj_{id}' for id in object_ids]

            for scene_name in lis_name_scenes:
                dir_scan = f'{dir_root_scannet}/{scene_name}'
                dir_neus = f'{dir_root_neus}/{scene_name}_selected_5'
                
                neuris_data.prepare_neuris_data_from_scannet(dir_scan, dir_neus, dir_snu_code, sample_interval=1, 
                                                    b_sample = True, 
                                                    b_generate_neus_data = True,
                                                    b_pred_normal = True, 
                                                    b_detect_planes = False,
                                                    b_with_obj_mask= True,
                                                    b_with_inpaint = False,
                                                    is_object = True,
                                                    b_pred_normal_full = False,
                                                    ) 
    
    print('Done')
