import os
from shutil import copyfile
import cv2
import numpy as np
from tqdm import tqdm
import json
import shutil

import clip
from PIL import Image
import torch
from pyquaternion import Quaternion
from sklearn.cluster import KMeans

def get_square_bbox(bbox, image_shape, scale=1.0):
    '''
    - bbox: [min_x, min_y, max_x, max_y]
    - image_shape: [W, H]
    '''
    
    side_length = max(bbox[2]-bbox[0], bbox[3]-bbox[1]) * scale
    ## make sure the square box can be bounded inside the image.
    side_length = min(side_length, image_shape[0], image_shape[1])
    
    ## change the bbox to square shape with the same center
    center = np.array([(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2])
    new_min_x = max(0, int(center[0] - side_length/2))
    new_min_y = max(0, int(center[1] - side_length/2))
    new_max_x = new_min_x + side_length
    new_max_y = new_min_y + side_length
    
    # import pdb; pdb.set_trace()
    if new_max_x >= image_shape[0]:
        new_max_x = image_shape[0]
        new_min_x = new_max_x - side_length
    
    if new_max_y >= image_shape[1]:
        new_max_y = image_shape[1]
        new_min_y = new_max_y - side_length
    
    return np.array([new_min_x, new_min_y, new_max_x, new_max_y]).astype(int)
        
        
def calculate_clip_scores(instance_images, instance_mask, device, save_dir):
    with open(f'{save_dir}/semantic.txt', 'r') as f:
        semantic_label = f.readlines()[0].strip()
    
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    clip_mean = (0.48145466, 0.4578275, 0.40821073)
    clip_std = (0.26862954, 0.26130258, 0.27577711)
    clip_mean = torch.tensor(clip_mean).to(device).reshape(1, 3, 1, 1)
    clip_std = torch.tensor(clip_std).to(device).reshape(1, 3, 1, 1)
    clip_score_file = f'{save_dir}/clip_score_before_filtered.json'
    # import pdb; pdb.set_trace()
    if not os.path.exists(clip_score_file):
        clip_scores = {}
        for idx,img in enumerate(instance_images):
            # import pdb; pdb.set_trace()
            
            img_mask = instance_mask[idx]
            ## [min_x, min_y, max_x, max_y]
            img_bbox = np.array([np.min(np.where(img_mask)[1]), np.min(np.where(img_mask)[0]), np.max(np.where(img_mask)[1]), np.max(np.where(img_mask)[0])])
            ## modify the bbox to square shape
            img_square_bbox = get_square_bbox(img_bbox, (img.shape[1], img.shape[0]))
            
            img = img[img_square_bbox[1]:img_square_bbox[3], img_square_bbox[0]:img_square_bbox[2], :]
            
            
            pil_image = Image.fromarray(cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_BGR2RGB))
            processed_img = clip_preprocess(pil_image).unsqueeze(0).to(device)
            img_feature = clip_model.encode_image(processed_img)
            text_feature = clip_model.encode_text(clip.tokenize(semantic_label).to(device))
            
            clip_scores[idx] = torch.cosine_similarity(img_feature, text_feature).item()
            
        with open(clip_score_file, 'w') as f:
            json.dump(clip_scores, f)
    else:
        with open(clip_score_file, 'r') as f:
            clip_scores = json.load(f)
        
        assert len(clip_scores) == len(instance_images)

        ## replace the keys from str to int
        clip_scores = {int(k):v for k,v in clip_scores.items()}
        print(f'Loaded clip scores from {clip_score_file}')
    
    return clip_scores

def filter_image_idx(instance_image_idx, clip_scores, pose_dir, save_dir, n_clusters = 20):
    ## load all poses in instance_image_idx
    pose_files = os.listdir(pose_dir)
    pose_files = [f for f in pose_files if f.endswith('.txt')]
    pose_files = [f for f in pose_files if int(f.split('.')[0]) in instance_image_idx]
    pose_mats = [np.loadtxt(os.path.join(pose_dir, f)) for f in pose_files]
    
    quatanions_all = []
    for pose in pose_mats:
        rotation_mat = pose[:3,:3]
        
        rotation_quat = Quaternion(matrix=rotation_mat, atol=1e-05)
        
        trans_vector = pose[:3,3]
        quatanions_all.append(np.concatenate([trans_vector, np.array(rotation_quat.elements)]))
        
        # quatanions_all.append(np.array(rotation_quat.elements))

    # import pdb; pdb.set_trace()
    ## cluster the quatanions
    clustered_quatanions = KMeans(n_clusters=n_clusters, random_state=0).fit(quatanions_all)
    clustered_img_idx_and_scores = {}
    
    clustering_selection_path = f'{save_dir}/clustering_selection'
    if not os.path.exists(clustering_selection_path):
        os.makedirs(clustering_selection_path)
    
    output_idx = []
    new_clip_scores = {}
    for idx_cluster in range(n_clusters):
        clustered_img_idx = np.array(range(len(instance_image_idx)))[clustered_quatanions.labels_ == idx_cluster]
        clustered_img_idx_and_scores[idx_cluster] = {}
        for ii in clustered_img_idx:
            clustered_img_idx_and_scores[idx_cluster][ii] = clip_scores[ii]                 
                            
        clustered_img_idx_and_scores[idx_cluster] = sorted(clustered_img_idx_and_scores[idx_cluster].items(), key=lambda x: x[1], reverse=True)
        
        ## find top-1 images from each cluster
        img_idx = clustered_img_idx_and_scores[idx_cluster][0][0]
        # score = clustered_img_idx_and_scores[idx_cluster][0][1]
        # cv2.imwrite(f'{clustering_selection_path}/{idx_cluster:03d}_{score:.3f}_{img_idx:03d}.png', (self.images_np[img_idx]*255).astype(np.uint8))
        
        ## img_idx: relative index in instance_image_idx
        ## instance_image_idx[img_idx]: absolute index in all images
        output_idx.append(instance_image_idx[img_idx])

        new_clip_scores[str(img_idx)] = clip_scores[img_idx]

    # sample_num = n_clusters
    # ## uniform sample from instance images
    # output_idx = instance_image_idx[0::len(instance_image_idx)//sample_num]
    
    return output_idx, new_clip_scores
    

if __name__ == "__main__":
    W, H = 640, 480
    # for scene_name in ['scene0549_00']:
    scannet_root = "./scannet"
    
    for scene_name in ['scene0008_00', 'scene0005_00', 'scene0050_00', 'scene0461_00', 'scene0549_00', 'scene0616_00']:
        data_dir = os.path.join(scannet_root, scene_name+"_scannet")
        
        inst_dir = os.path.join(data_dir, "instance-filt")
        
        inst_list = os.listdir(inst_dir)
        
        # background semantic classes: 
        # {1: 'wall', 5: 'chair', 23: 'book', 2: 'floor', 8: 'door', 40: 'others', 9: 'window', 7: 'table', 39: '', 18: 'pillow', 11: 'picture', 22: 'ceiling', 29: 'box', 3: 'cabinet', 14: 'desk', 15: 'shelves', 27: 'towel', 6: 'sofa', 34: 'sink', 35: 'lamp', 4: 'bed', 10: 'bookshelf', 19: 'mirror', 16: 'curtain', 30: 'whiteboard', 33: 'toilet', 37: 'bag', 21: 'clothes', 32: 'night stand', 25: 'television', 17: 'dresser', 24: 'refridgerator', 28: 'shower curtain', 36: 'bathtub', 12: 'counter', 38: 'glass', 20: 'floor mat', 26: 'paper', 31: 'person', 13: 'blinds'}
        background_cls_list_nyu40 = [0, 1, 2, 8, 9, 11, 13, 16, 20, 22, 38]
        background_cls_list_scannet = [0, 96, 156] # 156 fireplace
    
        scannet_label_to_text_nyu40 = {}
        scannet_label_to_text_scannet = {}
        with open(os.path.join(scannet_root, "scannetv2-labels.combined.tsv")) as f:
            scannet_label_xlxs = f.readlines()
        
        print(scannet_label_xlxs[0])
        
        for line in scannet_label_xlxs[1:]:
            # print(line)
            semantic_label_nyu40 = line.split('\t')[4]
            
            semantic_label_scannet = line.split('\t')[0]
            semantic_text_scannet = line.split('\t')[1].strip()
            scannet_label_to_text_scannet[semantic_label_scannet] = semantic_text_scannet
            # print(semantic_label_scannet, semantic_text_scannet)
            
            if int(semantic_label_nyu40) in background_cls_list_nyu40:
                # print(semantic_label_nyu40, semantic_text_nyu40, semantic_label_scannet, semantic_text_scannet)
                background_cls_list_scannet.append(int(semantic_label_scannet))
        
        # scannet_label_to_text_nyu40 = sorted(scannet_label_to_text_nyu40.items(), key=lambda x: x[0])
        # scannet_label_to_text_nyu40 = dict(scannet_label_to_text_nyu40)
        
        scannet_label_to_text_scannet['0'] = 'invalid'
            
        with open(os.path.join(scannet_root, "scannetv2-labels.combined.scannet.json"), 'w') as f:
            json.dump(scannet_label_to_text_scannet, f)

        
        object_ids_path = os.path.join(data_dir, "object_ids.txt")
        if os.path.exists(object_ids_path):
            object_ids = set()
            with open(object_ids_path, 'r') as f:
                for line in f.readlines():
                    object_ids.add(int(line.strip()))
        else:
            object_ids = set()
            for idx in tqdm(inst_list):
                inst_data = cv2.imread(os.path.join(inst_dir, idx), cv2.IMREAD_UNCHANGED).astype(np.int32)
            
                object_ids.update(set(np.unique(inst_data)))
            
            with open(object_ids_path, 'w') as f:
                for object_id in object_ids:
                    f.write(str(object_id)+'\n')
        
        
        for object_id in object_ids:
        # for object_id in [8]:
        
            data_dir = os.path.join(scannet_root, scene_name+"_scannet")
            # data_dir = "/data/yesheng/3D-Scene/NeuRIS/dataset/indoor/scene0616_00_scannet"
            inst_dir = os.path.join(data_dir, "instance-filt")
            
            semantic_dir_scannet = os.path.join(data_dir, "label-filt")
            color_dir = os.path.join(data_dir, "color")
            pose_dir = os.path.join(data_dir, 'pose')
            depth_dir = os.path.join(data_dir, 'depth')
            inst_list = os.listdir(inst_dir)

            save_dir = os.path.join(scannet_root, f"object_original_with_clip/{scene_name}_scannet_obj_{object_id}")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            # else:
            #     continue
            pose_save_dir = os.path.join(save_dir, 'pose')
            if not os.path.exists(pose_save_dir):
                os.makedirs(pose_save_dir)
            depth_save_dir = os.path.join(save_dir, 'depth')
            if not os.path.exists(depth_save_dir):
                os.makedirs(depth_save_dir)
            image_save_dir = os.path.join(save_dir, 'image_before_filtered')
            if not os.path.exists(image_save_dir):
                os.makedirs(image_save_dir)
            inst_save_dir = os.path.join(save_dir, 'inst')
            if not os.path.exists(inst_save_dir):
                os.makedirs(inst_save_dir)
            
            mask_save_dir = os.path.join(save_dir, 'mask_before_filtered')
            if not os.path.exists(mask_save_dir):
                os.makedirs(mask_save_dir)
            
            origin_save_dir = os.path.join(save_dir, 'origin_image_before_filtered')
            if not os.path.exists(origin_save_dir):
                os.makedirs(origin_save_dir)

            instance_image_idx = []
            instance_mask = []
            instance_images = []

            num = 0
            
            for idx in tqdm(inst_list):
                inst_data = cv2.imread(os.path.join(inst_dir, idx), cv2.IMREAD_UNCHANGED).astype(np.int32)

                inst_mask = (inst_data == object_id)
                
                if object_id == 8 and scene_name == 'scene0580_00':
                    inst_mask = np.logical_or(inst_mask, (inst_data == 9))
                    inst_mask = np.logical_or(inst_mask, (inst_data == 18))

                # first erode then dilate
                inst_mask = cv2.erode(inst_mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
                inst_mask = cv2.dilate(inst_mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
                
                inst_mask = inst_mask > 0.5
                
                if inst_mask.sum() < 40000:
                    # print(inst_mask.sum())
                    continue
                
                # import pdb; pdb.set_trace()
                
                semantic_map = cv2.imread(os.path.join(semantic_dir_scannet, idx), cv2.IMREAD_UNCHANGED).astype(np.int32)
                this_semantic_label_scannet = np.unique(semantic_map[inst_mask])[0] 
                
                if this_semantic_label_scannet in background_cls_list_scannet:
                    break  
                
                
                # print(os.path.join(color_dir, idx[:-4] + '.jpg'))
                color_data = cv2.imread(os.path.join(color_dir, idx[:-4] + '.jpg')).astype(np.uint8)
                cv2.imwrite(os.path.join(origin_save_dir, "%d"%(num) + '.jpg'), color_data)
                color_data[~inst_mask] = 0
                cv2.imwrite(os.path.join(image_save_dir, "%d"%(num) + '.jpg'), color_data)
                
                
                cv2.imwrite(os.path.join(mask_save_dir, "%d"%(num) + '.png'), inst_mask.astype(np.uint8)*255)
                
                instance_images.append(color_data)
                instance_mask.append(inst_mask)
                instance_image_idx.append(int(idx[:-4]))

                num += 1
                    
            if this_semantic_label_scannet in background_cls_list_scannet:
                shutil.rmtree(save_dir, ignore_errors=True)
                
                print(object_id, this_semantic_label_scannet, scannet_label_to_text_scannet[str(this_semantic_label_scannet)])
                
                continue
                
            if len(instance_image_idx) < 20:
                # import pdb; pdb.set_trace()
                shutil.rmtree(save_dir, ignore_errors=True)
                
                print(object_id, this_semantic_label_scannet, 'filter this small object!')
                
                continue
            
            this_semantic_text_scannet = scannet_label_to_text_scannet[str(this_semantic_label_scannet)]
            with open(os.path.join(save_dir, 'semantic.txt'), 'w') as f:
                f.write(this_semantic_text_scannet)
            
            
            # import pdb; pdb.set_trace()
            ## calculate clip scores
            device = 'cuda:0'
            clip_scores = calculate_clip_scores(instance_images, instance_mask, device, save_dir)
            
            # import pdb; pdb.set_trace()
            
            instance_image_idx, new_clip_scores = filter_image_idx(instance_image_idx, clip_scores, pose_dir, save_dir, n_clusters=min(max(20, len(instance_image_idx)//10), 40))
            
            new_clip_score_path = os.path.join(save_dir, 'clip_scores.json')
            with open(new_clip_score_path, 'w') as f:
                json.dump(new_clip_scores, f)
            
            image_save_dir = os.path.join(save_dir, 'image')
            if not os.path.exists(image_save_dir):
                os.makedirs(image_save_dir)
            mask_save_dir = os.path.join(save_dir, 'mask')
            if not os.path.exists(mask_save_dir):
                os.makedirs(mask_save_dir)
            origin_save_dir = os.path.join(save_dir, 'origin_image')
            if not os.path.exists(origin_save_dir):
                os.makedirs(origin_save_dir)
                

            # import pdb; pdb.set_trace()
            
            for i, idx in enumerate(instance_image_idx):
                copyfile(os.path.join(pose_dir, str(idx) + '.txt'), os.path.join(pose_save_dir, "%d"%(i) + '.txt'))
                copyfile(os.path.join(depth_dir, str(idx) + '.png'), os.path.join(depth_save_dir, "%d"%(i) + '.png'))
                copyfile(os.path.join(inst_dir, str(idx) + '.png'), os.path.join(inst_save_dir, "%d"%(i) + '.png'))
                
                color_data = cv2.imread(os.path.join(color_dir, str(idx) + '.jpg')).astype(np.uint8)
                cv2.imwrite(os.path.join(origin_save_dir, "%d"%(i) + '.jpg'), color_data)
                
                inst_data = cv2.imread(os.path.join(inst_dir, str(idx) + '.png'), cv2.IMREAD_UNCHANGED).astype(np.int32)

                inst_mask = (inst_data == object_id)
                if object_id == 8 and scene_name == 'scene0580_00':
                    inst_mask = np.logical_or(inst_mask, (inst_data == 9))
                    inst_mask = np.logical_or(inst_mask, (inst_data == 18))

                # first erode then dilate
                inst_mask = cv2.erode(inst_mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
                inst_mask = cv2.dilate(inst_mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
                
                inst_mask = inst_mask > 0.5
                
                color_data[~inst_mask] = 0
                cv2.imwrite(os.path.join(image_save_dir, "%d"%(i) + '.jpg'), color_data)
                
                
                cv2.imwrite(os.path.join(mask_save_dir, "%d"%(i) + '.png'), inst_mask.astype(np.uint8)*255)
            
            # intrisic
            intrin_save_dir = os.path.join(save_dir, 'intrinsic')
            if not os.path.exists(intrin_save_dir):
                os.makedirs(intrin_save_dir)
            intrin_dir = os.path.join(data_dir, 'intrinsic')
            
            copyfile(os.path.join(intrin_dir, 'extrinsic_color.txt'), os.path.join(intrin_save_dir, 'extrinsic_color.txt'))
            copyfile(os.path.join(intrin_dir, 'extrinsic_depth.txt'), os.path.join(intrin_save_dir, 'extrinsic_depth.txt'))
            copyfile(os.path.join(intrin_dir, 'intrinsic_color.txt'), os.path.join(intrin_save_dir, 'intrinsic_color.txt'))
            copyfile(os.path.join(intrin_dir, 'intrinsic_depth.txt'), os.path.join(intrin_save_dir, 'intrinsic_depth.txt'))


            # gt mesh
            copyfile(os.path.join(data_dir, f'{scene_name}_vh_clean_2.ply'), os.path.join(save_dir, f'{scene_name}_vh_clean_2.ply'))
            copyfile(os.path.join(data_dir, f'{scene_name}_vh_clean.ply'), os.path.join(save_dir, f'{scene_name}_vh_clean.ply'))
            
            print(object_id, "Finish!")
            
