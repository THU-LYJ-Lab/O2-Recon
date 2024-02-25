import torch
import numpy as np
from diffusers import StableDiffusionInpaintPipeline
import PIL.Image as Image
import cv2
import os

sd_pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "/home/huyb/.cache/huggingface/hub/models--stabilityai--stable-diffusion-2-inpainting/snapshots/8efdbe41a7ec336beae3084a750d7847443f5b14/",
        torch_dtype=torch.float32,
    ).to('cuda')



def sample_points_inside_mask(mask, points_num):
    ## generate grid points in the whole image, and then mask out the points outside the mask
    h, w = mask.shape
    x = np.linspace(0, w-1, 20).round().astype(np.int32)
    y = np.linspace(0, h-1, 20).round().astype(np.int32)
    xx, yy = np.meshgrid(x, y)
    points = np.stack([xx, yy], axis=-1).reshape(-1, 2)
    points = points[mask[points[:, 1], points[:, 0]]]
    points = points[np.random.choice(points.shape[0], points_num, replace=True)]
    
    return points
    

def crop_for_filling_pre(image: np.array, mask: np.array, crop_size: int = 512):
    # Calculate the aspect ratio of the image
    height, width = image.shape[:2]
    aspect_ratio = float(width) / float(height)

    # If the shorter side is less than 512, resize the image proportionally
    if min(height, width) < crop_size:
        if height < width:
            new_height = crop_size
            new_width = int(new_height * aspect_ratio)
        else:
            new_width = crop_size
            new_height = int(new_width / aspect_ratio)

        image = cv2.resize(image, (new_width, new_height))
        mask = cv2.resize(mask, (new_width, new_height))

    # Find the bounding box of the mask
    x, y, w, h = cv2.boundingRect(mask)
    
    # # scale the bounding box
    # scale = 1.2
    # center = (x + w // 2, y + h // 2)
    # img_w = image.shape[1]
    # img_h = image.shape[0]
    # new_x = max(0, center[0] - int(w * scale / 2))
    # new_y = max(0, center[1] - int(h * scale / 2))
    # new_w = min(img_w - 1 - new_x, int(w * scale))
    # new_h = min(img_h - 1 - new_y, int(h * scale))
    
    # x,y,w,h = new_x, new_y, new_w, new_h
    

    # Update the height and width of the resized image
    height, width = image.shape[:2]
    
    # # If the 512x512 square cannot cover the entire mask, resize the image accordingly
    if w > crop_size or h > crop_size:
        # padding to square at first
        if height < width:
            padding = width - height
            image = np.pad(image, ((padding // 2, padding - padding // 2), (0, 0), (0, 0)), 'constant')
            mask = np.pad(mask, ((padding // 2, padding - padding // 2), (0, 0)), 'constant')
        else:
            padding = height - width
            image = np.pad(image, ((0, 0), (padding // 2, padding - padding // 2), (0, 0)), 'constant')
            mask = np.pad(mask, ((0, 0), (padding // 2, padding - padding // 2)), 'constant')

        resize_factor = crop_size / max(w, h)
        image = cv2.resize(image, (0, 0), fx=resize_factor, fy=resize_factor)
        mask = cv2.resize(mask, (0, 0), fx=resize_factor, fy=resize_factor)
        x, y, w, h = cv2.boundingRect(mask)

    # Calculate the crop coordinates
    crop_x = min(max(x + w // 2 - crop_size // 2, 0), width - crop_size)
    crop_y = min(max(y + h // 2 - crop_size // 2, 0), height - crop_size)
    
    if crop_x + crop_size > image.shape[1]:
        crop_x = image.shape[1] - crop_size
    if crop_y + crop_size > image.shape[0]:
        crop_y = image.shape[0] - crop_size

    # Crop the image
    cropped_image = image[crop_y:crop_y + crop_size, crop_x:crop_x + crop_size]
    cropped_mask = mask[crop_y:crop_y + crop_size, crop_x:crop_x + crop_size]

    return cropped_image, cropped_mask


def crop_for_filling_post(
        image: np.array,
        mask: np.array,
        filled_image: np.array, 
        crop_size: int = 512,
        ):
    image_copy = image.copy()
    mask_copy = mask.copy()
    # Calculate the aspect ratio of the image
    height, width = image.shape[:2]
    height_ori, width_ori = height, width
    aspect_ratio = float(width) / float(height)

    # If the shorter side is less than 512, resize the image proportionally
    if min(height, width) < crop_size:
        if height < width:
            new_height = crop_size
            new_width = int(new_height * aspect_ratio)
        else:
            new_width = crop_size
            new_height = int(new_width / aspect_ratio)

        image = cv2.resize(image, (new_width, new_height))
        mask = cv2.resize(mask, (new_width, new_height))

    # Find the bounding box of the mask
    x, y, w, h = cv2.boundingRect(mask)
    
    # Update the height and width of the resized image
    height, width = image.shape[:2]

    # # If the 512x512 square cannot cover the entire mask, resize the image accordingly
    if w > crop_size or h > crop_size:
        flag_padding = True
        # padding to square at first
        if height < width:
            padding = width - height
            image = np.pad(image, ((padding // 2, padding - padding // 2), (0, 0), (0, 0)), 'constant')
            mask = np.pad(mask, ((padding // 2, padding - padding // 2), (0, 0)), 'constant')
            padding_side = 'h'
        else:
            padding = height - width
            image = np.pad(image, ((0, 0), (padding // 2, padding - padding // 2), (0, 0)), 'constant')
            mask = np.pad(mask, ((0, 0), (padding // 2, padding - padding // 2)), 'constant')
            padding_side = 'w'

        resize_factor = crop_size / max(w, h)
        image = cv2.resize(image, (0, 0), fx=resize_factor, fy=resize_factor)
        mask = cv2.resize(mask, (0, 0), fx=resize_factor, fy=resize_factor)
        x, y, w, h = cv2.boundingRect(mask)
    else:
        flag_padding = False

    # Calculate the crop coordinates
    crop_x = min(max(x + w // 2 - crop_size // 2, 0), width - crop_size)
    crop_y = min(max(y + h // 2 - crop_size // 2, 0), height - crop_size)
    
    if crop_x + crop_size > image.shape[1]:
        crop_x = image.shape[1] - crop_size
    if crop_y + crop_size > image.shape[0]:
        crop_y = image.shape[0] - crop_size


    # Fill the image
    image[crop_y:crop_y + crop_size, crop_x:crop_x + crop_size] = filled_image
    if flag_padding:
        image = cv2.resize(image, (0, 0), fx=1/resize_factor, fy=1/resize_factor)
        if padding_side == 'h':
            image = image[padding // 2:padding // 2 + height_ori, :]
        else:
            image = image[:, padding // 2:padding // 2 + width_ori]

    image = cv2.resize(image, (width_ori, height_ori))

    image_copy[mask_copy==255] = image[mask_copy==255]
    return image_copy

    
def fill_img_with_sd(
        img: np.ndarray,
        mask: np.ndarray,
        text_prompt: str,
        pipe,
):
    print(img.shape, mask.shape)
    
    img_crop, mask_crop = crop_for_filling_pre(img, mask)
    # cv2.imwrite('img_crop.png', img_crop)
    # cv2.imwrite('mask_crop.png', mask_crop)
    # exit(0)
    
    img_crop_filled = pipe(
        prompt=text_prompt,
        image=Image.fromarray(img_crop),
        mask_image=Image.fromarray(mask_crop)
    ).images[0]
    
    # cv2.imwrite('img_crop_filled.png', np.array(img_crop_filled))
    
    img_filled = crop_for_filling_post(img, mask, np.array(img_crop_filled))
    return img_filled


def generate_new_image_and_mask(root_dir):
    '''
    Here we take in inpaint-mask for only one image, predict the color and depth.
    Reproject the mask to other images, and inpaint the other images.
    '''
    
    img_dir = os.path.join(root_dir, 'image')
    new_img_dir = os.path.join(root_dir, 'image_filled')
    
    mask_dir = os.path.join(root_dir, 'mask')
    new_mask_dir = os.path.join(root_dir, 'mask_filled')
    
    if not os.path.exists(new_img_dir):
        os.makedirs(new_img_dir)
    if not os.path.exists(new_mask_dir):
        os.makedirs(new_mask_dir)
        
    new_masked_img_dir = os.path.join(root_dir, 'image_filled_masked')
    if not os.path.exists(new_masked_img_dir):
        os.makedirs(new_masked_img_dir)
    
    depth_dir = os.path.join(root_dir, 'depth')
    pose_dir = os.path.join(root_dir, 'pose') # camera to world
    # import pdb; pdb.set_trace()
    projected_mask_dir = os.path.join(root_dir, 'projected_mask')
    if not os.path.exists(projected_mask_dir):
        os.makedirs(projected_mask_dir)
    filled_depth_img_dir = os.path.join(root_dir, 'filled_depth_img')
    if not os.path.exists(filled_depth_img_dir):
        os.makedirs(filled_depth_img_dir)
    
    inpaint_mask_dir = os.path.join(root_dir, f'seg-{os.path.basename(root_dir).split("_")[0][-4:]}-obj{os.path.basename(root_dir).split("_")[-1]}')
    
    with open(f'{root_dir}/semantic.txt', 'r') as f:
        semantic_label = f.readlines()[0].strip()
        # import pdb; pdb.set_trace()
        
        if "stack of chairs" in semantic_label:
            semantic_label.replace("stack of chairs", "chair")
    semantic_text = 'a '+ semantic_label + ', no background'
    
    semantic_text_for_depth = 'a '+ semantic_label
    
    # import pdb; pdb.set_trace()
    
    img_intrinsic = np.loadtxt(os.path.join(root_dir, 'intrinsic','intrinsic_depth.txt'))[:3,:3]
    img_intrinsic_inv = np.linalg.inv(img_intrinsic)
    
    if 'replica' in os.path.basename(root_dir):
        only_seg_dir = os.path.join(root_dir, f'only-seg-{os.path.basename(root_dir).split("_")[0]}-{os.path.basename(root_dir).split("_")[1]}-obj{os.path.basename(root_dir).split("_")[-1]}')
    else:
        only_seg_dir = os.path.join(root_dir, f'only-seg-{os.path.basename(root_dir).split("_")[0][-4:]}-obj{os.path.basename(root_dir).split("_")[-1]}')
    
    reference_id_list = []
    reference_points_3d_list = []
    
    for name in os.listdir(only_seg_dir):
        if 'class_inpaint mask' in name:
            reference_id_list.append(int(name.split('_')[0]))
    
    for reference_id in reference_id_list:
        reference_img = cv2.imread(os.path.join(img_dir, f'{reference_id}.jpg'))
        reference_depth = cv2.imread(os.path.join(depth_dir, f'{reference_id}.png'), -1)
        reference_pose = np.loadtxt(os.path.join(pose_dir, f'{reference_id}.txt'))
        reference_mask = cv2.imread(os.path.join(mask_dir, f'{reference_id}.png'), 0)
        reference_depth[cv2.resize(reference_mask,(reference_depth.shape[1], reference_depth.shape[0]))<127] = 0
        
        reference_inpaint_mask = cv2.imread(os.path.join(only_seg_dir, f'{reference_id}_class_inpaint mask.png'), 0)
        if reference_inpaint_mask is not None and reference_inpaint_mask.sum() > 0:
            # import pdb; pdb.set_trace()
            # filled_reference_img = fill_img_with_sd(reference_img, reference_inpaint_mask, semantic_text, sd_pipe)
            
            reference_depth_img = reference_depth.copy() / 1000.0
            max_threshold = reference_depth_img.max()*1.1
            reference_depth_img = (reference_depth_img / max_threshold * 255).astype(np.uint8)[...,None].repeat(3, axis=-1)

            depth_filling_mask = reference_inpaint_mask.copy()
            depth_filling_mask[reference_mask>127] = 255
            depth_filling_mask = cv2.resize(depth_filling_mask,(reference_depth_img.shape[1], reference_depth_img.shape[0]))
            depth_filling_mask[reference_depth > 0] = 0
            depth_filling_mask = cv2.dilate(depth_filling_mask, np.ones((3, 3), np.uint8), iterations=2)
            
            cv2.imwrite(os.path.join(filled_depth_img_dir, f'{reference_id}.png'), depth_filling_mask)
            
            
            # filled_depths = []
            # for i in range(8):
            #     filled_depths.append(fill_img_with_sd(reference_depth_img.copy(), depth_filling_mask , semantic_text_for_depth, sd_pipe))

            filled_depths = []
            for i in range(1):
                filled_depths.append(fill_img_with_sd(reference_depth_img.copy(), depth_filling_mask , semantic_text_for_depth, sd_pipe))
            
            max_size_idx = np.argmax([(x>0).sum() for x in filled_depths])
            filled_reference_depth_img = filled_depths[max_size_idx]
            filled_reference_depth = (filled_reference_depth_img / 255.0 * max_threshold)[...,0:1]
            
            cv2_inpaint_depth_img = cv2.inpaint(reference_depth_img, cv2.resize(reference_inpaint_mask,(reference_depth_img.shape[1], reference_depth_img.shape[0])), 3, cv2.INPAINT_TELEA)
            cv2.imwrite(os.path.join(filled_depth_img_dir, f'cv2-{reference_id}.png'), cv2_inpaint_depth_img)
            
            cv2.imwrite(os.path.join(filled_depth_img_dir, f'{reference_id}.png'), filled_reference_depth_img) 
            
            # depth_candidates = []
            # for idx in range(8):
            #     depth_img = cv2.cvtColor(filled_depths[idx], cv2.COLOR_BGR2GRAY)
            #     depth_candidates.append(depth_img)
            #     cv2.imwrite(os.path.join(filled_depth_img_dir, f'{reference_id}_{idx}.png'), depth_img)
            
            # sum_img = np.vstack([np.hstack(depth_candidates[:4]), np.hstack(depth_candidates[4:])])
            # cv2.imwrite(os.path.join(filled_depth_img_dir, f'{reference_id}_sum.png'), sum_img)
            # cv2.imwrite(os.path.join(filled_depth_img_dir, f'{reference_id}_orig.png'), reference_depth_img.copy())
            # import pdb; pdb.set_trace()
                   

            # cv2.imwrite(os.path.join(new_img_dir, img_name), filled_reference_img)            
            
            reference_new_mask = reference_inpaint_mask.copy()
            reference_new_mask[reference_mask>127] = 255
            
            ## back project the mask to 3D points
            # get the coordinated of filled_reference_depth > 0
            reference_points_x = np.where(filled_reference_depth>0)[1]
            reference_points_y = np.where(filled_reference_depth>0)[0]
            reference_points_z = filled_reference_depth[reference_points_y, reference_points_x]
            
            reference_points_3d = np.stack([reference_points_x, reference_points_y, np.ones_like(reference_points_x)], axis=-1)
            
            # unproject from 2D to 3D
            reference_points_3d = np.matmul(img_intrinsic_inv, reference_points_3d.T).T
            reference_points_3d = reference_points_3d * reference_points_z
            
            # transform from camera to world
            reference_points_3d = np.matmul(reference_pose[:3,:3], reference_points_3d.T).T + reference_pose[:3,3]  
            
            reference_points_3d_list.append(reference_points_3d) 
        
    if len(reference_points_3d_list) > 0:
        reference_points_3d_all = np.concatenate(reference_points_3d_list, axis=0) 
        
        # import pdb; pdb.set_trace()
        
        for img_name in os.listdir(img_dir):
            
            img = cv2.imread(os.path.join(img_dir, img_name))

            # mask = cv2.imread(os.path.join(mask_dir, img_name), 0)

            mask = cv2.imread(os.path.join(mask_dir, img_name.replace('.jpg', '.png')), 0)
            
            img_mask = np.array(mask) > 0
            
            idx = img_name.split('.')[0]
            this_pose = np.loadtxt(os.path.join(pose_dir, f'{idx}.txt'))
            
            this_pose_inv = np.linalg.inv(this_pose)
            
            this_points_3d = np.matmul(this_pose_inv[:3,:3], reference_points_3d_all.T).T + this_pose_inv[:3,3]
            this_points_3d = this_points_3d / this_points_3d[:,2:3]
            
            # project from 3D to 2D
            this_points_3d = np.matmul(img_intrinsic, this_points_3d.T).T
            
            this_points_3d = this_points_3d.round().astype(np.int32)
            
            valid_mask = (this_points_3d[:,0] >= 0) & (this_points_3d[:,0] < reference_depth.shape[1]) & (this_points_3d[:,1] >= 0) & (this_points_3d[:,1] < reference_depth.shape[0])
            this_points_3d = this_points_3d[valid_mask]
            
            this_mask_from_projection = np.zeros_like(reference_depth).astype(np.uint8)
            this_mask_from_projection[this_points_3d[:,1], this_points_3d[:,0]] = 255
            cv2.imwrite(os.path.join(projected_mask_dir, f'{idx}.png'), this_mask_from_projection)
            # erode and dilate
            # this_mask_from_projection = cv2.erode(this_mask_from_projection, np.ones((3, 3), np.uint8), iterations=1)
            # convolve by 3x3 all-ones kernel
            kernel = torch.ones(3, 3).to(torch.float32)
            this_mask_from_projection_conv = torch.nn.functional.conv2d(torch.from_numpy(this_mask_from_projection.astype(np.float32)).unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), padding=1).squeeze().numpy()
            this_mask_from_projection[this_mask_from_projection_conv<255*5] = 0
            
            this_mask_from_projection = cv2.dilate(this_mask_from_projection, np.ones((5, 5), np.uint8), iterations=2)
            this_mask_from_projection = cv2.dilate(this_mask_from_projection, np.ones((3, 3), np.uint8), iterations=1)
            
            cv2.imwrite(os.path.join(projected_mask_dir, f'{idx}-dilate.png'), this_mask_from_projection)
            
            ## project reference_new_mask from reference_pose to this_pose
            
            
            inpaint_mask_annot = cv2.imread(os.path.join(inpaint_mask_dir, f'{idx}_class_inpaint mask.png'), 0)
            
            inpaint_mask = this_mask_from_projection.copy()
            inpaint_mask = cv2.resize(inpaint_mask, (img.shape[1], img.shape[0]))
            
            ## will not inpaint inside the image mask
            inpaint_mask[img_mask] = 0
            # dilation after filter the in-mask areas
            # inpaint_mask = cv2.imread(os.path.join(projected_mask_dir, f'{idx}-inpaint.png'))[...,0]
            
            inpaint_mask = cv2.dilate(inpaint_mask, np.ones((5, 5), np.uint8), iterations=2)
            inpaint_mask = cv2.dilate(inpaint_mask, np.ones((3, 3), np.uint8), iterations=1)
            if inpaint_mask_annot is None:
                inpaint_mask_annot = np.zeros_like(inpaint_mask)
            
            
            if inpaint_mask is not None and inpaint_mask.sum() > 0:
                # import pdb; pdb.set_trace()
                ##  inpaint with calculated mask
                cv2.imwrite(os.path.join(projected_mask_dir, f'{idx}-inpaint.png'), inpaint_mask)
                
                filled_img = fill_img_with_sd(img, inpaint_mask, semantic_text, sd_pipe)

                cv2.imwrite(os.path.join(new_img_dir, img_name), filled_img)            
                
                ## Judge the bad in-painted areas, according to gray values.
                ## Sometimes stable diffusion inpaint with background black, which is not good.
                filled_img_gray = cv2.cvtColor(filled_img, cv2.COLOR_BGR2GRAY)
                inpaint_mask[filled_img_gray<10] = 0
                
                vis_temp_mask = this_mask_from_projection.copy()
                vis_temp_mask[this_mask_from_projection>127] = 127
                vis_temp_mask[cv2.resize(img_mask.astype(np.uint8)*255, ((vis_temp_mask.shape[1], vis_temp_mask.shape[0])))>127] = 255
                vis_temp_mask = cv2.resize(vis_temp_mask, (inpaint_mask.shape[1], inpaint_mask.shape[0]))
                
                cv2.imwrite(os.path.join(projected_mask_dir, f'{idx}-dilate-inpaint.png'), cv2.hconcat([inpaint_mask_annot, inpaint_mask, vis_temp_mask]))
                cv2.imwrite(os.path.join(projected_mask_dir, f'{idx}-usable.png'), inpaint_mask) 

                new_mask = inpaint_mask.copy()
                new_mask[img_mask] = 255
                # import pdb; pdb.set_trace()
                cv2.imwrite(os.path.join(new_mask_dir, img_name.replace('jpg', 'png')), (new_mask).astype(np.uint8))
                cv2.imwrite(os.path.join(new_masked_img_dir, img_name), filled_img)
            
            else:
                cv2.imwrite(os.path.join(new_img_dir, img_name), img)
                cv2.imwrite(os.path.join(new_masked_img_dir, img_name), img)
                cv2.imwrite(os.path.join(new_mask_dir, img_name.replace('jpg', 'png')), mask)
                cv2.imwrite(os.path.join(projected_mask_dir, f'{idx}-usable.png'), np.zeros_like(mask)) 
    else:
        ## No reference image, just use the original image.
        ## Don't need in-painting.
        for img_name in os.listdir(img_dir):
            # if os.path.exists(os.path.join(new_masked_img_dir, img_name)):
            #     continue
            
            img = cv2.imread(os.path.join(img_dir, img_name))

            # mask = cv2.imread(os.path.join(mask_dir, img_name), 0)

            mask = cv2.imread(os.path.join(mask_dir, img_name.replace('.jpg', '.png')), 0)
            
            img_mask = np.array(mask) > 0
            
            idx = img_name.split('.')[0]
            

            cv2.imwrite(os.path.join(new_img_dir, img_name), img)
            cv2.imwrite(os.path.join(new_masked_img_dir, img_name), img)
            cv2.imwrite(os.path.join(new_mask_dir, img_name.replace('jpg', 'png')), mask)
            cv2.imwrite(os.path.join(projected_mask_dir, f'{idx}-usable.png'), np.zeros_like(mask)) 
        
   
    

def generate_new_normal(root_dir, save_dir):
    inpaint_normal_path = os.path.join(save_dir, 'inpaint_normal')
    if not os.path.exists(inpaint_normal_path):
        os.makedirs(inpaint_normal_path)
    
    project_mask_path = os.path.join(root_dir, 'projected_mask')
    
    pred_normal_path = os.path.join(save_dir, 'pred_normal')
    
    
    with open(f'{root_dir}/semantic.txt', 'r') as f:
        semantic_label = f.readlines()[0].strip()
        # import pdb; pdb.set_trace()
        
        if "stack of chairs" in semantic_label:
            semantic_label.replace("stack of chairs", "chair")
    semantic_text = 'a '+ semantic_label + ', normal map'
    
    for name in os.listdir(pred_normal_path):
        if not name.endswith('.npz'):
            continue
        normal_map = np.load(os.path.join(pred_normal_path, name))['arr_0']
        normal_map = (normal_map + 1) / 2.0 * 255
        normal_map = normal_map.astype(np.uint8)
        
        idx = int(name.split('.')[0])
        
        inpaint_mask = cv2.imread(os.path.join(project_mask_path, f'{idx}-usable.png'), 0)
        
        # import pdb; pdb.set_trace()
        if inpaint_mask.sum() == 0:
            inpaint_normal = normal_map.copy()
            cv2.imwrite(os.path.join(inpaint_normal_path, f'{idx}.png'), inpaint_normal[...,::-1])
        
            cv2.imwrite(os.path.join(inpaint_normal_path, f'{idx}-concat.png'), cv2.hconcat([normal_map[...,::-1], inpaint_normal[...,::-1]]))
            
            inpaint_normal = inpaint_normal.astype(np.float32) / 255.0 * 2.0 - 1.0
            np.savez_compressed(os.path.join(inpaint_normal_path, f'{idx}.npz'), inpaint_normal)
            continue
        
        inpaint_normal = fill_img_with_sd(normal_map, cv2.resize(inpaint_mask,(normal_map.shape[1], normal_map.shape[0])), semantic_text, sd_pipe)
        
        cv2.imwrite(os.path.join(inpaint_normal_path, f'{idx}.png'), inpaint_normal[...,::-1])
        
        cv2.imwrite(os.path.join(inpaint_normal_path, f'{idx}-concat.png'), cv2.hconcat([normal_map[...,::-1], inpaint_normal[...,::-1]]))
        
        inpaint_normal = inpaint_normal.astype(np.float32) / 255.0 * 2.0 - 1.0
        np.savez_compressed(os.path.join(inpaint_normal_path, f'{idx}.npz'), inpaint_normal)
        
        
        
        

def generate_new_depth(root_dir, save_dir):
    inpaint_depth_dir = os.path.join(save_dir, 'inpaint_depth')
    if not os.path.exists(inpaint_depth_dir):
        os.makedirs(inpaint_depth_dir)
    
    project_mask_path = os.path.join(root_dir, 'projected_mask')
    
    depth_dir = os.path.join(save_dir, 'depth')
    
    
    with open(f'{root_dir}/semantic.txt', 'r') as f:
        semantic_label = f.readlines()[0].strip()
        # import pdb; pdb.set_trace()
        
        if "stack of chairs" in semantic_label:
            semantic_label.replace("stack of chairs", "chair")
    semantic_text = 'a '+ semantic_label + ', no background'
    
    for name in os.listdir(depth_dir):
        if not name.endswith('.png'):
            continue
        
        depth = cv2.imread(os.path.join(depth_dir, name), -1)
        depth_img = depth.copy() / 1000.0
        max_threshold = depth_img.max()*1.1
        depth_img = (depth_img / max_threshold * 255).astype(np.uint8)[...,None].repeat(3, axis=-1)
        
        
        idx = int(name.split('.')[0])
        
        # import pdb; pdb.set_trace()
        mask = cv2.imread(os.path.join(root_dir, 'mask', f'{idx}.png'), 0)  
        depth_img[cv2.resize(mask,(depth_img.shape[1], depth_img.shape[0]))<127] = 0
        
        inpaint_mask = cv2.imread(os.path.join(project_mask_path, f'{idx}-usable.png'), 0)
        
        # import pdb; pdb.set_trace()
        
        filled_depths = []
        for i in range(3):
            filled_depths.append(fill_img_with_sd(depth_img, cv2.resize(inpaint_mask,(depth_img.shape[1], depth_img.shape[0])) , semantic_text, sd_pipe))
        
        # import pdb; pdb.set_trace()
        vis_images = [x[...,0] for x in filled_depths]
        vis_images.append(depth_img[...,0])
        vis_images.append(cv2.resize(inpaint_mask,(depth_img.shape[1], depth_img.shape[0]))) 
        # import pdb; pdb.set_trace()
          
        cv2.imwrite(os.path.join(inpaint_depth_dir, f'{idx}-vis-candidates.png'), 
                    cv2.hconcat(vis_images))
        
        max_size_idx = np.argmax([(x>0).sum() for x in filled_depths])
        filled_depth_img = filled_depths[max_size_idx]
        filled_depth = (filled_depth_img / 255.0 * max_threshold)[...,0:1]*1000
        filled_depth = filled_depth.astype(np.uint16)

        cv2.imwrite(os.path.join(inpaint_depth_dir, f'{idx}-vis.png'), filled_depth_img)  
        cv2.imwrite(os.path.join(inpaint_depth_dir, f'{idx}.png'), filled_depth)  
        
        # inpaint_normal = fill_img_with_sd(normal_map, cv2.resize(inpaint_mask,(normal_map.shape[1], normal_map.shape[0])), semantic_text, sd_pipe)
        
        # cv2.imwrite(os.path.join(inpaint_normal_path, f'{idx}.png'), inpaint_normal[...,::-1])
        
        # cv2.imwrite(os.path.join(inpaint_normal_path, f'{idx}-concat.png'), cv2.hconcat([normal_map[...,::-1], inpaint_normal[...,::-1]]))
        
        # inpaint_normal = inpaint_normal.astype(np.float32) / 255.0 * 2.0 - 1.0
        # np.savez_compressed(os.path.join(inpaint_normal_path, f'{idx}.npz'), inpaint_normal)
        
        
        
        
