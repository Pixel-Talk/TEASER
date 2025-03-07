import torch
import cv2
import numpy as np
from skimage.transform import estimate_transform, warp
from src.teaser_encoder import TeaserEncoder
from src.FLAME.FLAME import FLAME
from src.renderer.renderer import Renderer
import argparse
import os
import src.utils.masking as masking_utils
from utils.mediapipe_utils import run_mediapipe
from datasets.base_dataset import create_mask
import torch.nn.functional as F

from tqdm import tqdm

import copy

import pickle

def crop_face(frame, landmarks, scale=1.0, image_size=224):
    left = np.min(landmarks[:, 0])
    right = np.max(landmarks[:, 0])
    top = np.min(landmarks[:, 1])
    bottom = np.max(landmarks[:, 1])

    h, w, _ = frame.shape
    old_size = (right - left + bottom - top) / 2
    center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])

    size = int(old_size * scale)

    # crop image
    src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                        [center[0] + size / 2, center[1] - size / 2]])
    DST_PTS = np.array([[0, 0], [0, image_size - 1], [image_size - 1, 0]])
    tform = estimate_transform('similarity', src_pts, DST_PTS)

    return tform

def deep_copy_with_tensors(obj):
    """
    深度拷贝一个包含 PyTorch 张量的字典。
    """
    if isinstance(obj, torch.Tensor):
        return obj.clone()
    elif isinstance(obj, (list, tuple)):
        return type(obj)(deep_copy_with_tensors(item) for item in obj)
    elif isinstance(obj, dict):
        return {key: deep_copy_with_tensors(value) for key, value in obj.items()}
    else:
        return copy.deepcopy(obj)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_path_source', type=str, default='', help='Path to the source image')
    parser.add_argument('--input_path_target', type=str, default='', help='Path to the image of target expression')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on')
    parser.add_argument('--checkpoint', type=str, default='', help='Path to the checkpoint')
    parser.add_argument('--crop', action='store_true', help='Crop the face using mediapipe')
    parser.add_argument('--out_path', type=str, default='', help='Path to save the output (will be created if not exists)')
    parser.add_argument('--use_teaser_generator', action='store_true', help='Use TEASER neural image to image translator to reconstruct the image')
    parser.add_argument('--render_orig', action='store_true', help='Present the result w.r.t. the original image/video size')

    args = parser.parse_args()

    image_size = 224
    

    # ----------------------- initialize configuration ----------------------- #
    teaser_encoder = TeaserEncoder().to(args.device)
    checkpoint = torch.load(args.checkpoint)
    checkpoint_encoder = {k.replace('teaser_encoder.', ''): v for k, v in checkpoint.items() if 'teaser_encoder' in k} # checkpoint includes both teaser_encoder and teaser_generator

    teaser_encoder.load_state_dict(checkpoint_encoder)
    teaser_encoder.eval()

    if args.use_teaser_generator:
        from src.teaser_generator import TeaserGenerator
        teaser_generator = TeaserGenerator(in_channels=6, out_channels=3, init_features=32, res_blocks=5).to(args.device)

        checkpoint_generator = {k.replace('teaser_generator.', ''): v for k, v in checkpoint.items() if 'teaser_generator' in k} # checkpoint includes both teaser_encoder and teaser_generator
        teaser_generator.load_state_dict(checkpoint_generator)
        teaser_generator.eval()

    # ---- visualize the results ---- #

    flame = FLAME().to(args.device)
    renderer = Renderer().to(args.device)

    # check if input is an image or a video or webcam or directory
   
            
    image1 = cv2.imread(args.input_path_source)
    image2 = cv2.imread(args.input_path_target)

    orig_image_height, orig_image_width, _ = image1.shape

    kpt_mediapipe = run_mediapipe(image1)

    kpt_mediapipe = kpt_mediapipe[..., :2]

    tform = crop_face(image1,kpt_mediapipe,scale=1.4,image_size=image_size)
    
    cropped_image_1 = warp(image1, tform.inverse, output_shape=(224, 224), preserve_range=True).astype(np.uint8)

    cropped_kpt_mediapipe_1 = np.dot(tform.params, np.hstack([kpt_mediapipe, np.ones([kpt_mediapipe.shape[0],1])]).T).T
    cropped_kpt_mediapipe_1 = cropped_kpt_mediapipe_1[:,:2]

    
    cropped_image_1 = cv2.cvtColor(cropped_image_1, cv2.COLOR_BGR2RGB)
    cropped_image_1 = cv2.resize(cropped_image_1, (224,224))
    cropped_image_1 = torch.tensor(cropped_image_1).permute(2,0,1).unsqueeze(0).float()/255.0
    cropped_image_1 = cropped_image_1.to(args.device)

    outputs_1 = teaser_encoder(cropped_image_1)
    
    ###for image2
    kpt_mediapipe = run_mediapipe(image2)

    # crop face if needed

    kpt_mediapipe = kpt_mediapipe[..., :2]

    tform = crop_face(image2,kpt_mediapipe,scale=1.4,image_size=image_size)
    
    cropped_image_2 = warp(image2, tform.inverse, output_shape=(224, 224), preserve_range=True).astype(np.uint8)

    cropped_kpt_mediapipe_2 = np.dot(tform.params, np.hstack([kpt_mediapipe, np.ones([kpt_mediapipe.shape[0],1])]).T).T
    cropped_kpt_mediapipe_2 = cropped_kpt_mediapipe_2[:,:2]

    
    cropped_image_2 = cv2.cvtColor(cropped_image_2, cv2.COLOR_BGR2RGB)
    cropped_image_2 = cv2.resize(cropped_image_2, (224,224))
    cropped_image_2 = torch.tensor(cropped_image_2).permute(2,0,1).unsqueeze(0).float()/255.0
    cropped_image_2 = cropped_image_2.to(args.device)

    outputs_2 = teaser_encoder(cropped_image_2)
    
    #swap, transfer 2's expression to 1
    outputs_2_to_1 = deep_copy_with_tensors(outputs_1)
    #replace 1's expression params with 2's
    outputs_2_to_1['expression_params'] = outputs_2['expression_params']
    outputs_2_to_1['jaw_params'] = outputs_2['jaw_params']
    outputs_2_to_1['eyelid_params'] = outputs_2['eyelid_params']
    
    flame_output_2_to_1 = flame.forward(outputs_2_to_1)
    
    
    renderer_output_2_to_1 =  renderer.forward(flame_output_2_to_1['vertices'], outputs_2_to_1['cam'],
                                                landmarks_fan=flame_output_2_to_1['landmarks_fan'], landmarks_mp=flame_output_2_to_1['landmarks_mp'])
            
    rendered_img_2_to_1 = renderer_output_2_to_1['rendered_img']
    # ---- create the neural renderer reconstructed img ---- #
    
    
    if args.use_teaser_generator:
        if (kpt_mediapipe is None):
            print('Could not find landmarks for the image using mediapipe and cannot create the hull mask for the teaser generator. Exiting...')
            exit()

        mask_ratio_mul = 5
        mask_ratio = 0.01
        mask_dilation_radius = 10

        hull_mask = create_mask(cropped_kpt_mediapipe_1, (224, 224))
        
        rendered_mask = 1 - (rendered_img_2_to_1 == 0).all(dim=1, keepdim=True).float()
        
        hull_mask = torch.from_numpy(hull_mask).type(dtype = torch.float32).unsqueeze(0).to(args.device)
        
        masked_img_2_to_1 = masking_utils.masking_face(cropped_image_1, hull_mask, mask_dilation_radius, rendered_mask=rendered_mask)

        teaser_generator_input = torch.cat([rendered_img_2_to_1, masked_img_2_to_1], dim=1)

        reconstructed_img_2_to_1 = teaser_generator(teaser_generator_input, outputs_1['token'])
        
        
    grid = torch.cat([cropped_image_1, rendered_img_2_to_1, reconstructed_img_2_to_1, cropped_image_2], dim=3)
    

    grid_numpy = grid.squeeze(0).permute(1,2,0).detach().cpu().numpy()*255.0
    grid_numpy = grid_numpy.astype(np.uint8)
    grid_numpy = cv2.cvtColor(grid_numpy, cv2.COLOR_BGR2RGB)
    
    if not os.path.exists(args.out_path):
            os.makedirs(args.out_path)
        
    image_name = "swap_expression_" + args.input_path_target.split('/')[-1].split('.')[0] + '_' + args.input_path_source.split('/')[-1]

    cv2.imwrite(f"{args.out_path}/{image_name}", grid_numpy)

    
