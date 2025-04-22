import torch
import cv2
import numpy as np
from skimage.transform import estimate_transform, warp
from src.teaser_encoder import TeaserEncoder
from src.FLAME.FLAME import FLAME
from src.renderer.renderer import Renderer
import argparse
import os
import pickle
import imageio
from tqdm.auto import tqdm
import src.utils.masking as masking_utils
from utils.mediapipe_utils import run_mediapipe
from datasets.base_dataset import create_mask
import torch.nn.functional as F


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_path', type=str, default='', help='Path to the input image/video')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on')
    parser.add_argument('--checkpoint', type=str, default='', help='Path to the checkpoint')
    parser.add_argument('--crop', action='store_true', help='Crop the face using mediapipe')
    parser.add_argument('--out_path', type=str, default='', help='Path to save the output (will be created if not exists)')
    parser.add_argument('--use_teaser_generator', action='store_true', help='Use TEASER neural image to image translator to reconstruct the image')
    parser.add_argument('--render_orig', action='store_true', help='Present the result w.r.t. the original image/video size')

    args = parser.parse_args()

    input_image_size = 224
    

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

        # load also triangle probabilities for sampling points on the image
        face_probabilities = masking_utils.load_probabilities_per_FLAME_triangle()  


    # ---- visualize the results ---- #

    flame = FLAME().to(args.device)
    renderer = Renderer().to(args.device)

    for video_items in os.listdir(args.input_path):
            video_path = os.path.join(args.input_path, video_items)
            # print(video_path)
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                print('Error opening video file')
                exit()

            video_fps = cap.get(cv2.CAP_PROP_FPS)
            video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # calculate size of output video
            if args.render_orig:
                out_width = video_width
                out_height = video_height
            else:
                out_width = input_image_size
                out_height = input_image_size

            if args.use_teaser_generator:
                out_width *= 3
            else:
                out_width *= 2

            if not os.path.exists(args.out_path):
                os.makedirs(args.out_path)
            out_dirs = args.out_path
            if not os.path.exists(out_dirs):
                os.makedirs(out_dirs)
            
            f_out = f"{out_dirs}/{video_path.split('/')[-1].split('.')[0]}.mp4"
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            out_frames = []
            out_coeffs = []
            t_ind = 0
            for _ in range(n_frames):
                ret, image = cap.read()
                if t_ind > 300: break
                t_ind += 1

                if not ret:
                    break
                
                kpt_mediapipe = run_mediapipe(image)

                # crop face if needed
                if args.crop:
                    if (kpt_mediapipe is None):
                        print('Could not find landmarks for the image using mediapipe and cannot crop the face. Exiting...')
                        continue
                    
                    kpt_mediapipe = kpt_mediapipe[..., :2]

                    tform = crop_face(image,kpt_mediapipe,scale=1.4,image_size=input_image_size)
                    
                    cropped_image = warp(image, tform.inverse, output_shape=(224, 224), preserve_range=True).astype(np.uint8)

                    cropped_kpt_mediapipe = np.dot(tform.params, np.hstack([kpt_mediapipe, np.ones([kpt_mediapipe.shape[0],1])]).T).T
                    cropped_kpt_mediapipe = cropped_kpt_mediapipe[:,:2]
                else:
                    cropped_image = image
                    cropped_kpt_mediapipe = kpt_mediapipe

                
                cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
                cropped_image = cv2.resize(cropped_image, (224,224))
                cropped_image = torch.tensor(cropped_image).permute(2,0,1).unsqueeze(0).float()/255.0
                cropped_image = cropped_image.to(args.device)

                outputs = teaser_encoder(cropped_image)
                
                out_coeffs.append({k: v.cpu().detach().numpy() for k, v in outputs.items()})
                flame_output = flame.forward(outputs)
                renderer_output = renderer.forward(flame_output['vertices'], outputs['cam'],
                                                    landmarks_fan=flame_output['landmarks_fan'], landmarks_mp=flame_output['landmarks_mp'])
                
                rendered_img = renderer_output['rendered_img']

                if args.render_orig:
                    if args.crop:
                        rendered_img_numpy = (rendered_img.squeeze(0).permute(1,2,0).detach().cpu().numpy()*255.0).astype(np.uint8)               
                        rendered_img_orig = warp(rendered_img_numpy, tform, output_shape=(video_height, video_width), preserve_range=True).astype(np.uint8)
                        # back to pytorch to concatenate with full_image
                        rendered_img_orig = torch.Tensor(rendered_img_orig).permute(2,0,1).unsqueeze(0).float()/255.0
                    else:
                        rendered_img_orig = F.interpolate(rendered_img, (video_height, video_width), mode='bilinear').cpu()

                    full_image = torch.Tensor(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).permute(2,0,1).unsqueeze(0).float()/255.0
                    grid = torch.cat([full_image, rendered_img_orig], dim=3)
                else:
                    grid = torch.cat([cropped_image, rendered_img], dim=3)
                
          

                # ---- create the neural renderer reconstructed img ---- #
                if args.use_teaser_generator:
                    if (kpt_mediapipe is None):
                        print('Could not find landmarks for the image using mediapipe and cannot create the hull mask for the teaser generator. Exiting...')
                        exit()

                    mask_ratio_mul = 5
                    mask_ratio = 0.01
                    mask_dilation_radius = 10

                    hull_mask = create_mask(cropped_kpt_mediapipe, (224, 224))

                    rendered_mask = 1 - (rendered_img == 0).all(dim=1, keepdim=True).float()
                    
                    hull_mask = torch.from_numpy(hull_mask).type(dtype = torch.float32).unsqueeze(0).to(args.device)

                    # extra_points = cropped_image * pmask
                    masked_img = masking_utils.masking_face(cropped_image, hull_mask, mask_dilation_radius, rendered_mask=rendered_mask)

                    teaser_generator_input = torch.cat([rendered_img, masked_img], dim=1)

                    
                    reconstructed_img = teaser_generator(teaser_generator_input, outputs['token'])

                    if args.render_orig:
                        if args.crop:
                            reconstructed_img_numpy = (reconstructed_img.squeeze(0).permute(1,2,0).detach().cpu().numpy()*255.0).astype(np.uint8)               
                            reconstructed_img_orig = warp(reconstructed_img_numpy, tform, output_shape=(video_height, video_width), preserve_range=True).astype(np.uint8)
                            # back to pytorch to concatenate with full_image
                            reconstructed_img_orig = torch.Tensor(reconstructed_img_orig).permute(2,0,1).unsqueeze(0).float()/255.0
                        else:
                            reconstructed_img_orig = F.interpolate(reconstructed_img, (video_height, video_width), mode='bilinear').cpu()

                        grid = torch.cat([grid, reconstructed_img_orig], dim=3)
                    else:
                        grid = torch.cat([grid, reconstructed_img], dim=3)
                        # grid = reconstructed_img

                grid_numpy = grid.squeeze(0).permute(1,2,0).detach().cpu().numpy()*255.0
                grid_numpy = grid_numpy.astype(np.uint8)
                grid_numpy = cv2.cvtColor(grid_numpy, cv2.COLOR_BGR2RGB)
                out_frames.append(grid_numpy[..., ::-1])

            cap.release()
            imageio.mimsave(f_out, out_frames)
    




