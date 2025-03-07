from tqdm import tqdm
import numpy as np
import os 
import cv2
from ibug.face_detection import RetinaFacePredictor
from ibug.face_alignment import FANPredictor
import argparse
from ibug.face_alignment.utils import plot_landmarks
from multiprocessing import Pool

# Initialize the argument parser
parser = argparse.ArgumentParser(description='Process images/videos with https://github.com/hhj1897/face_alignment.')
parser.add_argument('--input_dir', type=str, required=True, help='Input directory path')
parser.add_argument('--output_dir', type=str, required=True, help='Output directory path')
parser.add_argument('--vis_dir', type=str, help='Directory to save visualizations')
parser.add_argument('--num_processes', type=int, default=10, help='Number of processes to use for processing')
args = parser.parse_args()


all_files = []
for root, _, files in os.walk(args.input_dir):
    for file_name in files:
        if file_name.lower().endswith(('.jpg', '.png', '.mp4', '.avi')):
            all_files.append((root, file_name))


def process_image(root, file_name, output_path, vis_path, face_detector, landmark_detector):
    image = cv2.imread(os.path.join(root, file_name))
    detected_faces = face_detector(image, rgb=False)
    landmarks, scores = landmark_detector(image, detected_faces, rgb=False)
    np.save(output_path, landmarks)
    for lmks, scs in zip(landmarks, scores):
        plot_landmarks(image, lmks, scs, threshold=0.2)
    if vis_path:
        os.makedirs(os.path.dirname(vis_path), exist_ok=True)
        cv2.imwrite(vis_path, image)

def process_video(root, file_name, output_path, vis_path, face_detector, landmark_detector):
    # print('*********')
    # print(os.path.join(root, file_name))
    cap = cv2.VideoCapture(os.path.join(root, file_name))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_landmarks = []
    frame_last = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # print(frame.shape)
        # print('---------')
        # if not ret:
        #     # plot landmarks on the last frame and break loop
        #     os.makedirs(os.path.dirname(vis_path), exist_ok=True)
        #     for lmks, scs in zip(landmarks, scores):
        #         plot_landmarks(frame_last, lmks, scs, threshold=0.2)
        #     cv2.imwrite(os.path.splitext(vis_path)[0]+'.jpg', frame_last)
        #     break
        detected_faces = face_detector(frame, rgb=False)
        landmarks, scores = landmark_detector(frame, detected_faces, rgb=False)
        frame_landmarks.append(landmarks)
        frame_last = frame
    cap.release()
    frame_landmarks = np.concatenate(frame_landmarks, 0)
    np.save(output_path, frame_landmarks)
    

def process_file(root, file_name, face_detector, landmark_detector):
    input_path = os.path.join(root, file_name)
    rel_path = os.path.relpath(input_path, args.input_dir)
    output_path = os.path.join(args.output_dir, os.path.splitext(rel_path)[0] + '.npy')
    vis_path = os.path.join(args.vis_dir, rel_path) if args.vis_dir else None
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if file_name.lower().endswith(('.jpg', '.png')):
        process_image(root, file_name, output_path, vis_path, face_detector, landmark_detector)
    if file_name.lower().endswith(('.mp4', '.avi')):
        process_video(root, file_name, output_path, vis_path, face_detector, landmark_detector)

def process_sample(args):
    # Create a RetinaFace detector using Resnet50 backbone, with the confidence
    # threshold set to 0.8
    face_detector = RetinaFacePredictor(threshold=0.8, device='cuda:0', 
                                        model=RetinaFacePredictor.get_model('mobilenet0.25'))
    # Create a facial landmark detector
    landmark_detector = FANPredictor(device='cuda:0', model=FANPredictor.get_model('2dfan2_alt'))
    root, file_name = args
    process_file(root, file_name, face_detector, landmark_detector)
    
with Pool(args.num_processes, maxtasksperchild=50) as pool:
    list(tqdm(pool.imap(process_sample, all_files), total=len(all_files)))