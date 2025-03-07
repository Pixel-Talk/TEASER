import os
import pickle
from datasets.base_dataset import BaseDataset
import numpy as np
import cv2

import os
import glob

class MEADSidesDataset(BaseDataset):
    def __init__(self, data_list, config, test=False):
        super().__init__(data_list, config, test)
        self.name = 'MEAD_SIDES'

    def __len__(self):
        return len(self.data_list)

    def __getitem_aux__(self, index):
        sample = self.data_list[index]

        landmarks_filename = sample[1]
        video_path = sample[0]

        if not os.path.exists(landmarks_filename):
            print('Mediapipe landmarks not found for %s'%(sample))
            return None

        landmarks_mediapipe = np.load(landmarks_filename)

        video = cv2.VideoCapture(video_path)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        # select randomly one file from this subject
        if num_frames == 0:
            print('Video %s has no frames'%(sample))
            return None
        
        # pick random frame
        frame_idx = np.random.randint(0, num_frames)
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, image = video.read()
        if not ret:
            print('Video %s has no frames'%(sample))
            return None

        landmarks_mediapipe = landmarks_mediapipe[frame_idx]

        data_dict = self.prepare_data(image=image, landmarks_fan=None, landmarks_mediapipe=landmarks_mediapipe)

        return data_dict
    


def get_datasets_MEAD_sides(config=None):
    
    files = [f for f in os.listdir(config.dataset.MEAD_fan_landmarks_path)]

    # this is the split used in the paper, randomly selected
    train_subjects = ['M003', 'M007', 'M009', 'M011', 'M012', 'M019', 'M024', 'M025', 'M026', 'M027', 'M029', 'M030', 'M031', 'M032', 'M033', 'M034', 'M035', 'M037', 'M039', 'M040', 'M041', 'W009', 'W011', 'W014', 'W015', 'W016', 'W019', 'W021', 'W023', 'W024', 'W025', 'W026', 'W035', 'W036', 'W037', 'W038', 'W040']
    val_subjects = ['M013', 'M023', 'M042', 'W018', 'W028']
    test_subjects = ['M005', 'M022', 'M028', 'W029', 'W033']

    # assert each subject is in exactly one split
    assert len(set(train_subjects).intersection(val_subjects)) == 0
    assert len(set(train_subjects).intersection(test_subjects)) == 0
    assert len(set(val_subjects).intersection(test_subjects)) == 0

    if not os.path.exists("assets/MEAD_lists.pkl"):
        print('Creating train, validation, and test lists for MEAD Sides... (This only happens once)')

        train_list = []
        val_list = []
        test_list = []
        for file in files:
            if file in train_subjects:
                base_path_video = os.path.join(config.dataset.MEAD_sides_path, file)
                base_path_fan_landmark = os.path.join(config.dataset.MEAD_fan_landmarks_path, file)
                base_path_mediapipe_landmark = os.path.join(config.dataset.MEAD_sides_mediapipe_landmarks_path, file)
                for root, dirs, files in os.walk(base_path_video):
                    if root.count(os.sep) - base_path_video.count(os.sep) == 0:
                        for dir in dirs:
                            for view in ['videos_left_30', 'videos_left_60', 'videos_right_30', 'videos_right_60']:
                            #只要front目录下的
                                view_path = os.path.join(root, dir, view)
                                if os.path.exists(view_path) and os.path.isdir(view_path):
                                    for subdir, _, _ in os.walk(view_path):
                                        for path in glob.glob(os.path.join(subdir, '*.mp4')):
                                            video_path = path
                                            fan_landmarks_path = path.replace(base_path_video,base_path_fan_landmark).replace('.mp4','.pkl')
                                            mediapipe_landmarks_path = path.replace(base_path_video,base_path_mediapipe_landmark).replace('.mp4','.npy')
                                            train_list.append([video_path, mediapipe_landmarks_path, fan_landmarks_path,  file])
            for file in files:
                if file in val_subjects:
                    base_path_video = os.path.join(config.dataset.MEAD_path, file)
                    base_path_fan_landmark = os.path.join(config.dataset.MEAD_fan_landmarks_path, file)
                    base_path_mediapipe_landmark = os.path.join(config.dataset.MEAD_mediapipe_landmarks_path, file)
                    for root, dirs, files in os.walk(base_path_video):
                        if root.count(os.sep) - base_path_video.count(os.sep) == 0:
                            for dir in dirs:
                                for view in ['videos_left_30', 'videos_left_60', 'videos_right_30', 'videos_right_60']:
                                #只要front目录下的
                                    view_path = os.path.join(root, dir, view)
                                    if os.path.exists(view_path) and os.path.isdir(view_path):
                                        for subdir, _, _ in os.walk(view_path):
                                            for path in glob.glob(os.path.join(subdir, '*.mp4')):
                                                video_path = path
                                                fan_landmarks_path = path.replace(base_path_video,base_path_fan_landmark).replace('.mp4','.pkl')
                                                mediapipe_landmarks_path = path.replace(base_path_video,base_path_mediapipe_landmark).replace('.mp4','.npy')
                                                val_list.append([video_path, mediapipe_landmarks_path, fan_landmarks_path, file])
            for file in files:
                if file in test_subjects:
                    base_path_video = os.path.join(config.dataset.MEAD_path, file)
                    base_path_fan_landmark = os.path.join(config.dataset.MEAD_fan_landmarks_path, file)
                    base_path_mediapipe_landmark = os.path.join(config.dataset.MEAD_mediapipe_landmarks_path, file)
                    for root, dirs, files in os.walk(base_path_video):
                        if root.count(os.sep) - base_path_video.count(os.sep) == 0:
                            for dir in dirs:
                                #只要front目录下的
                                view_path = os.path.join(root, dir, view)
                                if os.path.exists(view_path) and os.path.isdir(view_path):
                                    for subdir, _, _ in os.walk(view_path):
                                        for path in glob.glob(os.path.join(subdir, '*.mp4')):
                                            video_path = path
                                            fan_landmarks_path = path.replace(base_path_video,base_path_fan_landmark).replace('.mp4','.pkl')
                                            mediapipe_landmarks_path = path.replace(base_path_video,base_path_mediapipe_landmark).replace('.mp4','.npy')
                                            test_list.append([video_path, mediapipe_landmarks_path, fan_landmarks_path, file])
    
      
        pickle.dump([train_list,val_list,test_list], open(f"assets/MEAD_lists.pkl", "wb"))
    else:
        train_list, val_list, test_list = pickle.load(open("assets/MEAD_lists.pkl", "rb"))

    # print("MEAD Sides Train: ", len(train_list))
    # print("MEAD Sides Val: ", len(val_list))
    # print("MEAD Sides Test: ", len(test_list))

    return MEADSidesDataset(train_list, config), MEADSidesDataset(val_list, config, test=True), MEADSidesDataset(test_list, config, test=True)





