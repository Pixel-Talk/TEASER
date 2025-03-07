import os
import pickle
import datasets.data_utils as data_utils
from datasets.base_dataset import BaseDataset
import numpy as np
import cv2

class HDTFDataset(BaseDataset):
    def __init__(self, data_list, config, test=False):
        super().__init__(data_list, config, test)
        self.name = 'LRS3'

    def __len__(self):
        return len(self.data_list)

    def __getitem_aux__(self, index):
        sample = self.data_list[index]

        landmarks_filename = sample[1]
        mediapipe_landmarks_filename = sample[2]
        video_path = sample[0]

        if not os.path.exists(landmarks_filename):
            raise Exception('Video %s has no landmarks'%(sample))

        with open(landmarks_filename, "rb") as pkl_file:
            landmarks = pickle.load(pkl_file)
            preprocessed_landmarks = data_utils.landmarks_interpolate(landmarks)
            if preprocessed_landmarks is None:
                raise Exception('Video %s has no landmarks'%(sample))

        mediapipe_landmarks = np.load(mediapipe_landmarks_filename)

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
            raise Exception('Video %s has no frames'%(sample))
        
        landmarks_fan = preprocessed_landmarks[frame_idx]
        landmarks_mediapipe = mediapipe_landmarks[frame_idx]

        data_dict = self.prepare_data(image=image, landmarks_fan=landmarks_fan, landmarks_mediapipe=landmarks_mediapipe)

        return data_dict



def get_datasets_HDTF(config):
    #split the dataset
    if not os.path.exists('assets/HDTF_lists.pkl'):
        print('Creating train, validation lists for hdtf... (This only happens once)')

        from .data_utils import create_HDTF_lists
        create_HDTF_lists(config.dataset.HDTF_path, config.dataset.HDTF_mediapipe_path, config.dataset.HDTF_fan_landmarks_path)


    lists = pickle.load(open("assets/HDTF_lists.pkl", "rb"))
    train_list = lists[0]
    val_list = lists[1]
    # test_list = lists[2]
    #process the data into sutable form
    return HDTFDataset(train_list, config=config), HDTFDataset(val_list, config=config, test=True)

