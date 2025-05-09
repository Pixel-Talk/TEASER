
import torch
from datasets.lrs3_dataset import get_datasets_LRS3
from datasets.mead_dataset import get_datasets_MEAD
from datasets.mead_sides_dataset import get_datasets_MEAD_sides
from datasets.ffhq_dataset import get_datasets_FFHQ
from datasets.celeba_dataset import get_datasets_CelebA
from datasets.hdtf_dataset import get_datasets_HDTF
from datasets.mixed_dataset_sampler import MixedDatasetBatchSampler
import os
import numpy as np

def load_dataloaders(config):

    # ----------------------- initialize datasets ----------------------- #
    # train_dataset_MEAD, val_dataset_MEAD, test_dataset_MEAD = get_datasets_MEAD(config)

    train_dataset_LRS3, val_dataset_LRS3, test_dataset_LRS3 = get_datasets_LRS3(config)
    train_dataset_ffhq = get_datasets_FFHQ(config)
    train_dataset_celeba = get_datasets_CelebA(config)

    #the percentages of each dataset 
    dataset_percentages = {
        'LRS3': config.dataset.LRS3_percentage,
        'CELEBA': config.dataset.CelebA_percentage,
        'FFHQ': config.dataset.FFHQ_percentage
    }

    
    train_dataset = torch.utils.data.ConcatDataset([
        # train_dataset_MEAD, 
                                                    train_dataset_LRS3, 
                                                    # train_dataset_HDTF,
                                                    train_dataset_celeba,
                                                    train_dataset_ffhq
                                                    ])
    
    sampler = MixedDatasetBatchSampler([
        # len(train_dataset_MEAD),
                                        len(train_dataset_LRS3), 
                                        # len(train_dataset_HDTF),
                                        len(train_dataset_celeba),
                                        len(train_dataset_ffhq)
                                        ], 
                                       list(dataset_percentages.values()), 
                                       config.train.batch_size, config.train.samples_per_epoch)
    
    def collate_fn(batch):
        # filter none
        batch = [b for b in batch if b is not None]
        return torch.utils.data.dataloader.default_collate(batch)
        
    
    # val_dataset = torch.utils.data.ConcatDataset([val_dataset_LRS3, val_dataset_MEAD])
    val_dataset = torch.utils.data.ConcatDataset([val_dataset_LRS3])
                                             
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=sampler, num_workers=config.train.num_workers, collate_fn=collate_fn)
    
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.train.batch_size,
                                                num_workers=config.train.num_workers, shuffle=False, drop_last=True, collate_fn=collate_fn)

    return train_loader, val_loader





def linear_interpolate(landmarks, start_idx, stop_idx):
    """linear_interpolate.

    :param landmarks: ndarray, input landmarks to be interpolated.
    :param start_idx: int, the start index for linear interpolation.
    :param stop_idx: int, the stop for linear interpolation.
    """
    start_landmarks = landmarks[start_idx]
    stop_landmarks = landmarks[stop_idx]
    delta = stop_landmarks - start_landmarks
    for idx in range(1, stop_idx-start_idx):
        landmarks[start_idx+idx] = start_landmarks + idx/float(stop_idx-start_idx) * delta
    return landmarks

def landmarks_interpolate(landmarks):
    """landmarks_interpolate.

    :param landmarks: List, the raw landmark (in-place)

    """
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]  #array--[0,1,...N-1],N as the frame number
    if not valid_frames_idx:
        return None
    for idx in range(1, len(valid_frames_idx)):
        if valid_frames_idx[idx] - valid_frames_idx[idx - 1] == 1:
            continue
        else:
            landmarks = linear_interpolate(landmarks, valid_frames_idx[idx - 1], valid_frames_idx[idx])
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    # -- Corner case: keep frames at the beginning or at the end failed to be detected.
    # if valid_frames_idx:
    #     landmarks[:valid_frames_idx[0]] = [landmarks[valid_frames_idx[0]]] * valid_frames_idx[0]
    #     landmarks[valid_frames_idx[-1]:] = [landmarks[valid_frames_idx[-1]]] * (len(landmarks) - valid_frames_idx[-1])
    # valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    assert len(valid_frames_idx) == len(landmarks), "not every frame has landmark"
    return landmarks




def create_LRS3_lists(lrs3_path, lrs3_mediapipe_path, lrs3_landmarks_path):
    from sklearn.model_selection import train_test_split
    import pickle
    trainval_folder_list = list(os.listdir(f"{lrs3_path}/trainval"))
    train_folder_list, val_folder_list = train_test_split(trainval_folder_list, test_size=0.2, random_state=42)
    test_folder_list = list(os.listdir(f"{lrs3_path}/test"))

    
    def gather_LRS3_split(folder_list, split="trainval"):
        list_ = []
        cont1, cont2, cont3 = 0, 0, 0
        for folder in folder_list:
            for file in os.listdir(os.path.join(f"{lrs3_path}/{split}", folder)):
                cont1 += 1
                if file.endswith(".txt"):
                    cont2 += 1
                    file_without_extension = file.split(".")[0]
                    file_inner_path = f"{split}/{folder}/{file_without_extension}"
                    #replace .txt with .pkl
                    landmarks_filename = os.path.join(lrs3_landmarks_path, file_inner_path+".pkl")

                    valid = True
                    with open(landmarks_filename, "rb") as pkl_file:
                        landmarks = pickle.load(pkl_file)
                        # print('-------------')
                        # print(landmarks.shape)
                        preprocessed_landmarks = landmarks_interpolate(landmarks)
                        if preprocessed_landmarks is None:
                            valid = False

                    mediapipe_landmarks_filepath = os.path.join(lrs3_mediapipe_path, file_inner_path+".npy")
                    if not os.path.exists(mediapipe_landmarks_filepath):
                        valid = False
                    if os.path.exists(landmarks_filename) and valid:
                        cont3 += 1
                        subject = folder

                        #list,each element compose of four part: video, landmark, landmark(.pkl), landmark(mediapipe),folder name;
                        #only when two landmark and video exist, the list can be expanded.
                        list_.append([os.path.join(lrs3_path, file_inner_path + ".mp4"), os.path.join(lrs3_landmarks_path, file_inner_path+".pkl"), 
                                      mediapipe_landmarks_filepath,
                                      subject])
        print(f"cont1={cont1},cont2={cont2},cont3={cont3}")
        return list_

    train_list = gather_LRS3_split(train_folder_list, split="trainval")
    val_list = gather_LRS3_split(val_folder_list, split="trainval")
    test_list = gather_LRS3_split(test_folder_list, split="test")

    print(len(train_list), len(val_list), len(test_list))

    pickle.dump([train_list,val_list,test_list], open(f"assets/LRS3_lists.pkl", "wb"))

def create_HDTF_lists(HDTF_path, HDTF_mediapipe_path, HDTF_fan_path):
    from sklearn.model_selection import train_test_split
    import pickle
    trainval_folder_list = list(os.listdir(HDTF_path))
    train_folder_list, val_folder_list = train_test_split(trainval_folder_list, test_size=0.2, random_state=42)

    
    def gather_HDTF_split(folder_list):
        list_ = []
        cont1, cont2 = 0, 0
        for file in folder_list:
            if file.endswith(".mp4"):
                cont1 += 1
                file_without_extension = file.split(".")[0]
                file_inner_path = f"{file_without_extension}"
                #replace .txt with .pkl
                landmarks_filename = os.path.join(HDTF_fan_path, file_inner_path+".pkl")
                # print('-------------')
                # print(landmarks_filename)
                valid = True
                if not os.path.exists(landmarks_filename):
                    valid = False
                else:
                    with open(landmarks_filename, "rb") as pkl_file:
                        landmarks = pickle.load(pkl_file)
                        # print('-------------')
                        # print(landmarks.shape)
                        preprocessed_landmarks = landmarks_interpolate(landmarks)
                        if preprocessed_landmarks is None:
                            valid = False

                mediapipe_landmarks_filepath = os.path.join(HDTF_mediapipe_path, file_inner_path+".npy")
        
                if not os.path.exists(mediapipe_landmarks_filepath):
                    valid = False
                mediapipe_landmarks = np.load(mediapipe_landmarks_filepath)
                if mediapipe_landmarks is None:
                    valid = False
                if os.path.exists(landmarks_filename) and valid == True:
                    cont2 += 1
                

                    #list,each element compose of four part: video, landmark, landmark(.pkl), landmark(mediapipe),folder name;
                    #only when two landmark and video exist, the list can be expanded.
                    list_.append([os.path.join(HDTF_path, file_inner_path + ".mp4"), landmarks_filename, 
                                    mediapipe_landmarks_filepath])
            continue
        print(f"cont1={cont1},cont2={cont2}")
        return list_

    train_list = gather_HDTF_split(train_folder_list)
    val_list = gather_HDTF_split(val_folder_list)

    print(len(train_list), len(val_list))
    
    pickle.dump([train_list,val_list], open(f"assets/HDTF_lists.pkl", "wb"))