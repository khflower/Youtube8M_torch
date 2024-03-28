from torchdata.datapipes.iter import FileLister, FileOpener
import torch
from torch.utils.data import DataLoader, Dataset
import torchdata
import torch.nn as nn
import torch.optim as optim
from torch import nn


num_classes = 3862
def collate_fn(batch):
    collated_batch = {
        'mean_audio': torch.stack([data['mean_audio'].clone().detach() for data in batch]),
        'mean_rgb': torch.stack([data['mean_rgb'].clone().detach() for data in batch]),
        'labels': torch.stack([binary_encoding(data['labels'], num_classes) for data in batch])
    }
    return collated_batch


# 이진 인코딩을 위한 함수 정의
def binary_encoding(label, num_classes):
    binary_label = torch.zeros(num_classes)  # 모든 값을 0으로 초기화
    for l in label:
        binary_label[l] = 1  # 해당하는 클래스의 인덱스에 1 설정
    return binary_label




def create_dataloader(batch_size, data_type='train'):
    data_dir = "/data/yt8m/video"
    
    # data_type에 따라 파일 이름 규칙 설정
    if data_type == 'train':
        file_name_pattern = "train????.tfrecord"
    elif data_type == 'validate':
        file_name_pattern = "validate????.tfrecord"
    else:
        raise ValueError("data_type은 'train' 또는 'validate' 이어야 합니다.")
    
    # FileLister와 FileOpener를 사용하여 데이터 준비
    datapipe1 = FileLister(data_dir, file_name_pattern)
    datapipe2 = FileOpener(datapipe1, mode="b")
    dp = datapipe2.load_from_tfrecord()
    
    # 이진 인코딩을 위한 함수 정의
    def binary_encoding(label, num_classes):
        binary_label = torch.zeros(num_classes)  # 모든 값을 0으로 초기화
        for l in label:
            binary_label[l] = 1  # 해당하는 클래스의 인덱스에 1 설정
        return binary_label
    
    # 데이터로더 생성을 위한 collate 함수
    def collate_fn(batch):
        collated_batch = {
            'mean_audio': torch.stack([data['mean_audio'].clone().detach() for data in batch]),
            'mean_rgb': torch.stack([data['mean_rgb'].clone().detach() for data in batch]),
            'labels': torch.stack([binary_encoding(data['labels'], num_classes) for data in batch])
        }
        return collated_batch
    
    # DataLoader 생성
    dataloader = DataLoader(dp, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return dataloader