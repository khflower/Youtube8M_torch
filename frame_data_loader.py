from torchdata.datapipes.iter import FileLister, FileOpener
import torch
from torch.utils.data import DataLoader, Dataset
import torchdata
import torch.nn as nn
import torch.optim as optim
from torch import nn


num_classes = 3862


def create_dataloader(batch_size, data_type='train'):
    data_dir = "/data/yt8m/frame"
    
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
        MAX_LEN = 300
        rgb_stack = []
        audio_stack = []
        labels = []

        for data in batch:
            rgb_data = torch.Tensor(data["rgb"]).reshape(-1, 1024)
            audio_data = torch.Tensor(data["audio"]).reshape(-1, 128)

            # 데이터 길이가 MAX_LEN보다 작으면 0으로 패딩
            if len(rgb_data) < MAX_LEN:
                rgb_data = torch.cat([rgb_data, torch.zeros(MAX_LEN - len(rgb_data), 1024)], dim=0)
            else:
                rgb_data = rgb_data[:MAX_LEN]

            if len(audio_data) < MAX_LEN:
                audio_data = torch.cat([audio_data, torch.zeros(MAX_LEN - len(audio_data), 128)], dim=0)
            else:
                audio_data = audio_data[:MAX_LEN]

            rgb_stack.append(rgb_data)
            audio_stack.append(audio_data)
            labels.append(binary_encoding(data["labels"], 3862).clone().detach())

        return {
            "rgb": torch.stack(rgb_stack),
            "audio": torch.stack(audio_stack),
            "labels": torch.stack(labels)
        }
    
    # DataLoader 생성
    dataloader = DataLoader(dp, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return dataloader

