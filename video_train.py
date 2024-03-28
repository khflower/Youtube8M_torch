from torchdata.datapipes.iter import FileLister, FileOpener
import torch
from torch.utils.data import DataLoader, Dataset
import torchdata
import torch.nn as nn
import torch.optim as optim

from model import DeepCNN1D6 as DeepCNN1D #모델 임포트
from data_loader import create_dataloader


dataloader = create_dataloader(1024,'train')

# 모델 초기화
device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
model = DeepCNN1D(1, 3862).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

# 모델 저장 위치 설정
model_save_path = "yt8m_ckpt/model7.ckpt"

# 모델 학습
num_epochs = 100
for epoch in range(num_epochs):
    total_loss = 0
    for batch in dataloader:
        # 데이터 및 레이블 가져오기
        mean_audio, mean_rgb, labels = batch['mean_audio'], batch['mean_rgb'], batch['labels']
        mean_audio = mean_audio.to(device)
        mean_rgb = mean_rgb.to(device)
        labels = labels.to(device)
        
        # 순전파 + 역전파 + 최적화
        optimizer.zero_grad()
        outputs = model(mean_audio, mean_rgb)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # 에폭마다 손실 출력
    print(f"Epoch {epoch+1}, Loss: {total_loss}")
    

    # 10 에폭마다 모델 저장
    if ((epoch + 1) % 10 == 0)&(epoch + 1 != 10):
        save_path = f"1dcnn_epoch{epoch+1}.ckpt"
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")





