from torchdata.datapipes.iter import FileLister, FileOpener
import torch
from torch.utils.data import DataLoader, Dataset
import torchdata
import torch.nn as nn
import torch.optim as optim

from frame_models import FrameAverageLogisticRegression as FrameAverageLogisticRegression #모델 임포트
from frame_data_loader import create_dataloader
from eval import calculate_hit_at_one

dataloader = create_dataloader(512,'train')

# 모델 초기화
device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
model = FrameAverageLogisticRegression(1152, 3862).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

# 모델 저장 위치 설정
model_save_path = "yt8m_ckpt/frame_model.ckpt"

# 모델 학습
num_epochs = 30
save_batch_interval = 100 # 100개 배치마다 저장
for epoch in range(num_epochs):
    total_loss = 0
    for i, batch in enumerate(dataloader):
        # 데이터 및 레이블 가져오기
        audio, rgb, labels = batch['audio'].to(device), batch['rgb'].to(device), batch['labels']
        labels = labels.to(device)
        
        # 순전파 + 역전파 + 최적화
        optimizer.zero_grad()
        outputs = model(audio, rgb)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        print("loss : ", loss.item())
        print("hit1 : ", calculate_hit_at_one(torch.sigmoid(outputs), labels))

        # 일정 배치마다 모델 저장
        if (i + 1) % save_batch_interval == 0:
            save_path = f"1dcnn_epoch{epoch+1}_batch{i+1}.ckpt"
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")



    # 에폭마다 손실 출력
    print(f"Epoch {epoch+1}, Loss: {total_loss}")
    

    # 1 에폭마다 모델 저장
    if ((epoch + 1) % 1 == 0)&(epoch + 1 != 10):
        save_path = f"1dcnn_epoch{epoch+1}.ckpt"
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")





