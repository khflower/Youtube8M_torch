from torchdata.datapipes.iter import FileLister, FileOpener
import torch

from torch.utils.data import DataLoader, Dataset
import torchdata
import torch.nn as nn
import torch.optim as optim

from frame_models import FrameAverageLogisticRegression as FrameAverageLogisticRegression #모델 임포트
from frame_data_loader import create_dataloader
from eval import calculate_hit_at_one, calculate_precision_at_equal_recall_rate_torch, calculate_gap_torch

test_dataloader = create_dataloader(512,'vaildate')

# 모델 초기화
device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
model = FrameAverageLogisticRegression(1152, 3862).to(device)

model.load_state_dict(torch.load("yt8m_ckpt/1dcnn_epoch1.ckpt"))
model.eval()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)



def evaluate(model, dataloader):
    total_loss = 0
    total_correct = 0
    total_samples = 0
    total_positives = 0
    
    with torch.no_grad():
        hit1s= 0
        gap = 0
        perr = 0
        map = 0 
    
        for i, batch in enumerate(dataloader):
            audio, rgb, labels = batch['audio'].to(device), batch['rgb'].to(device), batch['labels']
            labels = labels.to(device)

            outputs = model(audio, rgb)
            pre_predicted = torch.sigmoid(outputs)
            predicted = (pre_predicted > 0.5).float()

            total_loss += criterion(outputs, labels).item()
            total_correct += (predicted.int() & labels.int()).sum().item()  # 수정된 부분
            total_positives += labels.sum().item()
            print("\n 평균 Accuracy : ", total_correct/total_positives)

            hit1s += calculate_hit_at_one(pre_predicted, labels)
            gap += calculate_gap_torch(pre_predicted, labels)
            perr += calculate_precision_at_equal_recall_rate_torch(pre_predicted, labels)
            print("\n 평균 hit1s", hit1s/(i+1))
            print("\n 평균 gap", gap/(i+1))
            print("\n 평균 perr", perr/(i+1))


            total_samples += labels.size(0)

    return total_loss / total_samples, total_correct / total_positives


# 테스트
test_loss, test_acc = evaluate(model, test_dataloader)

print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")