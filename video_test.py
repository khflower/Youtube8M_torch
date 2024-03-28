from torchdata.datapipes.iter import FileLister, FileOpener
import torch

from torch.utils.data import DataLoader, Dataset
import torchdata
import torch.nn as nn
import torch.optim as optim

from video_models import DeepCNN1D6 as DeepCNN1D #모델 임포트
from video_data_loader import create_dataloader
from eval import calculate_hit_at_one, calculate_precision_at_equal_recall_rate_torch, calculate_gap_torch, calculate_map_torch

test_dataloader = create_dataloader(256,'validate')

# 모델 초기화
device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
model = DeepCNN1D(1, 3862).to(device)

model.load_state_dict(torch.load("yt8m_ckpt/1dcnn_epoch30.ckpt"))
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
            mean_audio, mean_rgb, labels = batch['mean_audio'], batch['mean_rgb'], batch['labels'].float()
            mean_audio = mean_audio.to(device)
            mean_rgb = mean_rgb.to(device)
            labels = labels.to(device)

            outputs = model(mean_audio, mean_rgb)
            pre_predicted = torch.sigmoid(outputs)
            predicted = (pre_predicted > 0.5).float()

            total_loss += criterion(outputs, labels).item()
            total_correct += (predicted.int() & labels.int()).sum().item()  # 수정된 부분
            total_positives += labels.sum().item()
            print("\n 평균 Accuracy : ", total_correct/total_positives)

            hit1s += calculate_hit_at_one(pre_predicted, labels)
            gap += calculate_gap_torch(pre_predicted, labels)
            perr += calculate_precision_at_equal_recall_rate_torch(pre_predicted, labels)
            map += calculate_map_torch(pre_predicted, labels)
            print("\n 평균 hit1s", hit1s/(i+1))
            print("\n 평균 gap", gap/(i+1))
            print("\n 평균 perr", perr/(i+1))
            print("\n 평균 map", map/(i+1))


            total_samples += labels.size(0)

    return total_loss / total_samples, total_correct / total_positives


# 테스트
test_loss, test_acc = evaluate(model, test_dataloader)

print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")