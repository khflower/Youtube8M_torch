import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size * 2, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, mean_audio, mean_rgb):
        # 입력 데이터를 하나의 텐서로 결합
        combined_input = torch.cat((mean_audio, mean_rgb), dim=1)
        out = self.fc1(combined_input)
        out = self.relu(out)
        out = self.fc2(out)
        return out
# model = SimpleMLP(576, hidden_size, 3862)

# 로지스틱 회귀 모델 정의
class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, mean_audio, mean_rgb):
        combined_input = torch.cat((mean_audio, mean_rgb), dim=1)
        out = self.linear(combined_input)
        return out
# model = LogisticRegression(1152, 3862).to(device)


class MLP4(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLP4, self).__init__()
        self.fc1 = nn.Linear(input_size, 2048)
        self.bn1 = nn.BatchNorm1d(2048)

        self.fc2 = nn.Linear(2048, 4096)
        self.bn2 = nn.BatchNorm1d(4096)

        self.fc3 = nn.Linear(4096, 8192)
        self.bn3 = nn.BatchNorm1d(8192)

        self.fc4 = nn.Linear(8192, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, mean_audio, mean_rgb):
        combined_input = torch.cat((mean_audio, mean_rgb), dim=1)
        out = self.fc1(combined_input)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc3(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.dropout(out)


        out = self.fc4(out)
        return out
# model = MLP(1152, 3862).to(device)



class MLP6(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLP6, self).__init__()
        self.fc1 = nn.Linear(input_size, 2048)
        self.bn1 = nn.BatchNorm1d(2048)

        self.fc2 = nn.Linear(2048, 4096)
        self.bn2 = nn.BatchNorm1d(4096)

        self.fc3 = nn.Linear(4096, 8192)
        self.bn3 = nn.BatchNorm1d(8192)

        self.fc4 = nn.Linear(8192, 16384)
        self.bn4 = nn.BatchNorm1d(16384)

        self.fc5 = nn.Linear(16384, 8192)
        self.bn5 = nn.BatchNorm1d(8192)

        self.fc6 = nn.Linear(8192, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, mean_audio, mean_rgb):
        combined_input = torch.cat((mean_audio, mean_rgb), dim=1)
        out = self.fc1(combined_input)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc3(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc4(out)
        out = self.bn4(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc5(out)
        out = self.bn5(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc6(out)
        return out
# model = MLP(1152, 3862).to(device)


class MLP5(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLP5, self).__init__()
        self.fc1 = nn.Linear(input_size, 2048)
        self.bn1 = nn.BatchNorm1d(2048)

        self.fc2 = nn.Linear(2048, 4096)
        self.bn2 = nn.BatchNorm1d(4096)

        self.fc3 = nn.Linear(4096, 8192)
        self.bn3 = nn.BatchNorm1d(8192)

        self.fc4 = nn.Linear(8192, 16384)
        self.bn4 = nn.BatchNorm1d(16384)

        self.fc5 = nn.Linear(16384, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, mean_audio, mean_rgb):
        combined_input = torch.cat((mean_audio, mean_rgb), dim=1)
        out = self.fc1(combined_input)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc3(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc4(out)
        out = self.bn4(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc5(out)
        return out
# model = MLP(1152, 3862).to(device)
    
class DeepCNN1D6(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(DeepCNN1D6, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(p=0.2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(p=0.2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(p=0.2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(p=0.2)
        )
        self.layer5 = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(p=0.2)
        )
        self.layer6 = nn.Sequential(
            nn.Conv1d(256, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(p=0.2)
        )

        self.fc1 = nn.Linear(int(1024*(1152/64)), 16384)  # 288는 (1152/2/2)의 결과입니다.
        self.fc2 = nn.Linear(16384, 8192)
        self.fc3 = nn.Linear(8192, num_classes)
        self.dropout = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm1d(16384)
        self.bn2 = nn.BatchNorm1d(8192)
        self.relu = nn.ReLU()

        
    def forward(self, mean_audio, mean_rgb):
        combined_input = torch.cat((mean_audio, mean_rgb), dim=1)
        combined_input = combined_input.unsqueeze(1)

        out = self.layer1(combined_input)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = out.view(out.size(0), -1)  # Flatten the output
        out = self.fc1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc3(out)

        return out
# model = DeepCNN1D(1, 3862).to(device)