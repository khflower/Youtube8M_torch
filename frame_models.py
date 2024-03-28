import torch
import torch.nn as nn





class FrameAverageLogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(FrameAverageLogisticRegression, self).__init__()
        # input_size는 audio와 rgb를 평균낸 후 연결한 벡터의 크기입니다.
        # 예를 들어, audio (128)와 rgb (1024)를 평균 내면 각각 128, 1024 크기의 벡터가 되고, 이를 연결하면 1152가 됩니다.
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, audio, rgb):
        # audio와 rgb의 차원을 확인합니다: 예상되는 차원은 [배치사이즈, 300, 128] 및 [배치사이즈, 300, 1024]
        # 먼저, 프레임 차원(axis=1)에 대해 평균을 계산합니다.
        audio_mean = audio.mean(dim=1)  # [배치사이즈, 128]
        rgb_mean = rgb.mean(dim=1)  # [배치사이즈, 1024]
        # 이제 두 벡터를 연결합니다.
        x = torch.cat((audio_mean, rgb_mean), dim=1)  # [배치사이즈, 1152]
        # 연결된 벡터를 선형 레이어에 전달합니다.
        out = self.linear(x)
        return out