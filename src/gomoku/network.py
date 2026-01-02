"""
오목 AI 신경망
정책망(Policy Network) + 가치망(Value Network)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """ResNet 스타일 잔차 블록"""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class GomokuNet(nn.Module):
    """
    오목 AI 신경망
    입력: (batch, 3, 15, 15) - 3채널 보드 상태
    출력: 
      - policy: (batch, 225) - 각 위치의 착수 확률
      - value: (batch, 1) - 승률 예측 (-1 ~ 1)
    """
    def __init__(self, board_size=15, num_channels=128, num_res_blocks=5):
        super(GomokuNet, self).__init__()
        self.board_size = board_size
        
        # 초기 합성곱 층
        self.conv_input = nn.Conv2d(3, num_channels, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(num_channels)
        
        # 잔차 블록들
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(num_res_blocks)
        ])
        
        # 정책 헤드 (Policy Head)
        self.policy_conv = nn.Conv2d(num_channels, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, board_size * board_size)
        
        # 가치 헤드 (Value Head)
        self.value_conv = nn.Conv2d(num_channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, 256)
        self.value_fc2 = nn.Linear(256, 1)
        
    def forward(self, x):
        # 입력 처리
        x = F.relu(self.bn_input(self.conv_input(x)))
        
        # 잔차 블록 통과
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # 정책 헤드
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(-1, 2 * self.board_size * self.board_size)
        policy = self.policy_fc(policy)
        policy = F.log_softmax(policy, dim=1)
        
        # 가치 헤드
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(-1, self.board_size * self.board_size)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value
    
    def predict(self, board_state):
        """단일 보드 상태에 대한 예측"""
        self.eval()
        with torch.no_grad():
            if isinstance(board_state, torch.Tensor):
                x = board_state
            else:
                x = torch.FloatTensor(board_state)
            
            if len(x.shape) == 3:
                x = x.unsqueeze(0)
            
            policy, value = self.forward(x)
            policy = torch.exp(policy)  # log_softmax를 다시 확률로
            
            return policy.squeeze().cpu().numpy(), value.squeeze().cpu().item()

class SimplePolicyNet(nn.Module):
    """간단한 정책망 (빠른 학습용)"""
    def __init__(self, board_size=15):
        super(SimplePolicyNet, self).__init__()
        self.board_size = board_size
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        
        self.fc = nn.Linear(64 * board_size * board_size, board_size * board_size)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 64 * self.board_size * self.board_size)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)
    
    def predict(self, board_state):
        """단일 보드 상태에 대한 예측"""
        self.eval()
        with torch.no_grad():
            if isinstance(board_state, torch.Tensor):
                x = board_state
            else:
                x = torch.FloatTensor(board_state)
            
            if len(x.shape) == 3:
                x = x.unsqueeze(0)
            
            policy = self.forward(x)
            policy = torch.exp(policy)
            
            return policy.squeeze().cpu().numpy()
