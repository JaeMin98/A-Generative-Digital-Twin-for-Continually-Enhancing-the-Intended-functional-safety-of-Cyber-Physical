import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        """
        LSTM 모델 초기화
        :param input_size: 입력 feature의 수 (N개의 feature)
        :param hidden_size: LSTM 내부에서 사용할 hidden state의 크기
        :param num_layers: LSTM 층의 수
        """
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 10)  # 최종 출력을 10개 feature로 압축

    def forward(self, x):
        """
        모델의 순전파 로직
        :param x: 모델 입력 데이터, 크기는 (batch_size, sequence_length, input_size)
        :return: 최종 출력 데이터, 크기는 (batch_size, 10)
        """
        # LSTM을 통해 각 sequence의 마지막 hidden state를 얻는다
        output, (hn, cn) = self.lstm(x)
        
        # 마지막 시간 단계의 hidden state를 사용하여 최종 출력 계산
        out = self.fc(hn[-1])
        return out

# 예제 사용
# input_size: N, hidden_size: 임의로 20, num_layers: 1
model = LSTMModel(input_size=50, hidden_size=20, num_layers=1)
# 예시 데이터 (batch_size=3, sequence_length=5, input_size=N)
x_example = torch.randn(3, 5, 50)
output = model(x_example)

print(output)  # 최종 출력, 크기는 (3, 10)
