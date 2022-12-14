import torch.nn.functional as F
import torch
from torch import device, nn

class FC(nn.Module):
    def __init__(self,dropout_fc):
      super().__init__()

      # Batchnorm layers:
      self.batch1  = nn.BatchNorm1d(500)
      self.batch2  = nn.BatchNorm1d(500)
      self.batch3  = nn.BatchNorm1d(500)


      # Fully connected layers:
      self.fc1     = nn.Linear(753, 500)
      self.fc2     = nn.Linear(500, 500)
      self.fc3     = nn.Linear(500, 500)
      self.fc4     = nn.Linear(500,8)

      # Define proportion or neurons to dropout:
      self.dropout = nn.Dropout(dropout_fc)

    
    def forward(self, x):

      # Inputlayer:
      x = x.view(-1,753)

      # Hidden layers:
      x = self.dropout(F.leaky_relu(self.batch1(self.fc1(x))))
      x = self.dropout(F.leaky_relu(self.batch2(self.fc2(x))))
      x = self.dropout(F.leaky_relu(self.batch3(self.fc3(x))))

      # Decision layer:
      x = F.softmax(self.fc4(x),dim = 1)

      return x