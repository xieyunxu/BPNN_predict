import torch
class BPNN(torch.nn.Module):
    def __init__(self,num):
        super(BPNN, self).__init__()
        self.linear1 = torch.nn.Linear(num, 64)
        self.linear2 = torch.nn.Linear(64, 128)
        self.linear3 = torch.nn.Linear(128, 128)
        self.linear4 = torch.nn.Linear(128, 64)
        self.linear5 = torch.nn.Linear(64, 1)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x1):
        x1 = self.sigmoid(self.linear1(x1))
        x1 = self.sigmoid(self.linear2(x1))
        x1 = self.sigmoid(self.linear3(x1))
        x1 = self.sigmoid(self.linear4(x1))
        x1 = self.sigmoid(self.linear5(x1))
        return x1