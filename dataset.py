import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import random
class Dataset_(Dataset):
    def __init__(self, data):
        data = torch.Tensor(data)
        self.x_data = Variable(data[:,1:])
        self.y_data = Variable(data[:,0].unsqueeze(1))

    def __getitem__(self, index):
        x = self.x_data[index]
        y = self.y_data[index]
        sample = {'img': x, 'label': y}
        return sample
    def __len__(self):
        return len(self.y_data)

class splitData():
    def __init__(self, file_path=[],feature=7,Train_Ratio=0.9):
        print('LoadDataset')
        if len(file_path) != 1:
            raise ValueError
        self.img_path = file_path[0]
        data = self.my_normalize(self.img_path)
        self.pick_features(data,feature+1,Train_Ratio)


    def read_file(self, filename):
        data = []
        file = open(filename, 'r')
        file_data = file.readlines()
        for row in file_data:
            tmp_list = row.split('\t')
            tmp_list[-1] = tmp_list[-1].replace('\n', '')
            tmp_list = list(map(float, tmp_list))
            data.append(tmp_list)
        return data

    def my_normalize(self, filename):
        data = np.array(self.read_file(filename))
        max_y = np.max(data[:,0])
        min_y = np.min(data[:,0])
        data[:,0] = (data[:,0] - min_y) / (max_y-min_y)
        for col in range(1,np.size(data[1])):
            x_data = data[:, col]
            mean_ = np.mean(x_data)
            std_ = np.std(x_data)+0.0001
            data[:,col] = (data[:,col] - mean_) / std_
        random.shuffle(data)
        return data

    def pick_features(self, data_my, num, Train_Ratio):
        size_ = data_my.shape[0]
        len_ = int(size_*Train_Ratio)
        len2 = size_ - len_
        self.train_data = np.zeros((len_, num))
        self.test_data = np.zeros((len2,num))
        for i in range(size_):
            if i < len_:
                self.train_data[i] = data_my[i]
            else:
                self.test_data[i-len_] = data_my[i]

def train(net,device,dataloader,criterion,optimizer,epochs):
    net.train()
    train_iter = dataloader
    loss_list = []
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)
    for epoch in range(epochs):
        loss_epoch = 0
        for i,data in enumerate(train_iter):
            x = Variable(data['img'].to(device))
            y = Variable(data['label'].to(device))
            output = net(x)
            loss = criterion(output,y)
            loss_epoch += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        loss_list.append(loss_epoch)
        print('Epoch [%i/%i]'%(epoch+1,epochs))
        print('Loss: %6.4f  LR %8.5f'%(loss_epoch.item(),optimizer.param_groups[0]['lr']))
    torch.save(net, './BPNN_volume.pth')
    plt.plot(range(epochs),loss_list)
    plt.show()

def test(device,dataloader,criterion):
    net = torch.load('./BPNN_volume.pth')
    label = []
    predict = []
    for i,data in enumerate(dataloader):
        x = Variable(data['img'].to(device))
        y = Variable(data['label'].to(device))
        output=net(x)
        loss = criterion(output, y)
        label.append(y.cpu().numpy().item())
        predict.append(output.detach().cpu().numpy().item())
    print('the length of predict',len(predict))
    plt.scatter(label,predict)
    plt.plot([0,1], [0,1],'-b')
    #plt.xlim(0, 1)
    #plt.ylim(0, 1)
    plt.xlabel('Measurement')
    plt.ylabel('Predict')
    plt.show()

if __name__ == '__main__':
    net = torch.load(cfg.root + 'BPNN_volume.pth')
