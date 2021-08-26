import torch
from torch.utils.data import DataLoader
from dataset import splitData,Dataset_,train,test
from models import BPNN
from opts import opts
def demo(opt):
    net = BPNN(opt.features)
    # BP模型实例化
    criterion = torch.nn.MSELoss(reduction='mean')
    # 损失函数—均方误差
    optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr)
    # 参数优化方式 Adm
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # CPU及GPU的使用
    print(device)
    net.to(device)
    data_split = splitData([opt.filename],opt.features,opt.ratio)
    # 数据集的划分，按照cfg.Train_Ratio划分为训练及预测部分

    train_set = Dataset_(data_split.train_data)
    test_set = Dataset_(data_split.test_data)
    # 自定义数据集，继承torch Dataset类，通过对__getitem__重构进行调用

    train_iter = DataLoader(train_set, opt.batch, shuffle=True)
    test_iter = DataLoader(test_set, 1, shuffle=True)
    # 数据集的Dataloader,可以理解为将数据集进行打乱，并按照batch进行排列

    train(net,device,train_iter,criterion,optimizer,opt.epochs)
    # 训练
    test(device,test_iter,criterion)
    # 测试
if __name__ == '__main__':
    opt = opts().init()
    demo(opt)
