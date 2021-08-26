import argparse
class opts():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="demo of runup pred")
        # basic experiment setting
        self.parser.add_argument('--features', type=int, default=7,
                             help='The total number of choosen features')
        self.parser.add_argument('--lr', type=float, default=0.01,
                                 help='The value of learning rate')
        self.parser.add_argument('--batch', type=int, default=16,
                                 help='The total number of batch size')
        self.parser.add_argument('--epochs', type=int, default=50,
                                 help='The number of epochs')
        # split dataset
        self.parser.add_argument('--ratio', type=float, default=0.7,
                                 help='ratio of training and test')
        # root
        self.parser.add_argument('--filename', type=str, default='./data.txt')
        self.parser.add_argument('--trainfile', type=str, default='./data_train.txt')
        self.parser.add_argument('--testfile', type=str, default='./data_test.txt')
    def parse(self, args=''):
        if args == '':
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)
        return opt
    def init(self, args=''):
        args = self.parse(args)
        return args
