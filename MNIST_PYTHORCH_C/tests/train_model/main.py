#!/usr/bin/python3
# coding=utf-8

from ast import arg
import os,argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from   torch.autograd import Variable
from   torch.optim.lr_scheduler import StepLR

np.random.seed(1234)
torch.manual_seed(1234)

import matplotlib.pyplot as plt
import IPython

DATA_PATH='./data/'
EXPORT_MODEL_PATH='./export_model/'
EXPORT_CODE_PATH ='./export_code/'

if not os.path.exists(EXPORT_MODEL_PATH):
    os.makedirs(EXPORT_MODEL_PATH)
if not os.path.exists(EXPORT_CODE_PATH): 
    os.makedirs(EXPORT_CODE_PATH)
########################################
# 演示torch训练MNIST模型并验证算子的原理
########################################


## 加载数据，返回数据加载器
def load_data(batch_size, test_batch_size):
    print('[INF] Generating data loader...')
    
    # 加载numpy数据
    print('[INF] Loading mnist data from npy file...')
    train_x_np=np.load(DATA_PATH+'train_x.npy').astype(np.float32)
    train_y_np=np.load(DATA_PATH+'train_y.npy')
    test_x_np =np.load(DATA_PATH+'test_x.npy').astype(np.float32)
    test_y_np =np.load(DATA_PATH+'test_y.npy')

    # 构建data loader
    train_x = torch.from_numpy(train_x_np)
    train_y = torch.from_numpy(train_y_np).type(torch.LongTensor)
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_x,train_y), 
                                               batch_size=batch_size, shuffle=False)

    test_x = torch.from_numpy(test_x_np)
    test_y = torch.from_numpy(test_y_np).type(torch.LongTensor)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_x,test_y),
                                              batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader


## 网络
class model_c(nn.Module):
    def __init__(self):
        super(model_c, self).__init__()

        # self.conv1 = nn.Conv2d( 1, 32, 5, 1) # CI= 1, CO=32, K=5, S=1, (N, 1,28,28)->(N,32,24,24)
        # self.conv2 = nn.Conv2d(32, 32, 5, 1) # CI=32, CO=32, K=5, S=1, (N,32,24,24)->(N,32,20,20)
        # self.dropout = nn.Dropout2d(0.4)  
        # self.fc1 = nn.Linear(512, 1024)      # 512=32*4*4, (N,512)->(N,1024)
        # self.fc2 = nn.Linear(1024,  10)
        
        self.seq = nn.Sequential(
            nn.Conv2d( 1, 32, 5, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(1),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 10),
        )
        
        self.debug=False

    # 训练和推理
    def forward(self,x):
        # x = self.conv1(x)       # (N, 1,28,28)->(N,32,24,24)
        # x = F.relu(x)
        # x = F.max_pool2d(x, 2)  # (N,32,24,24)->(N,32,12,12)
        # x = self.conv2(x)       # (N,32,12,12)->(N,32, 8, 8)
        # x = F.relu(x)
        # x = F.max_pool2d(x, 2)  # (N,32, 8, 8)->(N,32, 4, 4)
        # x = torch.flatten(x, 1) # (N,32, 4, 4)->(N,512)
        # x = self.fc1(x)         # (N,512)->(N,1024)
        # x = F.relu(x)           
        # x = self.dropout(x)
        # x = self.fc2(x)         # (N,1024)->(N,10)
        
        x = self.seq(x)
        
        return x

    
## 训练
def train(args, model, device, train_loader, test_loader):
    model.to(device)

    print('[INF] Preparing for train...')
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data   = data.resize_(args.batch_size, 1, 28, 28)
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            
            loss = F.cross_entropy(output, target)
            loss.backward()
            
            optimizer.step()
            if batch_idx % 200 == 0:
                print('\rTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()),
                    end='')
        print()
        evaluate(args, model, device, test_loader)
        scheduler.step()


## 评估
def evaluate(args, model, device, test_loader): 
    model.to(device)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data   = data.resize_(args.test_batch_size, 1, 28, 28)
        
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


## 得到运行参数
def get_args():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size'     , type=int  , default=50   , metavar='N' , help='input batch size for training')
    parser.add_argument('--test-batch-size', type=int  , default=5 , metavar='N' , help='input batch size for testing')
    parser.add_argument('--epochs'         , type=int  , default=4    , metavar='N' , help='number of epochs to train')
    parser.add_argument('--lr'             , type=float, default=1.0  , metavar='LR', help='learning rate')
    parser.add_argument('--gamma'          , type=float, default=0.7  , metavar='M' , help='Learning rate step gamma')
    parser.add_argument('--no-cuda'        ,             default=False, action='store_true',  help='disables CUDA training')
    return parser.parse_args()


def export_param_to_c(model,path):
    # 生成c文件
    fname=path+'param.c'
    print('[INF] generating %s...'%fname)
    fp=open(fname,'wt')
    for name,value in model.state_dict().items():
        value=value.numpy().astype(np.float32)  # 网络参数
        name=name.replace('.','_')              # 网络参数名称

        # 输出权重形状数据(整数数组)
        if name[-7:]=='_weight':
            fp.write('const int %s[]={'%(name+'_shape'))
            for n in value.shape:
                fp.write('%d, '%n)
            fp.write('};\n')
        
        # 输出网络参数(浮点数组)
        fp.write('const float %s[]={'%name)
        for n,v in enumerate(value.flatten()):
            if n%10==0: fp.write('\n    ')
            fp.write('(float)%+0.8e, '%v)
        fp.write('\n};\n')
    fp.close()

    # 生成头文件(C程序里的数组申明)
    fname=path+'param.h'
    print('[INF] generating %s...'%fname)
    fp=open(fname,'wt')
    for name in model.state_dict():
        name=name.replace('.','_')
        fp.write('extern const float %s[];\n'%name)
        if name[-7:]=='_weight':
            fp.write('extern const int %s[];\n'%(name+'_shape'))
    fp.close()

def export_test_data_to_c(test_x,test_y,num,path):
    # 生成头文件
    fp=open(path+'test_data.h','wt')
    fp.write('#define TEST_DATA_NUM %d\n'%num)
    fp.write('#define TEST_DATA_SIZE %d\n'%test_x[0].size)
    fp.write('extern const float test_x[TEST_DATA_NUM][TEST_DATA_SIZE];\n')
    fp.write('extern const unsigned char test_y[TEST_DATA_NUM];\n')
    fp.close()
    
    # 生成C文件
    fp=open(path+'test_data.c','wt')
    fp.write('#include "test_data.h"\n')
    # 导出图像数据到C程序
    fp.write('const float test_x[TEST_DATA_NUM][TEST_DATA_SIZE]={\n')
    for n in range(num):
        fp.write('    {')
        for v in test_x[n]: fp.write('(float)%+0.8e, '%v)
        fp.write('},\n')
    fp.write('};\n')
    # 导出参考答案到C程序
    fp.write('const unsigned char test_y[TEST_DATA_NUM]={\n    ')
    for n in range(num):
        fp.write('%u, '%test_y[n])
    fp.write('\n};\n')
    fp.close()


## 主入口
if __name__ == '__main__':
    args = get_args()

    if args.no_cuda:
        device = torch.device('cpu')
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')

    print(f'[INF] using {device}...')
    
    train_loader, test_loader=load_data(args.batch_size,args.test_batch_size)
    
    ## 训练pytorch模型，测试并保存
    if True:
        print('[INF] Constructing model...')
        model = model_c()
        
        print('[INF] Trianing...')
        train(args, model, device, train_loader,test_loader)
        torch.save(model, EXPORT_MODEL_PATH+'mnist_cnn.pth')
        
    ## 加载训练好的pytorch模型并测试
    if True:
        print('[INF] Loading model...')
        model=torch.load(EXPORT_MODEL_PATH+'mnist_cnn.pth')
        
        print('[INF] Testing...')
        evaluate(args, model, device, test_loader)

    ## 导出模型参数到c文件
    if True:
        print('[INF] generating C source file for network parameters...')
        model=torch.load(EXPORT_MODEL_PATH+'mnist_cnn.pth').to(torch.device('cpu'))
        export_param_to_c(model,EXPORT_CODE_PATH)
        
    # 导出测试数据到c文件
    if True:
        print('[INF] generating C source file for test data...')
        test_x=np.array([x.reshape(28,28).flatten() for x in np.load(DATA_PATH+'test_x.npy')]).astype(np.float32)
        test_y=np.load(DATA_PATH+'test_y.npy').flatten().astype(np.uint8)
        export_test_data_to_c(test_x,test_y,    # 测试数据和参考答案
                              500,               # 导出前500个数据
                              EXPORT_CODE_PATH) # 路径

    # 编译模型到C
    USE_IM2COL=False
    import sys
    sys.path.append('../../python')
    from compiler import Compiler
    comp = Compiler(base_dir=EXPORT_CODE_PATH, use_im2col=USE_IM2COL)
    comp.compile(model.seq, in_shape=(1, 1, 28, 28))