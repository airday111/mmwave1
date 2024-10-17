from MLmodels.resnet import Residual
from MLmodels.resnet import resnet_block
from MLmodels.convLstm import *
import torch.nn as nn
from torch.utils.data import DataLoader
from meldataset import Mat_dataset
from model import *
from doubao import *
from tensorboardX import SummaryWriter
# 定义参数
channels=1
batch=10
model_path='./model_dir3/'
learning_rate=0.005
writer = SummaryWriter(log_dir='logs')
# 定义模型
# model = ConvLSTM(input_dim=channels,
#                  hidden_dim=[4, 8, 16],
#                  kernel_size=(3, 3),
#                  num_layers=3,
#                  batch_first=True,
#                  bias=True,
#                  return_all_layers=False)# 输入(b, t, c, h, w) 批次大小，时间序列，通道数，图像大小
# model_Conv1=nn.Conv2d(in_channels=64, out_channels=128, kernel_size=10, stride=3, padding=0, dilation=1, groups=1, bias=True)
# model_Conv2=nn.Conv2d(in_channels=128, out_channels=256, kernel_size=10, stride=3, padding=0, dilation=1, groups=1, bias=True)
# model_Conv1=nn.Conv2d(in_channels=4, out_channels=8, kernel_size=5, stride=2, padding=0, dilation=1, groups=1, bias=True)
# model_Conv2=nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=2, padding=0, dilation=1, groups=1, bias=True)
# model_Conv3=nn.Conv2d(in_channels=25, out_channels=32, kernel_size=5, stride=2, padding=0, dilation=1, groups=1, bias=True)
# 核心网络结构
# net = nn.Sequential(model,model_Conv1,model_Conv2,model_Conv3,nn.Flatten(), nn.Linear(2880, 6))
# 尝试加入Resnet
# b1 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=7, stride=2, padding=3),
#                    nn.BatchNorm2d(64), nn.ReLU(),
#                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
# b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
# b3 = nn.Sequential(*resnet_block(64, 128, 2))
# b4 = nn.Sequential(*resnet_block(128, 256, 2))
# b5 = nn.Sequential(*resnet_block(256, 512, 2))


# 当前模型
# t1=Transformer(n_trg_vocab=20,n_src_vocab=25)
# t1=nn.Sequential(nn.Conv2d(16, 64, kernel_size=7, stride=2, padding=3),
#                    nn.BatchNorm2d(64), nn.ReLU(),
#                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

# t2=Encoder()
# net = nn.Sequential(model_Conv3,b1,b3,
#                     nn.AdaptiveAvgPool2d((1,1)),
#                     nn.Flatten(), nn.Linear(128, 6))
# net=nn.Sequential(t1)

net=DNet()
net=net.cuda()

# 使用L1 Huber损失函数
loss = torch.nn.SmoothL1Loss(reduce=sum, size_average=False)
trainer = torch.optim.SGD(net.parameters(), lr=learning_rate)

# 导入数据
train_data=Mat_dataset('./')
train_loader = DataLoader(dataset=train_data, batch_size=batch, shuffle=True, num_workers=0, drop_last=False)
test_data=Mat_dataset('./')
test_loader = DataLoader(dataset=test_data, batch_size=batch, shuffle=True, num_workers=0, drop_last=False)

# 训练
def accuracy(y_hat, y):  #准确性检验
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def train_epoch_ch3(net, train_iter, loss, updater,epoch):  #@save
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    running_loss=0.0
    # 训练损失总和、训练准确度总和、样本数
    # metric = Accumulator(3)
    index=0
    # loaded_state_dict = torch.load('./model_dir2/epoch_000220.pth')
    # net.load_state_dict(loaded_state_dict)
    net.eval()  # 设置为评估模式（如果需要）
    for X, y in train_iter:
        index+=1
        if index>1000:
            break

        y_hat = net(X)
        # print(y_hat[0].shape)
        y_hat=y_hat.unsqueeze(0)
        # softmax_func = nn.Softmax(dim=1)
        # y_hat = softmax_func(y_hat) #softmax一下
        # print('soft_output:\n', soft_output)
        l = loss(y_hat, y)

        running_loss+=l.item()
        print(l)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        # metric.add(float(l.mean()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    # print(l.shape)
    epoch_loss = running_loss / len(train_loader)
    return l,epoch_loss

def evaluate_accuracy(net, data_iter):
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式

    # with torch.no_grad():
    #     for X, y in data_iter:
    #         # X = X.reshape(1, 1, 4, 14, 14)
    #         # y = torch.tensor(y).unsqueeze(0)
    #         metric.add(accuracy(net(X,y), y), y.numel())
    # return

def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  #@save

    # animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
    #                     legend=['train loss', 'train acc', 'test acc'])
    # 加载模型，继续训练
    state_dict=torch.load('./model_dir3/epoch_220_000170.pth')
    # print(state_dict.keys())
    net.load_state_dict(state_dict)
    for epoch in range(num_epochs):
        print(epoch)
        train_metrics,epoch_loss = train_epoch_ch3(net, train_iter, loss, updater,epoch)
        writer.add_scalar('loss', epoch_loss, epoch)
        test_acc = evaluate_accuracy(net, test_iter)
        if epoch%10==0:
            torch.save(net.state_dict(), model_path + 'epoch_220_{0:06d}'.format(epoch) + '.pth')
        # print(epoch)
        # animator.add(epoch + 1, train_metrics + (test_acc,))
    # train_loss, train_acc = train_metrics


    # assert train_loss < 0.5, train_loss
    # assert train_acc <= 1 and train_acc > 0.7, train_acc
    # assert test_acc <= 1 and test_acc > 0.7, test_acc

# 训练轮数
num_epochs = 300


train_ch3(net, train_loader, test_loader, loss, num_epochs, trainer)
