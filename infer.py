# 推理脚本
import torch
from MLmodels.resnet import Residual
from MLmodels.resnet import resnet_block
from MLmodels.convLstm import *
import torch.nn as nn
from torch.utils.data import DataLoader
from meldataset import Mat_dataset
from model import *
import numpy as np
from doubao import *
# 加载模型，继续训练
state_dict=torch.load('./model_dir3/epoch_220_000030.pth')
print(state_dict.keys())
# t1=Transformer(n_trg_vocab=20,n_src_vocab=25)
t1=DNet()
net=t1
net=net.cuda()
test_data=Mat_dataset('./')
batch=1
test_loader = DataLoader(dataset=test_data, batch_size=batch, shuffle=True, num_workers=0, drop_last=False)
index=0
for X, y in test_loader:
    index+=1
    with torch.no_grad():
        y_hat = net(X)
        y1=y_hat.cpu().numpy()
    np.save('./test2/moxing20_2d_'+str(index)+'.npy',y1)
