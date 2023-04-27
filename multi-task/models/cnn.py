import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import CrossEntropyLoss

thresh = 0.5
lens = 0.5
device = torch.device("cuda:0")

class ActFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input-thresh) < lens
        return grad_input * temp.float()

act_fun = ActFun.apply

# cnn_layer(in_planes, out_planes, stride, padding, kernel_size)
cfg_cnn = [(1, 10, 1, 1, 3),
           (10, 10, 1, 1, 3),]
# kernel size
cfg_kernel = [36, 18, 9]
# fc layer
cfg_fc = [512, 50]

class CNN_Model(nn.Module):
    
    def __init__(self, n_tasks):
        super(CNN_Model, self).__init__()
        self.n_tasks = n_tasks
        for i in range(self.n_tasks):
            setattr(self, 'task_{}'.format(i), nn.Linear(50, 10))
        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[0]
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding).to(device)
        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[1]
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding).to(device)
        self.fc = nn.Linear(cfg_kernel[-1] * cfg_kernel[-1] * cfg_cnn[-1][1], cfg_fc[1]).to(device)
    def forward(self, input, win=15):
        batch_size = input.size(0)
        x = self.conv1(input)
        x = nn.ReLU()(x)
        x = act_fun(x)
        x = F.avg_pool2d(x,2)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = act_fun(x)
        x = F.avg_pool2d(x,2)
        x = x.view(batch_size,-1)
        outs = self.fc(x)
        output=[]
        for i in range(self.n_tasks):
            layer = getattr(self, 'task_{}'.format(i))
            output.append(layer(outs))
        return torch.stack(output, dim=1)