import torch
import numpy as np
import torch.nn as nn

device = torch.device("cuda:0")
thresh = 0.5

class ActFun(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input-thresh) < 0.5
        return grad_input * temp.float()

act_fun = ActFun.apply

class ANN(nn.Module):
    def __init__(self, n_tasks):
        super(ANN, self).__init__()
        self.n_tasks = n_tasks
        for i in range(self.n_tasks):
            setattr(self, 'task_{}'.format(i), nn.Linear(50, 10))
            
        self.fc1 = nn.Linear(36*36*1,512)
        self.fc_output = nn.Linear(512,50)
        
    def forward(self,input):
        batch_size = input.shape[0]
        input = input.float().view(batch_size,-1)
        x = self.fc1(input)
        x = nn.ReLU()(x)
        spike = act_fun(x)
        outs = self.fc_output(spike)
        
        outputs = []
        for i in range(self.n_tasks):
            layer = getattr(self, 'task_{}'.format(i))
            outputs.append(layer(outs))
        return torch.stack(outputs, dim=1)