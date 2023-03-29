import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import CrossEntropyLoss

v_min, v_max = -1e3, 1e3
thresh = 0.3
lens = 0.5
decay = 0.2
device = torch.device("cpu")

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

cfg_fc = [512,50]

class SNN_Model_LIF(nn.Module):

    def __init__(self, n_tasks, win):
        super(SNN_Model_LIF, self).__init__()
        self.n_tasks = n_tasks
        for i in range(self.n_tasks):
            setattr(self, 'task_{}'.format(i), nn.Linear(50, 10))

        self.fc1 = nn.Linear(36*36*1, cfg_fc[0], )
        self.fc = nn.Linear(cfg_fc[0], cfg_fc[1])
        self.win = win
        self.add_op = 0
        self.mul_op = 0
        print(self)

    def forward(self, input, win=15):
        win = self.win
        batch_size = input.size(0)
        h1_mem = h1_spike = torch.zeros(batch_size, cfg_fc[0], device=device)
        h1_sumspike = torch.zeros(batch_size, cfg_fc[0], device=device)
        for step in range(win):

            x = input.view(batch_size, -1)
            h1_mem, h1_spike = self.mem_update(self.fc1, x, h1_mem, h1_spike)
            h1_sumspike = h1_sumspike + h1_spike

        outs = self.fc(h1_sumspike/win)
        self.mul_op += cfg_fc[0]*cfg_fc[1] 
        self.mul_op += self.win*3*cfg_fc[0]#inside neuron
        self.add_op += (cfg_fc[0]-1)*cfg_fc[1] 
        self.add_op += self.win*3*cfg_fc[0]#inside neuron

        output=[]
        for i in range(self.n_tasks):
            layer = getattr(self, 'task_{}'.format(i))
            output.append(layer(outs))
        return torch.stack(output, dim=1)

    def mem_update(self,layer, x, mem, spike):

        torch.cuda.empty_cache()
        mem = mem * decay * (1 - spike) + layer(x)
        spike = act_fun(mem)

        input_size  = layer.in_features
        hedden_size = layer.out_features
        self.mul_op += input_size*hedden_size 
        self.add_op += (input_size-1)*hedden_size
        
        return mem, spike

