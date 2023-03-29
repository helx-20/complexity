import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F

v_min, v_max = -1e3, 1e3
thresh = 2
lens = 0.4
decay = 0.5
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

cfg_fc = [512,50]

class lif_hh(nn.Module):
    def __init__(self,in_planes, out_planes):
        super(lif_hh,self).__init__()
        self.fc1 = nn.Linear(in_planes,out_planes)
        self.fc2 = nn.Linear(in_planes,out_planes)
        self.fc3 = nn.Linear(in_planes,out_planes)
        self.lif_fc = nn.Linear(3,1)
        self.lif_fc.weight.data = abs(self.lif_fc.weight.data)
    
    def forward(self,input,mem,spike):
        input1 = self.fc1(input)
        input2 = self.fc2(input)
        input3 = self.fc3(input)
        inner_input = self.lif_fc(mem[:,:,0:3])
        mem1 = torch.zeros_like(mem,device=device)
        spike1 = torch.zeros_like(spike,device=device)
        mem1[:,:,0],spike1[:,:,0] = mem_update(input1,mem[:,:,0],spike[:,:,0])
        mem1[:,:,1],spike1[:,:,1] = mem_update(input2,mem[:,:,1],spike[:,:,1])
        mem1[:,:,2],spike1[:,:,2] = mem_update(input3,mem[:,:,2],spike[:,:,2])
        mem1[:,:,3],spike1[:,:,3] = mem_update(inner_input[:,:,0],mem[:,:,3],spike[:,:,3])
        return mem1,spike1

class SNN_Model_LIF_hh(nn.Module):

    def __init__(self, n_tasks):
        super(SNN_Model_LIF_hh, self).__init__()
        self.n_tasks = n_tasks
        for i in range(self.n_tasks):
            setattr(self, 'task_{}'.format(i), nn.Linear(50, 10))
        
        self.fc_output = nn.Linear(cfg_fc[0]*4, cfg_fc[1])
        self.lif_4 = lif_hh(36*36*1, cfg_fc[0])

    def forward(self, input, win=15):
        batch_size = input.size(0)
        h1_mem = h1_spike = torch.zeros(batch_size, cfg_fc[0], 4, device=device)
        h1_sumspike = torch.zeros(batch_size, cfg_fc[0], 4, device=device)
        for step in range(win):
            x = input.view(batch_size, -1)
            h1_mem, h1_spike = self.lif_4(x, h1_mem, h1_spike)
            h1_sumspike = h1_sumspike + h1_spike
        x = h1_sumspike
        x = x.view(batch_size,-1)
        outs = self.fc_output(x/win)

        output = []
        for i in range(self.n_tasks):
            layer = getattr(self, 'task_{}'.format(i))
            output.append(layer(outs))
        return torch.stack(output, dim=1)

def mem_update(x, mem, spike):
  
    mem = mem * decay * (1 - spike) + x
    spike1 = act_fun(mem)
    return mem, spike1

