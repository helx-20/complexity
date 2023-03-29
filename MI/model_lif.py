import torch.nn as nn
import torch
import torch.nn.functional as F
from scipy.integrate import odeint
from torch.distributions import Normal

v_min, v_max = -1e3, 1e3
thresh = 0.2
lens = 0.4
decay = 0.2
device = torch.device("cuda:0")

class ActFun(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(0.).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input) < lens
        return grad_input * temp.float()

act_fun = ActFun.apply

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class LIF_neuron(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(LIF_neuron, self).__init__()
        self.fc = nn.Linear(in_planes, out_planes)
        self.inp = in_planes
        self.out = out_planes
        self.decay = 0.2
        self.thresh = 0.2
        self.apply(weights_init_)

    def mem_update(self, fc, x, mem, spike):
        mem1 = mem * self.decay * (1 - spike) + fc(x)
        spike = act_fun(mem1-self.thresh)
        return mem1, spike

    def forward(self, input, win=15):
        if input.dim() == 2:
            batch_size = input.size(0)
            spikes = torch.zeros(batch_size, win, self.out, device=device)
            h1_mem = h1_spike = h1_sumspike = torch.zeros(batch_size, self.out, device=device)
            for step in range(win):
                x = input.view(batch_size, -1)
                h1_mem, h1_spike = self.mem_update(self.fc, x, h1_mem, h1_spike)
                h1_sumspike = h1_sumspike + h1_spike
                spikes[:,step,:] = h1_spike
        else:
            batch_size = input.size(1)
            spikes = torch.zeros(input.size(0),batch_size, win, self.out, device=device) 
            h1_mem = h1_spike = h1_sumspike = torch.zeros(input.size(0),batch_size, self.out, device=device)
            for step in range(win):
                x = input.view(input.size(0),batch_size, -1)
                h1_mem, h1_spike = self.mem_update(self.fc, x, h1_mem, h1_spike)
                h1_sumspike = h1_sumspike + h1_spike
                spikes[:,:,step,:] = h1_spike
        return spikes
