import numpy as np
import torch.nn as nn
import torch
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import math
# Here are the main function of learning HH models that set the conductant parameters learnable
#  If you want to reset the relevant parameters after spiking, replace "update_neuron" with "update_neuron_reset". Both are defined as below.

v_min, v_max = -1e3, 1e3
thresh = 0.2
lens = 0.3
device = torch.device("cpu")

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

cfg_fc = [512, 50]

class HH_neuron(nn.Module):

    def __init__(self, in_planes, out_planes,win):
        super(HH_neuron, self).__init__()
        self.fc = nn.Linear(in_planes, out_planes)
        self.fc.weight.data = self.fc.weight.data * 0.01

        self.inp = in_planes
        self.oup = out_planes

        self.V_Na = 115
        self.V_K = -12
        self.V_L = 10.6

        self.gbar_Na = 120
        self.gbar_K = 36
        self.gbar_L = 0.3

        coe_scaling = 1e-2
        self.dt = 1e-2

        learnable = False
        self.a_n_coe =  nn.Parameter(coe_scaling * torch.rand(out_planes), requires_grad=learnable)
        self.b_n_coe =  nn.Parameter(coe_scaling * torch.rand(out_planes), requires_grad=learnable)
        self.a_m_coe =  nn.Parameter(coe_scaling * torch.rand(out_planes), requires_grad=learnable)
        self.b_m_coe =  nn.Parameter(coe_scaling * torch.rand(out_planes), requires_grad=learnable)
        self.a_h_coe =  nn.Parameter(coe_scaling * torch.rand(out_planes), requires_grad=learnable)
        self.b_h_coe =  nn.Parameter(coe_scaling * torch.rand(out_planes), requires_grad=learnable)
        self.win = win
        self.add_op = 0
        self.mul_op = 0
        
    def zeros_state(self, size):
        zero_state = [torch.zeros(*size).to(device)]*4
        return zero_state

    def update_neuron(self, inputs, states=None):
        if states is None:
            v, y, m, h = torch.zeros_like(inputs).to(device),torch.zeros_like(inputs).to(device),torch.zeros_like(inputs).to(device),torch.zeros_like(inputs).to(device)
        else:
            v, y, m, h = states
            v, y, m, h = v.to(device), y.to(device), m.to(device), h.to(device)

        a_n = self.a_n_coe
        b_n = self.b_n_coe
        a_m = self.a_m_coe
        b_m = self.b_m_coe
        b_h = self.b_h_coe
        a_h = self.a_h_coe

        g_Na = self.gbar_Na * h * m ** 3
        g_K = self.gbar_K * (y ** 4)
        I_Na = g_Na * (v - self.V_Na)
        I_K = g_K * (v - self.V_K)
        I_L = self.gbar_L * (v - self.V_L)
        
        new_v = v + (inputs - I_Na - I_K - I_L) * self.dt  # emit C_m =1
        new_n = y + (a_n * (1 - y) - b_n * y) * self.dt
        new_m = m + (a_m * (1 - m) - b_m * m) * self.dt
        new_h = h + (a_h * (1 - h) - b_h * h) * self.dt

        spike_out = act_fun(new_v - thresh)
        new_state = (new_v, new_n, new_m, new_h)
        return new_state,spike_out

    def forward(self, input,  wins=15):
        wins = self.win
        batch_size = input.size(0)
        state1 = self.zeros_state([batch_size, self.oup])
        spikes = torch.zeros([batch_size, wins, self.oup]).to(device)

        for step in range(wins):
            state1,spike_out = self.update_neuron(self.fc(input[:,step,:]), state1)
            
            input_size = self.fc.in_features
            hedden_size =  self.fc.out_features
            self.mul_op += input_size*hedden_size 
            self.add_op += (input_size-1)*hedden_size 
                
            spikes[:,step,:] = spike_out
        
        return spikes

class SNN_Model_HH(nn.Module):
    def __init__(self, n_tasks,win):
        super(SNN_Model_HH, self).__init__()
        self.n_tasks = n_tasks
        for i in range(self.n_tasks):
            setattr(self, 'task_{}'.format(i), nn.Linear(50, 10))

        self.fc1 = HH_neuron(36*36*1, cfg_fc[0],win)
        self.fc_output = nn.Linear(cfg_fc[0], cfg_fc[1])
        
        self.win = win
        self.add_op = 0
        self.mul_op = 0
        print(self)

    def forward(self, input, wins=15):
        batch_size = input.size(0)

        input = input.view(batch_size, -1).float().to(device) 

        input_seq = torch.stack([input]*wins,dim=1)
        
        input_seq = self.fc1(input_seq)

        output = torch.mean(input_seq,dim=1)
        outs = self.fc_output(output)
        
        self.add_op += self.fc1.add_op
        self.mul_op += self.fc1.mul_op

        self.mul_op += cfg_fc[0]*cfg_fc[1] 
        self.mul_op += self.win*23*cfg_fc[0]#inside neuron
        self.add_op += (cfg_fc[0]-1)*cfg_fc[1] 
        self.add_op += self.win*16*cfg_fc[0]#inside neuron
        
        outputs = []
        for i in range(self.n_tasks):
            layer = getattr(self, 'task_{}'.format(i))
            outputs.append(layer(outs))
        return torch.stack(outputs, dim=1)