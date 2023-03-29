import numpy as np
import torch.nn as nn
import torch
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import math
import torch.nn.functional as F
# Here are the main function of learning HH models that set the conductant parameters learnable
#  If you want to reset the relevant parameters after spiking, replace "update_neuron" with "update_neuron_reset". Both are defined as below.

v_min, v_max = -1e3, 1e3
batch_size = 20
# tau_w  = 30
num_epochs = 101
learning_rate = 5e-4
time_window = 5
thresh = 2.3
lens = 0.5
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

cfg_cnn = [(1, 10, 1, 1, 3),
           (10, 10, 1, 1, 3),]
# kernel size
cfg_feature_size = [36, 18, 9]
# fc layer
cfg_fc = [512, 50]


class HH_neuron(nn.Module):

    def __init__(self, in_planes,out_planes,stride,padding, kernel_size,feature_map_size):
        super(HH_neuron, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding).to(device)
        self.conv.weight.data = self.conv.weight.data * 1
        self.inp = in_planes
        self.channel = out_planes
        self.fms = feature_map_size

        self.V_Na = 115
        self.V_K = -12
        self.V_L = 10.6

        self.gbar_Na = 120
        self.gbar_K = 36
        self.gbar_L = 0.3

        coe_scaling = 1e-2
        self.dt = 1e-1
        self.thresh = thresh
        learnable = False
        self.a_n_coe_conv1 =  nn.Parameter(coe_scaling * torch.rand(self.fms), requires_grad=learnable).to(device)
        self.b_n_coe_conv1 =  nn.Parameter(coe_scaling * torch.rand(self.fms), requires_grad=learnable).to(device)
        self.a_m_coe_conv1 =  nn.Parameter(coe_scaling * torch.rand(self.fms), requires_grad=learnable).to(device)
        self.b_m_coe_conv1 =  nn.Parameter(coe_scaling * torch.rand(self.fms), requires_grad=learnable).to(device)
        self.a_h_coe_conv1 =  nn.Parameter(coe_scaling * torch.rand(self.fms), requires_grad=learnable).to(device)
        self.b_h_coe_conv1 =  nn.Parameter(coe_scaling * torch.rand(self.fms), requires_grad=learnable).to(device)

    def zeros_state(self, size):
        zero_state = [torch.zeros(*size).to(device)]*4
        return zero_state

    def update_neuron(self, inputs, states=None):
        if states is None:
            v, y, m, h = torch.zeros_like(inputs).to(device),torch.zeros_like(inputs).to(device),torch.zeros_like(inputs).to(device),torch.zeros_like(inputs).to(device)
        else:
            v, y, m, h = states
            v, y, m, h = v.to(device), y.to(device), m.to(device), h.to(device)
        # learnable verision
        a_n = self.a_n_coe_conv1 #* (10 - v) / (torch.exp((10 - v) / 10) - 1)
        b_n = self.b_n_coe_conv1 #* torch.exp(-v / 80)
        a_m = self.a_m_coe_conv1 #* (25 - v)/(torch.exp((25 - v) / 10) - 1)
        b_m = self.b_m_coe_conv1 #* torch.exp(-v / 18)
        b_h = self.a_h_coe_conv1 #/ (torch.exp((30 - v) / 10))
        a_h = self.b_h_coe_conv1 #* torch.exp(-v / 20)

        g_Na = self.gbar_Na * h * m ** 3
        g_K = self.gbar_K * (y ** 4)
        I_Na = g_Na * (v - self.V_Na)
        I_K = g_K * (v - self.V_K)
        I_L = self.gbar_L * (v - self.V_L)
    
        new_v = v + (inputs - I_Na.detach() - I_K.detach() - I_L.detach()) * self.dt  # emit C_m =1
        new_n = y + (a_n * (1 - y) - b_n * y) * self.dt
        new_m = m + (a_m * (1 - m) - b_m * m) * self.dt
        new_h = h + (a_h * (1 - h) - b_h * h) * self.dt

        spike_out = act_fun(new_v - self.thresh)
        new_state = (new_v, new_n, new_m, new_h)
        return new_state,spike_out

    def forward(self, input, wins=15):
        #print(input.size())
        batch_size = input.size(0)

        state1 = self.zeros_state([batch_size, self.channel, self.fms, self.fms])
        mems = torch.zeros([batch_size, wins, self.channel, self.fms//2, self.fms//2]).to(device)
    
        for step in range(wins):
            conv_output = self.conv(input[:,step,...])
            state1,spike_out = self.update_neuron(conv_output, state1)
            spike_out = F.avg_pool2d(spike_out, 2)
            mems[:,step,...] = spike_out
            #print(state1[0][0,0,0,0])
            #print(torch.mean(state1[0]))
        #print("end")
        return mems

class SCNN_Model_HH(nn.Module):
    def __init__(self, n_tasks):
        super(SCNN_Model_HH, self).__init__()
        self.n_tasks = n_tasks
        for i in range(self.n_tasks):
            setattr(self, 'task_{}'.format(i), nn.Linear(50, 10))
            
        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[0]
        fms = cfg_feature_size[0]
        self.fc1 = HH_neuron(in_planes, out_planes, stride, padding, kernel_size, fms)
        
        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[1]
        fms = cfg_feature_size[1]
        self.fc2 = HH_neuron(in_planes, out_planes, stride, padding, kernel_size, fms)

        self.layers = nn.Sequential(self.fc1,self.fc2)		
        self.fc_output = nn.Linear(cfg_feature_size[-1] * cfg_feature_size[-1] * cfg_cnn[-1][1], cfg_fc[1])

    def forward(self, input, wins=15):

        batch_size = input.size(0)
        input = input.float().to(device) 

        input_seq = torch.stack([input]*wins,dim=1)
        
        for layer in self.layers:
            input_seq = layer(input_seq,wins)
            sizes = input_seq.size()
            #print(input_seq.size(),torch.sum(input_seq)/(sizes[0]*sizes[1]*sizes[2]*sizes[3]*sizes[4]))
    
        input_seq = input_seq.view(batch_size, wins, -1).float().to(device)
        output = torch.mean(input_seq,dim=1)

        outs = self.fc_output(output)

        outputs = []
        for i in range(self.n_tasks):
            layer = getattr(self, 'task_{}'.format(i))
            outputs.append(layer(outs))
        return torch.stack(outputs, dim=1)