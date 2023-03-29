import torch.nn as nn
import torch
import torch.nn.functional as F
from scipy.integrate import odeint
from torch.distributions import Normal

v_min, v_max = -1e3, 1e3
thresh = 2.3
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

class HH_neuron(nn.Module):

    def __init__(self, in_planes, out_planes):
        super(HH_neuron, self).__init__()
        self.fc = nn.Linear(in_planes, out_planes)
        self.apply(weights_init_)
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

    def forward(self, input, wins=15):
        input = input.float().to(device)
        if input.dim() == 2:
            batch_size = input.size(0)
            state1 = self.zeros_state([batch_size, self.oup])
            mems = torch.zeros([batch_size, wins, self.oup]).to(device)
            for step in range(wins):
                state1,spike_out = self.update_neuron(self.fc(input), state1)
                mems[:,step,:] = spike_out
        else:
            batch_size = input.size(1)
            state1 = self.zeros_state([input.size(0),batch_size, self.oup])
            mems = torch.zeros([input.size(0),batch_size, wins, self.oup]).to(device)
            for step in range(wins):
                state1,spike_out = self.update_neuron(self.fc(input), state1)
                mems[:,:,step,:] = spike_out
        return mems
