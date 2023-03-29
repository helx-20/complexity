import torch.nn as nn
import torch
import torch.nn.functional as F
from scipy.integrate import odeint
from torch.distributions import Normal

v_min, v_max = -1e3, 1e3
thresh = 0.2
lens = 0.4
decay = 0.2
device = torch.device("cpu")

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

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

cfg_fc = [16, 16, 16, 2]

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
        batch_size = input.size(0)
        spikes = torch.zeros(batch_size, win, self.out, device=device)
        h1_mem = h1_spike = h1_sumspike = torch.zeros(batch_size, self.out, device=device)
        for step in range(win):
            x = input[:, step,...].view(batch_size, -1)
            h1_mem, h1_spike = self.mem_update(self.fc, x, h1_mem, h1_spike)
            h1_sumspike = h1_sumspike + h1_spike
            spikes[:,step,:] = h1_spike
        return spikes

def mem_update(self, fc, x, mem, spike):
        mem1 = mem * self.decay * (1 - spike) + fc(x)
        spike = act_fun(mem1-self.thresh)
        return mem1, spike

class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        # Q1 architecture
        self.fc1 = nn.Linear(num_inputs + num_actions, hidden_dim)	
        self.fc_output1 = nn.Linear(hidden_dim, 1)
        
        # Q2 architecture
        self.fc2 = nn.Linear(num_inputs + num_actions, hidden_dim)	
        self.fc_output2 = nn.Linear(hidden_dim, 1)
        
        self.apply(weights_init_)

    def forward(self, state, action, wins=15):
        inputs = torch.cat([state, action], 2)
        batch_size = inputs.size(0)

        h1_mem = h1_spike = torch.zeros(batch_size, self.hidden_dim, device=device)
        h1_sumspike = torch.zeros(batch_size, self.hidden_dim, device=device)
        h2_mem = h2_spike = torch.zeros(batch_size, self.hidden_dim, device=device)
        h2_sumspike = torch.zeros(batch_size, self.hidden_dim, device=device)

        for step in range(wins):
            x = inputs[:,step,...].view(batch_size, -1)
            h1_mem, h1_spike = mem_update(self.fc1, x, h1_mem, h1_spike)
            h1_sumspike = h1_sumspike + h1_spike
            
            h2_mem, h2_spike = mem_update(self.fc2, x, h2_mem, h2_spike)
            h2_sumspike = h2_sumspike + h2_spike
            
        out1 = self.fc_output1(h1_sumspike/wins)
        out2 = self.fc_output2(h2_sumspike/wins)
        return out1, out2
    
class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()
        
        self.lif_layer = LIF_neuron(num_inputs, hidden_dim)
        self.linear1_1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear1_2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2_1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2_2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        input_tmp = []
        for i in range(5):
            input_tmp += [state[:,i,...]]*3
        state = torch.stack(input_tmp,dim=1) 
        x = self.lif_layer(state)
        x = torch.mean(x, dim=1)
        x1 = self.linear1_1(x)
        x1 = nn.ReLU()(x1)
        x1 = self.linear1_2(x1)
        x1 = nn.ReLU()(x1)
        mean = self.mean_linear(x1)
        x2 = self.linear2_1(x)
        x2 = nn.ReLU()(x2)
        x2 = self.linear2_2(x2)
        x2 = nn.ReLU()(x2)
        log_std = self.log_std_linear(x2)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)