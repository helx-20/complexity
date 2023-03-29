import torch.nn as nn
import torch
import torch.nn.functional as F
from scipy.integrate import odeint
from torch.distributions import Normal

v_min, v_max = -1e3, 1e3
thresh = 2.3
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
        #temp = abs(input-thresh) < lens
        temp = abs(input) < lens
        return grad_input * temp.float()

act_fun = ActFun.apply

cfg_fc = [16, 16, 16, 2]

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
        self.dt = 1e-1 * 2 * torch.rand(out_planes, requires_grad=False)

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
        # learnable verision
        a_n = self.a_n_coe #* (10 - v) / (torch.exp((10 - v) / 10) - 1)
        b_n = self.b_n_coe #* torch.exp(-v / 80)
        a_m = self.a_m_coe #* (25 - v)/(torch.exp((25 - v) / 10) - 1)
        b_m = self.b_m_coe #* torch.exp(-v / 18)
        b_h = self.b_h_coe #/ (torch.exp((30 - v) / 10))
        a_h = self.a_h_coe #* torch.exp(-v / 20)
        '''
        a_n = 0.01 * (10 - v) / (torch.exp((10 - v) / 10) - 1)
        b_n = 0.125 * torch.exp(-v / 80)
        a_m = 0.1 * (25 - v)/(torch.exp((25 - v) / 10) - 1)
        b_m = 4 * torch.exp(-v / 18)
        b_h = 1. / (torch.exp((30 - v) / 10) + 1)
        a_h = 0.07 * torch.exp(-v / 20)
        '''
        g_Na = self.gbar_Na * h * m ** 3
        g_K = self.gbar_K * (y ** 4)
        I_Na = g_Na * (v - self.V_Na)
        I_K = g_K * (v - self.V_K)
        I_L = self.gbar_L * (v - self.V_L)
        
        new_v = v + (inputs - I_Na - I_K - I_L) * self.dt  # emit C_m =1
        new_n = y + (a_n * (1 - y) - b_n * y) * self.dt
        new_m = m + (a_m * (1 - m) - b_m * m) * self.dt
        new_h = h + (a_h * (1 - h) - b_h * h) * self.dt

        #spikes = s #act_fun(new_v - thresh)
        #new_v = new_v * (1 - spikes)

        spike_out = act_fun(new_v - thresh)
        new_state = (new_v, new_n, new_m, new_h)
        #print("new:",new_state[0],new_state[1],new_state[2],new_state[3])
        return new_state,spike_out

    def forward(self, input, wins=15):
        #print("input:",input)
        #print("weight:",self.fc.weight.data)
        batch_size = input.size(0)
        state1 = self.zeros_state([batch_size, self.oup])
        mems = torch.zeros([batch_size, wins, self.oup]).to(device)

        #print("new:")
        for step in range(wins):
            state1,spike_out = self.update_neuron(self.fc(input[:,step,:]), state1)
            mems[:,step,:] = spike_out
        #    print(torch.mean(state1[0]))
        #print("state:",state1[0])
        return mems

class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        # Q1 architecture
        self.fc1 = HH_neuron(num_inputs + num_actions, hidden_dim)	
        self.fc_output1 = nn.Linear(hidden_dim, 1)
        
        # Q2 architecture
        self.fc2 = HH_neuron(num_inputs + num_actions, hidden_dim)	
        self.fc_output2 = nn.Linear(hidden_dim, 1)
        
        self.apply(weights_init_)

    def forward(self, state, action, wins=15):
        #print(state.size(),action.size())
        input = torch.cat([state, action], 2)
        batch_size = input.size(0)
        #input = input.view(batch_size, wins, -1).float().to(device) 

        output1 = self.fc1(input, wins)
        out1 = self.fc_output1(torch.mean(output1, dim=1))
        
        output2 = self.fc2(input, wins)
        out2 = self.fc_output2(torch.mean(output2, dim=1))
        #print("spiking_rate:",input_seq.size(),torch.sum(input_seq)/(sizes[0]*sizes[1]*sizes[2]))

        return out1, out2

class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()
        
        self.hh_layer1 = HH_neuron(num_inputs, hidden_dim)
        self.hh_layer2 = HH_neuron(hidden_dim, hidden_dim)
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
        x = self.hh_layer1(state)
        #x = self.hh_layer2(x)
        x = torch.mean(x, dim=1)
        #print(torch.mean(x))
        x1 = self.linear1_1(x)
        x1 = nn.ReLU()(x1)
        x1 = self.linear1_2(x1)
        x1 = nn.ReLU()(x1)
        #x1 = torch.mean(x1, dim=1)
        mean = self.mean_linear(x1)
        x2 = self.linear2_1(x)
        x2 = nn.ReLU()(x2)
        x2 = self.linear2_2(x2)
        x2 = nn.ReLU()(x2)
        #x2 = torch.mean(x2, dim=1)
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