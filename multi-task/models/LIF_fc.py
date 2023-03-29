import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import CrossEntropyLoss
# Here are the main function of learning HH models that set the conductant parameters learnable
#  If you want to reset the relevant parameters after spiking, replace "update_neuron" with "update_neuron_reset". Both are defined as below.

v_min, v_max = -1e3, 1e3
batch_size = 20
# tau_w  = 30
num_epochs = 101
learning_rate = 5e-4
time_window = 5
thresh = 0.3
lens = 0.5
decay = 0.2
num_classes = 10
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

class SNN_Model_LIF(nn.Module):

    def __init__(self, n_tasks):
        super(SNN_Model_LIF, self).__init__()
        self.n_tasks = n_tasks
        for i in range(self.n_tasks):
            setattr(self, 'task_{}'.format(i), nn.Linear(50, 10))

        self.fc1 = nn.Linear(36*36*1, cfg_fc[0], )
        self.fc = nn.Linear(cfg_fc[0], cfg_fc[1])

    def forward(self, input, win=15):
        batch_size = input.size(0)
        h1_mem = h1_spike = torch.zeros(batch_size, cfg_fc[0], device=device)
        h1_sumspike = torch.zeros(batch_size, cfg_fc[0], device=device)
        for step in range(win):

            x = input.view(batch_size, -1)
            h1_mem, h1_spike = mem_update(self.fc1, x, h1_mem, h1_spike)
            h1_sumspike = h1_sumspike + h1_spike

        outs = self.fc(h1_sumspike/win)
        #print(torch.mean(h1_sumspike/win))

        output=[]
        for i in range(self.n_tasks):
            layer = getattr(self, 'task_{}'.format(i))
            output.append(layer(outs))
        return torch.stack(output, dim=1)

def mem_update(fc, x, mem, spike):
  
    mem = mem * decay * (1 - spike) + fc(x)
    spike = act_fun(mem)
    return mem, spike
