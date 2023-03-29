from xml.dom import xmlbuilder
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
device = torch.device("cuda:0")
thresh = 0.8 # neuronal threshold
lens = 0.4 # hyper-parameters of approximate function
decay = 0.2 # decay constants
num_classes = 10
#batch_size  = 50
learning_rate = 1e-3
num_epochs = 100 # max epoch
# define approximate firing function
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
# membrane potential update
def mem_update(x, x_4, mem, spike, ops):
    for i in range(3):
        mem[:,:,:,:,i] = mem[:,:,:,:,i] * decay * (1. - spike) + x
    mem[:,:,:,:,3] = mem[:,:,:,:,3] * decay * (1. - spike) + x_4
    #mem1 = mem[:,:,:,:,0] + mem[:,:,:,:,1] + mem[:,:,:,:,2] + mem[:,:,:,:,3]
    mem1 = ops(mem[:,:,:,:,0:3])
    mem1 = mem1[:,:,:,:,0] + mem[:,:,:,:,3]
    spike1 = act_fun(mem1-thresh) # act_fun : approximation firing function
    return mem, spike1

# cnn_layer(in_planes, out_planes, stride, padding, kernel_size)
cfg_cnn = [(1, 10, 1, 1, 3),
           (10, 30, 1, 1, 3),]
# kernel size
cfg_feature_size = [36, 18, 9]
# fc layer
cfg_fc = [512, 50]

class LIF_hh_neuron(nn.Module):

    def __init__(self, in_planes,out_planes,stride,padding, kernel_size, feature_map_size):
        super(LIF_hh_neuron, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding).to(device)
        self.conv.weight.data = self.conv.weight.data * 5
        self.inp = in_planes
        self.channel = out_planes
        self.fms = feature_map_size
        self.lif_fc = nn.Linear(3,1).to(device)
        #self.lif_fc2 = nn.Linear(3,1).to(device)
        #self.lif_fc.weight.requires_grad = False
        #self.lif_fc2.weight.requires_grad = False
        self.thresh = thresh

    def update_neuron(self,input,mem,spike,ops):
        inner_input = ops(mem[:,:,:,:,0:3])
        mem1 = torch.zeros_like(mem,device=device)
        spike_out = torch.zeros_like(spike,device=device)
        #print(input1.size(),mem[:,:,:,:,0].size())
        mem1,spike_out = mem_update(input,inner_input[:,:,:,:,0],mem,spike,ops)
        return mem1, spike_out

    def forward(self, input, wins=15):
        #print(input.size())
        batch_size = input.size(0)

        mem = torch.zeros([batch_size, self.channel, self.fms, self.fms, 4]).to(device)
        spike = torch.zeros([batch_size, self.channel, self.fms, self.fms]).to(device)
        spikes = torch.zeros([batch_size, wins, self.channel, self.fms//2, self.fms//2]).to(device)
    
        for step in range(wins):
            conv_output = self.conv(input[:,step,...])
            mem, spike = self.update_neuron(conv_output, mem, spike, self.lif_fc)
            spike_out = F.avg_pool2d(spike, 2)
            spikes[:,step,...] = spike_out
            #print(state1[0])
            #print(torch.mean(state1[0]))
        #print("end")
        return spikes

class SCNN_Model_LIF_hh(nn.Module):
    def __init__(self, n_tasks):
        super(SCNN_Model_LIF_hh, self).__init__()
        self.n_tasks = n_tasks
        for i in range(self.n_tasks):
            setattr(self, 'task_{}'.format(i), nn.Linear(50, 10))

        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[0]
        fms = cfg_feature_size[0]
        self.fc1 = LIF_hh_neuron(in_planes, out_planes, stride, padding, kernel_size, fms)
        
        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[1]
        fms = cfg_feature_size[1]
        self.fc2 = LIF_hh_neuron(in_planes, out_planes, stride, padding, kernel_size, fms)

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

        outputs=[]
        for i in range(self.n_tasks):
            layer = getattr(self, 'task_{}'.format(i))
            outputs.append(layer(outs))
        return torch.stack(outputs, dim=1)
