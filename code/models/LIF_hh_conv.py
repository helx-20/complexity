import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import CrossEntropyLoss

device = torch.device("cuda:0")
thresh = 0.3 # neuronal threshold
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
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - thresh) < lens
        return grad_input * temp.float()

act_fun = ActFun.apply
# membrane potential update
def mem_update(x, mem, spike):
    mem1 = mem * decay * (1. - spike) + x
    spike1 = act_fun(mem1) # act_fun : approximation firing function
    return mem1, spike1

# cnn_layer(in_planes, out_planes, stride, padding, kernel_size)
cfg_cnn = [(1, 10, 1, 1, 3),
           (10, 10, 1, 1, 3),]
# kernel size
cfg_kernel = [36, 18, 9]
# fc layer
cfg_fc = [512, 50]
        
class lif_hh(nn.Module):
    def __init__(self,in_planes, out_planes, kernel_size, stride, padding):
        super(lif_hh,self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding).to(device)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding).to(device)
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding).to(device)
    
    def forward(self,input,mem,spike,ops,is_spike_input = True):
        #print(input.size())
        if is_spike_input :
            input1 = self.conv1(input[:,:,:,:,0])+self.conv1(input[:,:,:,:,1])+self.conv1(input[:,:,:,:,2])+self.conv1(input[:,:,:,:,3])
            input2 = self.conv2(input[:,:,:,:,0])+self.conv2(input[:,:,:,:,1])+self.conv2(input[:,:,:,:,2])+self.conv2(input[:,:,:,:,3])
            input3 = self.conv3(input[:,:,:,:,0])+self.conv3(input[:,:,:,:,1])+self.conv3(input[:,:,:,:,2])+self.conv3(input[:,:,:,:,3])
        else:
            input1 = self.conv1(input)
            input2 = self.conv2(input)
            input3 = self.conv3(input)
        inner_input = ops(mem[:,:,:,:,0:3])
        mem1 = torch.zeros_like(mem,device=device)
        spike1 = torch.zeros_like(spike,device=device)
        #print(input1.size(),mem[:,:,:,:,0].size())
        mem1[:,:,:,:,0],spike1[:,:,:,:,0] = mem_update(input1,mem[:,:,:,:,0],spike[:,:,:,:,0])
        mem1[:,:,:,:,1],spike1[:,:,:,:,1] = mem_update(input2,mem[:,:,:,:,1],spike[:,:,:,:,1])
        mem1[:,:,:,:,2],spike1[:,:,:,:,2] = mem_update(input3,mem[:,:,:,:,2],spike[:,:,:,:,2])
        mem1[:,:,:,:,3],spike1[:,:,:,:,3] = mem_update(inner_input[:,:,:,:,0],mem[:,:,:,:,3],spike[:,:,:,:,3])
        return mem1,spike1

class SCNN_Model_LIF_hh(nn.Module):
    def __init__(self, n_tasks):
        super(SCNN_Model_LIF_hh, self).__init__()
        self.n_tasks = n_tasks
        for i in range(self.n_tasks):
            setattr(self, 'task_{}'.format(i), nn.Linear(50, 10))

        self.lif_fc = nn.Linear(3,1).to(device)
        
        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[0]
        #self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        self.lif_4_1 = lif_hh(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[1]
        #self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        self.lif_4_2 = lif_hh(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        #self.fc1 = nn.Linear(cfg_kernel[-1] * cfg_kernel[-1] * cfg_cnn[-1][1]*4, cfg_fc[0])
        self.fc = nn.Linear(cfg_kernel[-1] * cfg_kernel[-1] * cfg_cnn[-1][1]*4, cfg_fc[1]).to(device)

    def forward(self, input, time_window = 10):
        #print(input.size())
        batch_size = input.size(0)
        c1_spike = torch.zeros(batch_size, cfg_cnn[0][1], cfg_kernel[0], cfg_kernel[0], 4,device=device)
        c1_mem = torch.zeros(batch_size, cfg_cnn[0][1], cfg_kernel[0], cfg_kernel[0], 4,device=device)
        c2_spike = torch.zeros(batch_size, cfg_cnn[1][1], cfg_kernel[1], cfg_kernel[1], 4,device=device)
        c2_mem = torch.zeros(batch_size, cfg_cnn[1][1], cfg_kernel[1], cfg_kernel[1], 4,device=device)
        h1_mem = h1_spike = h1_sumspike = torch.zeros(batch_size, cfg_kernel[-1] * cfg_kernel[-1] * cfg_cnn[-1][1]*4, device=device)
        for step in range(time_window): # simulation time steps
            c1_mem,c1_spike = self.lif_4_1(input,c1_mem,c1_spike,self.lif_fc,False)
            x = torch.zeros(c1_spike.size(0),c1_spike.size(1),c1_spike.size(2)//2,c1_spike.size(3)//2,4,device=device)
            x[:,:,:,:,0] = F.avg_pool2d(c1_spike[:,:,:,:,0], 2)
            x[:,:,:,:,1] = F.avg_pool2d(c1_spike[:,:,:,:,1], 2)
            x[:,:,:,:,2] = F.avg_pool2d(c1_spike[:,:,:,:,2], 2)
            x[:,:,:,:,3] = F.avg_pool2d(c1_spike[:,:,:,:,3], 2)
            #x = self.conv2(x)
            c2_mem,c2_spike = self.lif_4_2(x,c2_mem,c2_spike,self.lif_fc)
            x = torch.zeros(c2_spike.size(0),c2_spike.size(1),c2_spike.size(2)//2,c2_spike.size(3)//2,4,device=device)
            x[:,:,:,:,0] = F.avg_pool2d(c2_spike[:,:,:,:,0], 2)
            x[:,:,:,:,1] = F.avg_pool2d(c2_spike[:,:,:,:,1], 2)
            x[:,:,:,:,2] = F.avg_pool2d(c2_spike[:,:,:,:,2], 2)
            x[:,:,:,:,3] = F.avg_pool2d(c2_spike[:,:,:,:,3], 2)
            #x = F.avg_pool2d(c2_spike[:,:,:,:,3], 2)
            x = x.view(batch_size, -1)
            h1_sumspike += x
        outs = self.fc(h1_sumspike / time_window)

        output=[]
        for i in range(self.n_tasks):
            layer = getattr(self, 'task_{}'.format(i))
            output.append(layer(outs))
        return torch.stack(output, dim=1)
