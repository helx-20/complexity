import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import CrossEntropyLoss

v_min, v_max = -1e3, 1e3
thresh = 0.3
lens = 0.5
decay = 0.2
device = torch.device("cpu")

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

# cnn_layer(in_planes, out_planes, stride, padding, kernel_size)
cfg_cnn = [(1, 40, 1, 1, 3),
           (40, 40, 1, 1, 3),]
# kernel size
cfg_kernel = [36, 18, 9]
# fc layer
cfg_fc = [512, 50]

class SCNN_Model_4LIF(nn.Module):

    def __init__(self, n_tasks, win):
        super(SCNN_Model_4LIF, self).__init__()
        self.n_tasks = n_tasks
        for i in range(self.n_tasks):
            setattr(self, 'task_{}'.format(i), nn.Linear(50, 10))

        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[0]
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding).to(device)

        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[1]
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding).to(device)

        self.fc = nn.Linear(cfg_kernel[-1] * cfg_kernel[-1] * cfg_cnn[-1][1], cfg_fc[1]).to(device)
        
        self.win = win
        self.add_op = 0
        self.mul_op = 0
        print(self)
        
    def mem_update(self,ops, x, mem, spike):
        torch.cuda.empty_cache()
        mem = mem * decay * (1 - spike) + ops(x)
        spike = act_fun(mem)

        if isinstance(ops,nn.Conv2d):
            weight = ops.weight
            kernel_size = weight.shape[3]
            inchannel =  weight.shape[1]
            outchennel = weight.shape[0]
            outsize = x.shape[-1]      
            self.add_op +=  (inchannel * kernel_size**2 - 1) * outchennel * outsize**2
            self.mul_op +=  inchannel * outchennel * outsize**2 * kernel_size**2 
        else:
            input_size  = ops.in_features
            hedden_size = ops.out_features
            self.mul_op += input_size*hedden_size 
            self.add_op += (input_size-1)*hedden_size
        
        return mem, spike

    def forward(self, input_):
        win = self.win
        batch_size = input_.size(0)
        c1_spike = torch.zeros(batch_size, cfg_cnn[0][1], cfg_kernel[0], cfg_kernel[0],device=device)
        c1_mem = torch.zeros(batch_size, cfg_cnn[0][1], cfg_kernel[0], cfg_kernel[0],device=device)
        c2_spike = torch.zeros(batch_size, cfg_cnn[1][1], cfg_kernel[1], cfg_kernel[1],device=device)
        c2_mem = torch.zeros(batch_size, cfg_cnn[1][1], cfg_kernel[1], cfg_kernel[1],device=device)
        p1_sumspike = torch.zeros(batch_size, cfg_cnn[0][1], cfg_kernel[1], cfg_kernel[1],device=device)
        p2_sumspike = torch.zeros(batch_size, cfg_cnn[-1][1], cfg_kernel[-1], cfg_kernel[-1],device=device)
        for step in range(win):
            c1_mem, c1_spike = self.mem_update(self.conv1, input_, c1_mem, c1_spike)
            p1_spike = F.avg_pool2d(c1_spike, 2)
            p1_sumspike+=p1_spike

            c2_mem, c2_spike = self.mem_update(self.conv2, p1_spike, c2_mem, c2_spike)
            p2_spike = F.avg_pool2d(c2_spike, 2)
            p2_sumspike += p2_spike

        sumspike = p2_sumspike.view(batch_size, -1)
        
        print(sumspike.shape)
        
        outs = self.fc(sumspike / win)

        self.mul_op += cfg_kernel[-1] * cfg_kernel[-1] * cfg_cnn[-1][1]*cfg_fc[1] 
        self.mul_op += self.win*3*cfg_kernel[0] * cfg_kernel[0] * cfg_cnn[0][1]#inside neuron
        self.mul_op += self.win*3*cfg_kernel[1] * cfg_kernel[1] * cfg_cnn[1][1]#inside neuron
        self.add_op += (cfg_kernel[-1] * cfg_kernel[-1] * cfg_cnn[-1][1]-1)*cfg_fc[1] 
        self.add_op += self.win*3*cfg_kernel[0] * cfg_kernel[0] * cfg_cnn[0][1]#inside neuron
        self.add_op += self.win*3*cfg_kernel[1] * cfg_kernel[1] * cfg_cnn[1][1]#inside neuron
        
        output=[]
        for i in range(self.n_tasks):
            layer = getattr(self, 'task_{}'.format(i))
            output.append(layer(outs))
            
        return p2_sumspike