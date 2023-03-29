from statistics import mode
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl

seed_dict = [0,1,2,3,4,5,6,7,8,9,10]
need_4lif = True
length = 300000
total_reward_dict = {"LIF":[], "HH":[], "LIF_HH":[], "4LIF":[],"ANN":[]}
mean_reward_dict = {"LIF":[], "HH":[], "LIF_HH":[], "4LIF":[],"ANN":[]}
std_reward_dict = {"LIF":[], "HH":[], "LIF_HH":[], "4LIF":[],"ANN":[]}
label_dict = {"LIF":"s-LIF", "HH":"HH", "LIF_HH":"s-LIF2HH", "4LIF":"4s-LIF","ANN":"ANN"}
for seed in seed_dict:
    for model_name in ["LIF","HH","LIF_HH","4LIF","ANN"]:
        if model_name == "4LIF" and not need_4lif:
            continue
        data = np.load("./record/{}/reward_iteration_{}.npy".format(model_name, seed), allow_pickle=True)
        data = data.tolist()
        iteration = data["iteration"]
        reward_dict = data["reward_dict"]
        avg_reward_dict = reward_dict
        n = 1000
        for i in range(iteration - n):
            avg_reward_dict[i] = np.mean(reward_dict[i:i+n])
        total_reward_dict[model_name].append(avg_reward_dict[0:length])
        #plt.plot(iter, avg_reward_dict[0:iteration-n])
        
for model_name in ["LIF","HH","LIF_HH","4LIF","ANN"]:
    if(model_name == "4LIF" and not need_4lif):
        continue
    std_reward_dict[model_name] = np.std(total_reward_dict[model_name], axis=0)
    mean_reward_dict[model_name] = np.mean(total_reward_dict[model_name], axis=0)

iter = np.linspace(start=1, stop=length, num=length)
color_number = 0
color_dict = [[30,30,230],[220,180,30],[255,20,0],[80,180,100],[255,150,80]]
for model_name in ["LIF","HH","LIF_HH","4LIF","ANN"]:
    if(model_name == "4LIF" and not need_4lif):
        continue
    r,g,b = np.array(color_dict[color_number])/255
    color_number += 1
    plt.fill_between(iter, mean_reward_dict[model_name]-1*std_reward_dict[model_name],
                     mean_reward_dict[model_name]+1*std_reward_dict[model_name],
                     color=(r, g, b, 0.1))
    plt.plot(iter, mean_reward_dict[model_name], label=label_dict[model_name], linewidth=0.8, color = (r,g,b))
plt.legend(prop = {'size':17}, loc = "lower right")
plt.xlabel("iteration",fontsize = 17)
plt.ylabel("average reward",fontsize = 17)
plt.tick_params(labelsize=14) #调整坐标轴数字大小

ax = pl.gca()
ax.xaxis.get_major_formatter().set_powerlimits((0,1))

plt.savefig("reward_picture.png",dpi=600)