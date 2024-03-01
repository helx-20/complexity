# Network Model with  Internal Complexity Bridges Artificial Intelligence and Neuroscience
## Requirements
````
pip install -r requirements.txt
````

## Time-varied model simulation
To get the results in Fig.3a and Fig.3b, you can __run the main.m__ in "time_varied_simulation/fully-connected" or "time_varied_simulation/xor" folder with MATLAB. These two files run the simulation of HH and tv-LIF2HH models for Supplementary Figures 2.1 and 2.2 .
````
matlab main
````

## Simplified model simulation
To get the results in Fig.3d, you can __run the main.m__ in "simplified_simulation" folder with MATLAB. Different types and amplitudes of input signals can be chosen in main.m. This file run the simulation of HH and s-LIF2HH models for Figure 3.
````
matlab main
````

## Multi-task learning experiment
For model training, you can __run the main.py__ in "multi-task" folder with Python. Different model names, including "LIF_fc", "HH_fc", "LIF_hh_fc", "4LIF_fc", "ANN", "LIF_conv", "HH_conv", "LIF_hh_conv", "4LIF_conv" and "CNN", can be chosen. 
````
cd multi-task
python main.py --model_name LIF_fc
````
For robustness test, the amplitude of noise can be modified by changing the value of "A". Different model names, including "LIF_fc", "HH_fc", "LIF_hh_fc", "4LIF_fc", "ANN", "LIF_conv", "HH_conv", "LIF_hh_conv", "4LIF_conv" and "CNN", can be chosen. You can __run the noise_test.py__ in "multi-task" folder with Python.
````
python noise_test.py --model_name LIF_fc --A 75
````

## Deep reinforcement learning experiment
For the InvertedPendulum environment, you can __run the main.py__ in "drl_InvertedPendulum" folder with Python. The model name can chosen in "LIF", "HH", "LIF_HH", "4LIF" and "ANN". The results are recorded in the "record/model_name" folder.
````
cd drl_InvertedPendulum
python main.py --model_name LIF
````
For the InvertedDoublePendulum environment, you can __run the main.py__ in "drl_InvertedDoublePendulum" folder with Python. The model name can chosen in "LIF", "HH", "LIF_HH", "4LIF" and "ANN". The results are recorded in the "record/model_name" folder. You can add state or reward noise and change the amplitude of noise in the main.py. 
````
cd drl_InvertedDoublePendulum
python main.py --model_name LIF
````

## Mutual information analysis
To measure the mutual information of each network, you can __run the main.py__ in "MI" folder with Python. The model name "LIF" can be changed to "HH" or "LIF_HH".
````
cd MI
python main.py --model LIF
````

## FLOPs measurement
To measure the FLOPs in each network, you can __run the cal_flops.ipynb__ in "calculate_flops" folder.
