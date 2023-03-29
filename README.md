# Network Model with  Internal Complexity Bridges Artificial Intelligence and Neuroscience
## Time-varied model simulation
To get the results in Fig.3a and Fig.3b, you can run the main.m in "time_varied_simulation/fully-connected" or "time_varied_simulation/xor" folder with MATLAB.

## Simplified model simulation
To get the results in Fig.3d, you can run the main.m in "simplified_simulation" folder with MATLAB. Different types and amplitudes of input signals can be chosen in main.m.

## Multi-task learning experiment
For model training, you can run the following code. Different models, including s-LIF, HH, s-LIF2HH, 4s-LIF and b-ANN, can be chosen in train_pure.py.
````
cd multi-task
python train_pure.py
````
For robustness test, the amplitude of noise can be modified in noise_test.py.
````
python noise_test.py
````

## Deep reinforcement learning experiment
For the InvertedPendulum environment, you can run the following code. The model name can chosen in "LIF", "HH", "LIF_HH", "4LIF" and "ANN". The results are recorded in the "record/model_name" folder.
````
cd drl_InvertedPendulum
python main.py --model_name LIF
````
For the InvertedDoublePendulum environment, you can run the following code. The model name can chosen in "LIF", "HH", "LIF_HH", "4LIF" and "ANN". The results are recorded in the "record/model_name" folder.
````
cd drl_InvertedDoublePendulum
python main.py --model_name LIF
````

## Mutual information analysis
To measure the mutual information of each network, you can run the following code. The model name "LIF" can be changed to "HH" or "LIF-HH".
````
cd MI
python main.py --model LIF
````

## FLOPs measurement
To measure the FLOPs in each network, you can run the cal_flops.ipynb in "calculate_flops" folder.
