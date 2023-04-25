<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logo-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="docs/assets/logo-light.png">
  <img alt="Hydro logo" width="60%" src="docs/assets/logo.png">
</picture>
<h1>Surrogate-based Hyperparameter Tuning System</h1>
</div>

Hydro is a holistic system that automatically applies the hyperparameter transfer theory together with multiple system techniques to jointly improve the tuning efficiency. To learn more about how Hydro works, please refer our paper.


## Installation

```
pip install hydro-tune
```


## Artifact Evaluation
We provide code and document describes how to reproduce the key results presented in the OSDI '23 paper. 

We highlight three major claims to be evaluated in the artifact:

  1. **Automatic**. The surrogate model can be be automatically generated (via scaling and fusion) and evaluated.

  2. **Efficient**. Model scaling and fusion mechanisms can significantly improve training throughput and hardware efficiency.

  3. **Faithful**. Hydro can well maintain the hyperparameter fidelity even using a scaled and fused model.

### **A. Environment Preparation**

For convenient artifact evaluation, we will rent a 4-GPU cloud server (4x RTX 3090) for reviewers to reproduce all the experiments. These scripts have been tested on following environment:

|             Software             |           Hardware          |
|:--------------------------------:|:---------------------------:|
| Ray 2.3 <br /> PyTorch 2.0 <br /> Python 3.9 | 4x RTX 3090  <br />(24GB memory per GPU) |


If you want to reproduce the results on your own machine (recommend equipped with multiple GPUs), we provide two options to prepare the environment:

(Option 1) We suggest using a conda environment to install the dependencies:

```bash
conda create -n hydro python=3.9
conda activate hydro
cd Hydro
pip install -r requirements.txt
pip install hydro-tune
```

(Option 2) We also provide a Docker image to run the experiments in a container:

```bash
docker pull tonyhao96/hydro
```

### **B. Figure 8: Fusion and Scaling Efficiency.**
**(Evaluation Time Estimation: 10 minutes)**


To reproduce Figure 8 in the paper (that is, accumulated throughout and GPU memory footprint of fused surrogate model w.r.t packing the target model), one can use the following command line:

```bash
cd artifact
bash 1_run_fidelity.sh 
```

It sweep different fusion number with & without scaling ($S=8$ by default) of surrogate model, and packing target model directly. These workloads are placed on different GPUs. Note that here the MPS throughput is measured without starting the MPS server to avoid potential failures in subsequent experiments.

*The maximum number of model packing/fusion may vary significantly due to differences in hardware environments (as the paper uses A100 80GB), but the overall trend should remain consistent.*


`Visualize the Result`. 

+ Open `plot.ipynb` file

+ Run the first two cells (`1. Fusion Efficiency`) inside the notebook. 

+ From the Figure, we can see Hydro achieves over 10× training throughput improvement and over 20× GPU memory conservation (fuse more models). It should be noted that the MPS throughput measurements may not be entirely accurate due to the asynchronous execution of multiple trials, where some trials start after others have ended.

### **C. Figure 10: Scaling Fidelity Validation.**
**(Evaluation Time Estimation: 2 hours)** 

To reproduce Figure 10 in the paper (that is, hyperparameter transfer effect across different scaling ratios $S=1,2,4,8$), one can use the following command line:

```bash
bash 2_run_fidelity.sh 
```

It selects 8 hyperparameter sets ([batchsize, lr, momentum]) on the ResNet-18 model and build surrogate models with Hydro using different scaling ratios
$S = 2,4,8$, where $S = 1$ represents the target (i.e., original) model. We train each model for 100 epochs on the CIFAR-10 dataset with a fixed seed=1.

The output of this script looks like this:
```
== Status ==
Current time: 2023-04-25 00:08:04 (running for 00:00:21.64)
Memory usage on this node: 44.7/251.5 GiB 
Using FIFO scheduling algorithm.
Resources requested: 5.0/64 CPUs, 1.0/1 GPUs, 0.0/158.42 GiB heap, 0.0/71.88 GiB objects (0.0/1.0 accelerator_type:G)
Result logdir: /home/qhhu/ray_results/HydroTrainer_2023-04-25_00-07-42
Number of trials: 1/1 (1 RUNNING)
+--------------------------+----------+----------------------+--------+------------------+---------+-----------+--------------+
| Trial name               | status   | loc                  |   iter |   total time (s) |    loss |   val_acc |   _timestamp |
|--------------------------+----------+----------------------+--------+------------------+---------+-----------+--------------|
| HydroTrainer_24a0f_00000 | RUNNING  | 10.100.79.96:2591256 |      2 |          17.4336 | 1.62664 |    0.3947 |   1682352484 |
+--------------------------+----------+----------------------+--------+------------------+---------+-----------+--------------+
```
During workload execution, the reduction in memory usage resulting from scaling down the model can be observed through nvidia-smi (with varying memory footprints on different GPUs).

If you meet any error, please try to comment out the completed points in `POINTS_TO_EVALUATE` of `2_scaling_fidelity.py` and retry the experiment.


`Visualize the Result`. 

+ Open `plot.ipynb` file

+ Run the third cells (`2. Scaling Fidelity`) inside the notebook. 

+ From the Figure, where different hyperparameter configurations are denoted as A∼H, we can see the ranking of hyperparameters transfers well across different scaling ratios. It demonstrates that Hydro can well tune hypermeter on a smaller surrogate model.


### **D. Table 2: End-to-End Hyperparameter Tuning.**
**(Evaluation Time Estimation: 2 hours)** 


Hyperparameter tuning is a time-consuming process that requires the completion of the entire training process for trials. In order to complete the tuning in a reasonable time, we have scaled down the number of trials and the number of epochs. It can well present the end-to-end hyperparameter tuning performance of Hydro.

First, we execute the baseline Ray through one line:
```bash
python 3_example_ray.py
```

The output of this script looks like this:
```
== Status ==
Current time: 2023-04-25 09:46:21 (running for 01:06:05.40)
Memory usage on this node: 96.5/251.5 GiB 
Using FIFO scheduling algorithm.
Resources requested: 0/64 CPUs, 0/4 GPUs, 0.0/159.49 GiB heap, 0.0/72.34 GiB objects (0.0/1.0 accelerator_type:G)
Current best trial: befae_00002 with val_acc=0.9161 and parameters={'train_loop_config': {'lr': 0.1102, 'momentum': 0.584, 'batch_size': 128, 'gamma': 0.14, 'dataset': 'cifar10', 'seed': 10}}
Result logdir: /home/qhhu/ray_results/TorchTrainer_2023-04-25_08-40-15
Number of trials: 50/50 (50 TERMINATED)
+--------------------------+------------+----------------------+------------------------+------------------------+------------------------+------------------------+--------+------------------+----------+-----------+--------------+
| Trial name               | status     | loc                  |   train_loop_config/ba |   train_loop_config/ga |   train_loop_config/lr |   train_loop_config/mo |   iter |   total time (s) |     loss |   val_acc |   _timestamp |
|                          |            |                      |               tch_size |                    mma |                        |                 mentum |        |                  |          |           |              |
|--------------------------+------------+----------------------+------------------------+------------------------+------------------------+------------------------+--------+------------------+----------+-----------+--------------|
| TorchTrainer_befae_00000 | TERMINATED | 10.100.79.96:3752353 |                    128 |                   0.4  |                 0.1217 |                  0.51  |     50 |          439.46  | 0.381101 |    0.8899 |   1682383658 |
| TorchTrainer_befae_00001 | TERMINATED | 10.100.79.96:3752641 |                    256 |                   0.19 |                 0.2126 |                  0.791 |     50 |          283.339 | 0.338873 |    0.9088 |   1682383505 |
| TorchTrainer_befae_00002 | TERMINATED | 10.100.79.96:3752642 |                    128 |                   0.14 |                 0.1102 |                  0.584 |     50 |          401.142 | 0.319458 |    0.9161 |   1682383623 |
| TorchTrainer_befae_00003 | TERMINATED | 10.100.79.96:3752643 |                    512 |                   0.8  |                 0.0544 |                  0.907 |     50 |          261.278 | 0.434503 |    0.8686 |   1682383483 |
| TorchTrainer_befae_00004 | TERMINATED | 10.100.79.96:3948466 |                    128 |                   0.27 |                 0.0006 |                  0.515 |     50 |          416.374 | 0.687269 |    0.7584 |   1682383905 |
| TorchTrainer_befae_00005 | TERMINATED | 10.100.79.96:3960820 |                    256 |                   0.38 |                 0.4689 |                  0.857 |     50 |          273.111 | 0.523968 |    0.8298 |   1682383782 |
| TorchTrainer_befae_00006 | TERMINATED | 10.100.79.96:4039350 |                    512 |                   0.4  |                 0.1303 |                  0.791 |     50 |          264.846 | 0.398156 |    0.8882 |   1682383893 |
| TorchTrainer_befae_00007 | TERMINATED | 10.100.79.96:4063856 |                    128 |                   0.07 |                 0.0054 |                  0.808 |     50 |          393.092 | 0.353571 |    0.8934 |   1682384055 |
| TorchTrainer_befae_00008 | TERMINATED | 10.100.79.96:4157411 |                    512 |                   0.47 |                 0.0002 |                  0.882 |     50 |          253.747 | 0.922589 |    0.6684 |   1682384040 |
| TorchTrainer_befae_00009 | TERMINATED | 10.100.79.96:47576   |                    512 |                   0.28 |                 0.4311 |                  0.659 |     50 |          266.564 | 0.386158 |    0.8957 |   1682384164 |
| TorchTrainer_befae_00010 | TERMINATED | 10.100.79.96:54750   |                    512 |                   0.4  |                 0.0003 |                  0.914 |     50 |          258.722 | 0.760819 |    0.7343 |   1682384167 |
| TorchTrainer_befae_00011 | TERMINATED | 10.100.79.96:170932  |                    128 |                   0.19 |                 0.0003 |                  0.865 |     50 |          403.627 | 0.583267 |    0.7999 |   1682384448 |
| TorchTrainer_befae_00012 | TERMINATED | 10.100.79.96:181051  |                    128 |                   0.54 |                 0.2675 |                  0.675 |     50 |          408.078 | 0.53149  |    0.8261 |   1682384468 |
| TorchTrainer_befae_00013 | TERMINATED | 10.100.79.96:265008  |                    256 |                   0.16 |                 0.3087 |                  0.991 |     50 |          278.226 | 0.74957  |    0.7359 |   1682384447 |
| TorchTrainer_befae_00014 | TERMINATED | 10.100.79.96:267724  |                    512 |                   0.76 |                 0.0037 |                  0.547 |     50 |          251.926 | 0.573533 |    0.8096 |   1682384425 |
| TorchTrainer_befae_00015 | TERMINATED | 10.100.79.96:460998  |                    512 |                   0.89 |                 0.0433 |                  0.797 |     50 |          259.426 | 0.401074 |    0.8838 |   1682384688 |
| TorchTrainer_befae_00016 | TERMINATED | 10.100.79.96:474876  |                    512 |                   0.16 |                 0.0067 |                  0.912 |     50 |          255.586 | 0.404216 |    0.8817 |   1682384706 |
| TorchTrainer_befae_00017 | TERMINATED | 10.100.79.96:476676  |                    128 |                   0.54 |                 0.0039 |                  0.818 |     50 |          383.35  | 0.428186 |    0.8828 |   1682384836 |
| TorchTrainer_befae_00018 | TERMINATED | 10.100.79.96:488705  |                    256 |                   0.28 |                 0.0001 |                  0.678 |     50 |          281.235 | 1.28622  |    0.5325 |   1682384753 |
| TorchTrainer_befae_00019 | TERMINATED | 10.100.79.96:676335  |                    256 |                   0.35 |                 0.0021 |                  0.886 |     50 |          276.65  | 0.454192 |    0.8628 |   1682384968 |
| TorchTrainer_befae_00020 | TERMINATED | 10.100.79.96:689604  |                    256 |                   0.05 |                 0.0008 |                  0.557 |     50 |          264.31  | 0.825498 |    0.7049 |   1682384975 |
| TorchTrainer_befae_00021 | TERMINATED | 10.100.79.96:728441  |                    128 |                   0.32 |                 0.3309 |                  0.881 |     50 |          407.548 | 0.578922 |    0.8135 |   1682385166 |
| TorchTrainer_befae_00022 | TERMINATED | 10.100.79.96:788109  |                    128 |                   0.54 |                 0.0663 |                  0.91  |     50 |          380.398 | 0.450211 |    0.8529 |   1682385220 |
| TorchTrainer_befae_00023 | TERMINATED | 10.100.79.96:890549  |                    512 |                   0.04 |                 0.0011 |                  0.65  |     50 |          259.285 | 0.868014 |    0.6892 |   1682385233 |
| TorchTrainer_befae_00024 | TERMINATED | 10.100.79.96:892834  |                    512 |                   0.68 |                 0.0021 |                  0.5   |     50 |          251.103 | 0.674722 |    0.7664 |   1682385230 |
| TorchTrainer_befae_00025 | TERMINATED | 10.100.79.96:1041390 |                    256 |                   0.31 |                 0.0004 |                  0.694 |     50 |          284.924 | 0.843443 |    0.6965 |   1682385455 |
| TorchTrainer_befae_00026 | TERMINATED | 10.100.79.96:1086703 |                    256 |                   0.31 |                 0.0096 |                  0.988 |     50 |          274.245 | 0.334855 |    0.907  |   1682385498 |
| TorchTrainer_befae_00027 | TERMINATED | 10.100.79.96:1093892 |                    128 |                   0.77 |                 0.2199 |                  0.732 |     50 |          385.804 | 0.69946  |    0.7678 |   1682385620 |
| TorchTrainer_befae_00028 | TERMINATED | 10.100.79.96:1096484 |                    512 |                   0.11 |                 0.0002 |                  0.732 |     50 |          258.094 | 1.28396  |    0.5339 |   1682385495 |
| TorchTrainer_befae_00029 | TERMINATED | 10.100.79.96:1280305 |                    256 |                   0.32 |                 0.015  |                  0.86  |     50 |          262.082 | 0.404869 |    0.8983 |   1682385721 |
| TorchTrainer_befae_00030 | TERMINATED | 10.100.79.96:1312001 |                    256 |                   0.04 |                 0.0179 |                  0.65  |     50 |          269.508 | 0.366861 |    0.8865 |   1682385769 |
| TorchTrainer_befae_00031 | TERMINATED | 10.100.79.96:1314346 |                    512 |                   0.31 |                 0.024  |                  0.603 |     50 |          252.88  | 0.439658 |    0.8755 |   1682385756 |
| TorchTrainer_befae_00032 | TERMINATED | 10.100.79.96:1415510 |                    512 |                   0.23 |                 0.0003 |                  0.589 |     50 |          248.932 | 1.25369  |    0.5459 |   1682385873 |
| TorchTrainer_befae_00033 | TERMINATED | 10.100.79.96:1508219 |                    512 |                   0.11 |                 0.002  |                  0.501 |     50 |          263.823 | 0.788178 |    0.7228 |   1682385989 |
| TorchTrainer_befae_00034 | TERMINATED | 10.100.79.96:1539347 |                    256 |                   0.87 |                 0.0952 |                  0.734 |     50 |          278.516 | 0.44667  |    0.8563 |   1682386039 |
| TorchTrainer_befae_00035 | TERMINATED | 10.100.79.96:1550289 |                    512 |                   0.05 |                 0.1827 |                  0.846 |     50 |          247.249 | 0.365865 |    0.8876 |   1682386021 |
| TorchTrainer_befae_00036 | TERMINATED | 10.100.79.96:1646877 |                    128 |                   0.82 |                 0.001  |                  0.771 |     50 |          390.512 | 0.516335 |    0.8416 |   1682386268 |
| TorchTrainer_befae_00037 | TERMINATED | 10.100.79.96:1744988 |                    128 |                   0.51 |                 0.039  |                  0.672 |     50 |          380.994 | 0.355041 |    0.9004 |   1682386374 |
| TorchTrainer_befae_00038 | TERMINATED | 10.100.79.96:1767826 |                    256 |                   0.32 |                 0.0136 |                  0.507 |     50 |          273.866 | 0.429537 |    0.878  |   1682386298 |
| TorchTrainer_befae_00039 | TERMINATED | 10.100.79.96:1781849 |                    256 |                   0.74 |                 0.0204 |                  0.584 |     50 |          279.689 | 0.466781 |    0.8767 |   1682386323 |
| TorchTrainer_befae_00040 | TERMINATED | 10.100.79.96:1952998 |                    512 |                   0.18 |                 0.1778 |                  0.594 |     50 |          246.258 | 0.420455 |    0.8789 |   1682386519 |
| TorchTrainer_befae_00041 | TERMINATED | 10.100.79.96:1977733 |                    512 |                   0.46 |                 0.0045 |                  0.615 |     50 |          257.028 | 0.533169 |    0.8245 |   1682386561 |
| TorchTrainer_befae_00042 | TERMINATED | 10.100.79.96:1997194 |                    128 |                   0.87 |                 0.1889 |                  0.545 |     50 |          408.723 | 0.507385 |    0.829  |   1682386736 |
| TorchTrainer_befae_00043 | TERMINATED | 10.100.79.96:2035775 |                    256 |                   0.4  |                 0.0011 |                  0.931 |     50 |          261.505 | 0.47571  |    0.8617 |   1682386640 |
| TorchTrainer_befae_00044 | TERMINATED | 10.100.79.96:2160156 |                    256 |                   0.41 |                 0.0015 |                  0.519 |     50 |          263.424 | 0.621918 |    0.7832 |   1682386786 |
| TorchTrainer_befae_00045 | TERMINATED | 10.100.79.96:2194431 |                    256 |                   0.33 |                 0.0954 |                  0.778 |     50 |          274.282 | 0.393707 |    0.8935 |   1682386839 |
| TorchTrainer_befae_00046 | TERMINATED | 10.100.79.96:2259651 |                    128 |                   0.77 |                 0.0001 |                  0.967 |     50 |          382.196 | 0.512903 |    0.8298 |   1682387027 |
| TorchTrainer_befae_00047 | TERMINATED | 10.100.79.96:2332967 |                    128 |                   0.5  |                 0.0259 |                  0.89  |     50 |          392.918 | 0.367887 |    0.8898 |   1682387134 |
| TorchTrainer_befae_00048 | TERMINATED | 10.100.79.96:2369572 |                    128 |                   0.42 |                 0.3071 |                  0.981 |     50 |          389.501 | 1.99141  |    0.2726 |   1682387181 |
| TorchTrainer_befae_00049 | TERMINATED | 10.100.79.96:2405645 |                    256 |                   0.81 |                 0.0113 |                  0.742 |     50 |          266.701 | 0.496088 |    0.8686 |   1682387110 |
+--------------------------+------------+----------------------+------------------------+------------------------+------------------------+------------------------+--------+------------------+----------+-----------+--------------+
```

You can check the `Current best trial` shown in the log after all trial finished, and the total tuning time is record in `Current time: (running for xxx)`.

> If you find Ray process is not killed after the script is finished (e.g., GPU memory is still occupied), you can kill it manually by: `ray stop --force`.

Similarly, we execute the Hydro through one line:
```bash
python 3_example_hydro.py
```

The final output of this script looks like this:
```
== Status ==
Current time: 2023-04-25 08:20:42 (running for 00:23:42.54)
Memory usage on this node: 25.5/251.5 GiB 
Using FIFO scheduling algorithm.
Resources requested: 0/64 CPUs, 0/4 GPUs, 0.0/157.96 GiB heap, 0.0/71.69 GiB objects (0.0/1.0 accelerator_type:G)
Current best trial: b38f6_T0001(target trial) with val_acc=0.9162 and parameters={'lr': 0.1102, 'momentum': 0.584, 'batch_size': 128, 'gamma': 0.14, 'dataset': 'cifar10', 'seed': 10, 'FUSION_N': 0, 'SCALING_N': 0}
Result logdir: /home/qhhu/ray_results
Number of trials: 8/50 (8 TERMINATED)
+--------------------------+------------+----------------------+----------+------+----------------------+----------------------+----------------------+--------+------------------+--------------+---------------------+-----------------------+
| Trial name               | status     | loc                  | hydro    |   bs | gamma                | lr                   | momentum             |   iter |   total time (s) |   _timestamp |   _time_this_iter_s |   _training_iteration |
|--------------------------+------------+----------------------+----------+------+----------------------+----------------------+----------------------+--------+------------------+--------------+---------------------+-----------------------|
| HydroTrainer_b38f6_T0001 | TERMINATED | 10.100.79.96:3657182 | Target   |  128 | 0.14                 | 0.1102               | 0.584                |     50 |          384.496 |   1682382041 |             7.71278 |                    50 |
| HydroTrainer_b38f6_T0000 | TERMINATED | 10.100.79.96:3479472 | Target   |  512 | 0.05                 | 0.1827               | 0.846                |     50 |          279.453 |   1682381326 |             5.47435 |                    50 |
| HydroTrainer_b38f6_F0000 | TERMINATED | 10.100.79.96:3223763 | F=9, S=8 |  256 | [0.74, 0.38, 0._df80 | [0.0204, 0.4689_6d40 | [0.584, 0.857, _dec0 |     50 |          427.166 |   1682381050 |             8.64703 |                    50 |
| HydroTrainer_b38f6_F0001 | TERMINATED | 10.100.79.96:3223967 | F=8, S=8 |  256 | [0.32, 0.31, 0._fb80 | [0.013600000000_23c0 | [0.507, 0.69400_6500 |     50 |          415.149 |   1682381041 |             9.09435 |                    50 |
| HydroTrainer_b38f6_F0002 | TERMINATED | 10.100.79.96:3223968 | F=9, S=8 |  512 | [0.46, 0.28, 0._2b00 | [0.004500000000_f080 | [0.615, 0.659, _2180 |     50 |          382.011 |   1682381008 |             7.61978 |                    50 |
| HydroTrainer_b38f6_F0003 | TERMINATED | 10.100.79.96:3223969 | F=8, S=8 |  512 | [0.04, 0.4, 0.0_eb40 | [0.0011, 0.1303_eec0 | [0.65, 0.791, 0_fe00 |     50 |          357.47  |   1682380984 |             6.90358 |                    50 |
| HydroTrainer_b38f6_F0004 | TERMINATED | 10.100.79.96:3451196 | F=8, S=8 |  128 | [0.14, 0.54, 0._8280 | [0.1102, 0.2675_a200 | [0.584, 0.675, _8400 |     50 |          773.026 |   1682381761 |            15.4453  |                    50 |
| HydroTrainer_b38f6_F0005 | TERMINATED | 10.100.79.96:3464377 | F=8, S=8 |  128 | [0.42, 0.54, 0._a140 | [0.307100000000_e880 | [0.981, 0.81800_8f40 |     50 |          737.031 |   1682381749 |            14.6641  |                    50 |
+--------------------------+------------+----------------------+----------+------+----------------------+----------------------+----------------------+--------+------------------+--------------+---------------------+-----------------------+
```

You can check the `Current best trial` and `Current time: (running for xxx)` shown in the log after all trial finished, and compare it with the result of Ray.

More details please refer to `ray.csv` and `hydro_parsed.csv` in the `results` folder.


## Acknowledgements

Hydro is built upon many fabulous open-source repositories, including

[ray](https://github.com/ray-project/ray) | [mup](https://github.com/microsoft/mup) | [hfta](https://github.com/UofT-EcoSystem/hfta) | [pytorch](https://github.com/pytorch/pytorch) | [
transformers](https://github.com/huggingface/transformers)




