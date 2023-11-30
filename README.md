<p align="center">
    <img src="asset/dordis.png" height=400>
</p>

<p align="center">
    <a href="https://arxiv.org/abs/2209.12528"><img src="https://img.shields.io/badge/arxiv-2209.12528-silver" alt="Paper"></a>
    <a href=""><img src="https://img.shields.io/badge/Pub-EuroSys'24-olive" alt="Pub"></a>
    <a href="https://github.com/SamuelGong/Dordis"><img src="https://img.shields.io/badge/-github-teal?logo=github" alt="github"></a>
    <a href="https://github.com/SamuelGong/Dordis/blob/main/LICENSE"><img src="https://img.shields.io/github/license/SamuelGong/Dordis?color=yellow" alt="License"></a>
    <img src="https://badges.toozhao.com/badges/01HCERSP3HP3DQDCZGBGN0BFYX/green.svg" alt="Count"/>
</p>

<h1 align="center">Dordis: Efficient Federated Learning with Dropout-Resilient Differential Privacy (EuroSys 2024)</h1>

This repository contains the evaluation artifacts of our paper titled 
*Dordis: Efficient Federated Learning with Dropout-Resilient Differential Privacy*, 
which will be presented at EuroSys'24 conference.
You can find the preprint of the paper [here](https://arxiv.org/pdf/2209.12528.pdf).

[Zhifeng Jiang](http://home.cse.ust.hk/~zjiangaj/), [Wei Wang](https://home.cse.ust.hk/~weiwa/), [Ruichuan Chen](https://www.ruichuan.org/)

**Keywords**: Federated Learning, Distributed Differential Privacy, Client Dropout, Secure Aggregation, Pipeline

<details> <summary><b>Abstract (Tab here to expand)</b></summary>

Federated learning (FL) is increasingly deployed among multiple clients to train a shared model over decentralized data. To address privacy concerns, FL systems need to safeguard the clients' data from disclosure during training and control data leakage through trained models when exposed to untrusted domains. Distributed differential privacy (DP) offers an appealing solution in this regard as it achieves a balanced tradeoff between privacy and utility without a trusted server. However, existing distributed DP mechanisms are impractical in the presence of *client dropout*, resulting in poor privacy guarantees or degraded training accuracy. In addition, these mechanisms suffer from severe efficiency issues.

We present Dordis, a distributed differentially private FL framework that is highly efficient and resilient to client dropout. Specifically, we develop a novel `add-then-remove` scheme that enforces a required noise level precisely in each training round, even if some sampled clients drop out. This ensures that the privacy budget is utilized prudently, despite unpredictable client dynamics. To boost performance, Dordis operates as a distributed parallel architecture via encapsulating the communication and computation operations into stages. It automatically divides the global model aggregation into several chunk-aggregation tasks and pipelines them for optimal speedup. Large-scale deployment evaluations demonstrate that Dordis efficiently handles client dropout in various realistic FL scenarios, achieving the optimal privacy-utility tradeoff and accelerating training by up to 2.4× compared to existing solutions.

</details>


## Table of Contents
1. [Overview](#1-overview)
2. [Prerequisites](#2-prerequisites)
   * Necessary dependencies installation before anything begins.
3. [Simulation](#3-simulation)  
   * Learn how to run experiment in simulation mode.  
4. [Cluster Deployment](#4-cluster-deployment)
   * Learn how to run experiments in a distributed manner.
5. [Reproducing Experimental Results](#5-reproducing-experimental-results)
   * Learn how to reproduce paper experiments.
6. [Repo Structure](#6-repo-structure)
   * What are contained in the project root folder.
7. [Support](#7-support)
8. [License](#8-license)
9. [Citation](#9-citation)
  
## 1. Overview

The system supports two modes of operation:

1. **Simulation**:
This mode allows you to run experiments on a local machine or GPU server.
It is primarily used for validating functionality, privacy, or utility.
2. **Cluster Deployment**:
This mode enables you to run experiments on an AWS EC2 cluster 
funded by your own account. Alternatively, you can also run experiments on 
an existing cluster of Ubuntu nodes (currently undocumented). 
Cluster Deployment Mode is typically used for evaluating runtime performance.

## 2. Prerequisites

To work with the project, you need to have a Python 3 Anaconda environment set up 
in your host machine (Ubuntu system assumed) with specific dependencies installed. 
To simplify the setup process, we provide a shortcut:

```bash
# assumes you are working from the project folder
cd exploration/dev
bash standalone_install.sh
conda activate dordis
```

**Note**

1. Most the dependencies will be installed in a newly created environment called 
**dordis**, minimizing interference with your original system setup.
2. However, please note that the `redis-server` application needs to be installed 
at the **system** level with **sudo** previlige, as mentioned in the Line 49-52 of 
the `standalone_install.sh` script. If you do not have sudo privileges, you can 
follow the instructions provided [here](https://techmonger.github.io/40/redis-without-root/)
to install Redis without root access. In that case, you should comment out these 
lines before executing the command `bash standalone_install.sh`.

## 3. Simulation

### 3.1 Preparing Working Directory

Start by choosing a name for the working directory. 
For example, let's use `ae-simulator` in the following instructions.

```bash
# assumes you are working from the project folder
cd exploration
cp -r simulation_folder_template ae-simulator
cd ae-simulator
```

### 3.2 Run Experiments

To run an experiment with a specific configuration file in the background,
follow these steps:

```bash
bash simulator_run.sh start_a_task [target folder]/[target configuration file]
```

The primarily logged information will be output to the following file:

```
[target folder]/[timestamp]/dordis-coordinator/log.txt
```

**Note**

1. When you execute the above command, the command line will prompt you with `[timestamp]`, 
which represents the relevant timestamp and output folder.
2. You can use the simulator_run.sh script for task-related control.
You don't need to remember the commands because the prompt will inform you
whenever you start a task. Here are a few examples:
    ```bash
    # To kill a task halfway
    bash simulator_run.sh kill_a_task [target folder]/[timestamp]
    # To analyze the output and generate insightful figures/tables
    bash simulator_run.sh analyze_a_task [target folder]/[timestamp]
    ```
   
### 3.3 Batch Tasks to Run

The simulator also supports batching tasks to run. You can specify the tasks 
to run in the background by writing them in the `batch_plan.txt` file, 
as shown below:

```
[target folder]/[target configuration file]
[target folder]/[target configuration file]
[target folder]/[target configuration file]
```

To sequentially run the tasks in a batch, execute the following command:

```
bash batch_run.sh batch_plan.txt
```

The execution log will be available at `batch_log.txt`.

**Note**

1. To stop the batching logic halfway and prevent it from issuing any new tasks,
you can use the command `kill -9 [pid]`. The `[pid]` value can be found at the 
beginning of the file `batch_log.txt`.
2. If you want to stop a currently running task halfway, you can kill it using the
command `bash simulator_run.sh kill_a_task [...]`, as explained in the previous
subsection. The information needed to kill the job will also be available in the log.

## 4. Cluster Deployment

You can initiate the cluster deployment process either from your local host
machine (ensuring a stable network connection) or from a dedicated remote node
specifically designed for **coordination** purposes (we thus call it the coordinator node).
It is important to note that the remote node does not necessarily need to be a
powerful machine.

### 4.1 Install and Configure AWS CLI

Before proceeding, please ensure that you have an **AWS account**.
Additionally, on the coordinator node, it is essential to have the latest version of
**aws-cli** installed and properly configured with the necessary credentials.
This configuration will allow us to conveniently manage all the nodes in the cluster
remotely using command-line tools.

**Reference**

1. [Install AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html).
    * Example command for installing into Linux x86 (64-bit):
    ```bash
    # You can work from any directory, e.g., at your home directory
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
    sudo apt install unzip
    unzip awscliv2.zip
    sudo ./aws/install
    ```
2. [Configure AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-quickstart.html).
    * Example command for configuring one's AWS CLI:
    ```bash
    # You can work from any directory, e.g., at your home directory
    aws configure
    ```
    You will be prompted to enter your AWS Access Key ID, AWS Secret Access Key,
Default region name, and Default output format. Provide the required information as prompted.

### 4.2 Configure and Allocate a Cluster

To begin, let's choose a name for the working directory. For example, we can use `ae-cluster` 
as the name throughout this section (please note that this name should not be confused with
the above-mentioned simulator folder).

```bash
# assumes you are working from the project folder
cd exploration
cp -r cluster_folder_template ae-cluster
cd ae-cluster

# make some modifications that suit your need/budget
# 1. EC2-related
# relevant key:
#   BlockDeviceMappings/Ebs/VolumeSize: how large is the storage of each node
#   KeyName: the path to the key file (relative to ~/.ssh/) you plan to use to 
#            log into each node of the cluster from your coordinator node
vim ../dev/ec2_node_template.yml
# 2. cluster-related
# relevant key:
#     type and region for the server each client, and
#     count and bandwidth for each client
#     (the provided images are specifically for Ubuntu 18.04)
vim ec2_cluster_config.yml
# 3. minor
# relevant value: LOCAL_PRIVATE_KEY
#     set the value the same as KeyName mentioned above
vim manage_cluster.sh
```

Then one can launch a cluster from scratch:

```bash
bash manage_cluster.sh launch
```

**Note**

1. The `manage_cluster.sh` script provides a range of cluster-related controls for
your convenience. Here are some examples of how to use it:

    ```bash
    # start an already launch cluster
    bash manage_cluster.sh start
    # stop a running cluster
    bash manage_cluster.sh stop
    # restart a running cluster
    bash manage_cluster.sh reboot
    # terminate an already launch cluster
    bash manage_cluster.sh terminate
    # show the public IP addresses for each node
    bash manage_cluster.sh show
    # scale up/down an existing cluster 
    # (with the ec2_cluster_config.yml modified correspondingly)
    bash manage_cluster.sh scale
    # configure the bandwidth for each client node
    # (with the ec2_cluster_config.yml modified correspondingly)
    bash manage_cluster.sh limit_bandwidth
    # reset the bandwidth for each node
    bash manage_cluster.sh free_bandwidth
    ```

### 4.3 Setting Up the Cluster

Execute the following commands:

```bash
bash setup.sh install
bash setup.sh deploy_cluster
```

**Note**
1. Additionally, the `setup.sh` script offers a wide range of app-related controls.
Here are a couple of examples:
    ```bash
    # update all running nodes' Github repo
    bash setup.sh update
    # add pip package to the used conda environment for all running nodes
    bash setup.sh add_pip_dependency [package name (w/ or w/o =version)]
    # add apt package for all running nodes (recall that we are using Ubuntu)
    bash setup.sh add_apt_dependency [package name (w/ or w/o =version)]
    ```

### 4.4 Run Experiments

Like what in simulation, once you have a running cluster, you can start a task
with distributed deployment by running commands like:

```bash
bash cluster_run.sh start_a_task [target folder]/[target configuration file]
```

**Remark**
1. After you execute the above command, the command line will prompt you with `[timestamp]`, 
which represents the relevant timestamp and output folder.
2. To control the task, you can use the following commands with `cluster_run.sh`:
    ```bash
    # for killing the task halfway
    bash cluster_run.sh kill_a_task [target folder]/[timestamp]
    # for fetching logs from all running nodes to the coordinator
    # (i.e., the machine where you type this command)
    bash cluster_run.sh conclude_a_task [target folder]/[timestamp]
    # for analyzing the collected log to generate some insightful figures/tables
    # Do it only after the command ... conclude_a_task ... has been executed successfully
    bash cluster_run.sh analyze_a_task [target folder]/[timestamp]
    ```

### 4.5 Batch Tasks to Run

In addition to running tasks individually, the cluster mode also supports batch 
execution of tasks. To run a batch of tasks in the background, you need to specify
the tasks to be executed in the `batch_plan.txt` file using the following format:

```
[target folder]/[target configuration file]
[target folder]/[target configuration file]
[target folder]/[target configuration file]
```

Then you can sequentially execute them as a batch by running the following command:

```
bash batch_run.sh batch_plan.txt
```

The execution log will be generated and can be found at `batch_log.txt`.

**Note**

1. If you need to terminate the batch execution before completion, you can use the command
`kill -9 [pid]` to stop the batching logic. The process ID `pid` can be found at the
beginning of the log file.
2. If you want to stop a specific task that is currently running, you can use the command
`bash simulator_run.sh kill_a_task [...]` as explained in the subsection above. The
necessary information for killing a job, denoted by `[...]`, can also be found in the log file.

## 5. Reproducing Experimental Results

### 5.1 Major Claims

* Our noise enforcement scheme, *XNoise*, guarantees the consistent achievement of
the desired privacy level, even in the presence of client dropout, while maintaining
the model utility. This claim is supported by the simulation experiment (**E1**)
outlined in Section 6.2 of the paper, with the corresponding results presented in
Figure 8, Table 2, and Figure 9.
* The *XNoise* scheme introduces acceptable runtime overhead, with the network
overhead being scalable with respect to the model's expanding size. This can be
proven by the cluster experiment (**E2**) described in Section 6.3 of the paper,
whose results are reported in Figure 10 and Table 3.
* The pipeline-parallel aggregation design employed by Dordis significantly boosts
training speed, leading to a remarkable improvement of up to 2.4X in the training
round time. This finding is supported by the cluster experiment (**E3**) discussed
in Section 6.4, and the corresponding results are depicted in Figure 10.

### 5.2 Experiments

*Note: for certain conclusions that require extensive analysis, we have included
instructions for both full experiments (**Full**) and minimal executable 
examples (**Minimal**).*

#### E1 [*Effectiveness of Noise Enforcement*]

* **Preparation** Before proceeding with the simulation, please ensure that you have
followed the instructions provided in Section [3.1](#31-preparing-working-directory)
to prepare your working directory.

* **Execution and Expected Results**
  - Step 1: Reproducing Figure 8 can be accomplished by running the following commands.
This step should only take **a few seconds** to complete. After this step, you should be
able to replicate the exact images presented in Figure 8.
  
   ```bash
   # starting from the exploration/ae-simulator directory
   cd privacy
   bash run.sh
   # after which you should replicate exactly Figure 8
   ```

  - Step 2 **[Full]**: To reproduce Figure 9 and Table 2, execute the following commands for batch
processing. Please note that this step may require a significant amount of time to finish.
The duration can vary depending on the computational resources available.
For example, when we used a node with an Intel Xeon 16 Cores 32 Threads E5-2683V4 2.1GHz
processor and 8 NVIDIA GeForce RTX 2080 Ti GPUs exclusively for this simulation, 
it took us **approximately one to two months**.
Should you have either faster CPUs or faster GPUs,
the time cost should be reduced. If you have faster CPUs or GPUs,
the processing time should be reduced.
  
   ```bash
   # starting from exploration/ae-simulator
   bash batch_run.sh batch_plan.txt
   ```
  
  - Step 2 **[Minimal]**: We also provide a minimal version of the above step,
which only reproduces the sub-figure (a) of Figure 9 and the particular cell that
corresponds to FEMNIST with d=20% in Table 2. It should take
**one to two days** (or less should you have better GPUs or CPUs than ours,
see the above description).

  ```bash
  # starting from exploration/ae-simulator
  bash batch_run.sh batch_plan_minimal.txt
  ```

  - Step 3 **[Full]**: Once Step 2 [Full] completes successfully,
proceed with the following fast commands
to visualize the collected data. These commands should only take **a few seconds**
to execute. You should be able to replicate **similar** results as the ones presented by
Table 2 and Figure 9.

  ```bash
  # starting from exploration/ae-simulator
  cd utility
  bash run.sh
  # after which you should replicate results similar to Table 2
  
  cd ..
  bash batch_plot.sh
  cd experiments
  # after which you should replicate results similar to Figure 9
  ```
  
  - Step 3 **[Minimal]**: Once Step 2 [Minimal] finishes, proceed with the
following fast commands to visualize the collected data. These commands should
only take **a few seconds** to execute. You should be able to replicate **similar**
results as the ones presented by part of the Table 2 and Figure 9.

  ```bash
  # starting from exploration/ae-simulator
  cd utility_minimal
  bash run.sh
  # after which you should replicate one cell of Table 2 (FEMNIST, d=20%) with similar data
  
  cd ..
  bash batch_plot_minimal.sh
  cd experiments
  # after which you should replicate the FEMNIST part of Figure 9 with similar data
  ```

#### E2 [*Efficiency of Noise Enforcement*]

* **Preparation** Before proceeding with the experiment, please ensure that you have
followed the instructions provided in Section [4.1](#41-install-and-configure-aws-cli),
[4.2](#42-configure-and-allocate-a-cluster), and [4.3](#43-setting-up-the-cluster)
and enter the folder `exploration/ae-cluster`.

* **Execution and Expected Results**

  - Step 1: To reproduce Table 3, perform the following deterministic calculations
using the provided commands. This step should take **less than one second** to complete.
After this step, you are expected to see the **exact** Table 3 in the command line.
  
  ```bash
  # starting from the exploration/ae-cluster directory
  cd network
  python main.py
  ```

  - Step 2: Start the allocated AWS EC2 cluster and set the bandwidth limit using
the following commands. This step will take **approximately two to three minutes** to
complete:

  ```bash
  # starting from the exploration/ae-cluster directory

  # if the cluster is not started
  bash manage_cluster.sh start
  # if the cluster is just started, wait for about a minute before executing it
  bash manage_cluster.sh limit_bandwidth
  ```

  - Step 3 **[Full]**: To reproduce a part of Figure 10, execute the following
commands. This step may take **approximately four to five days** to complete.
On completion, the cluster should be automatically shut down to save your expense.
  
  ```bash
  # starting from the exploration/ae-cluster directory
  bash batch_run.sh batch_plan_no_pipeline.txt
  ```

  - Step 3 **[Minimal]**: To reproduce a minimal part of the expected result,
execute the following commands. This step replicates the `plain` part of Figure 10(e),
and the experiment may take **around three hours** to complete.
  
  ```bash
  # starting from the exploration/ae-cluster directory
  bash batch_run.sh batch_plan_no_pipeline_minimal.txt
  ```

  - Step 4 **[Full]**: Once Step 3 [Full] completes successfully, proceed with the following
fast commands to visualize the collected data. These commands should only
take **a few seconds** to execute. You should able to replicate
the `plain` part (meaning no pipeline acceleration) of Figure 10 with **similar** figures.

  ```bash
  # starting from the exploration/ae-cluster directory
  bash batch_plot_no_pipeline.sh
  cd no-pipeline-time
  # after which you should replicate results similar to the 'plain' part of Figure 10
  ```
  
  - Step 4 **[Minimal]**: Once Step 3 [Minimal] completes successfully, proceed with the following
fast commands to visualize the collected data. These commands should only
take **a few seconds** to execute. You should able to replicate
the `plain` part of Figure 10(e) with **similar** figures.

  ```bash
  # starting from the exploration/ae-cluster directory
  bash batch_plot_no_pipeline_minimal.sh
  cd no-pipeline-time
  # after which you should replicate results similar to the plain part of Figure 10(e)
  ```

#### E3 [*Efficiency of Pipeline Acceleration*]

* **Preparation**
Identical to **E2**, before proceeding with the experiment, please ensure that you have
entered the folder `exploration/ae-cluster` and started the cluster.

* **Execution and Expected Results**

  - Step 1 **[Full]**: To reproduce the remaining part of Figure 10, execute the following
commands. This step may take approximately three to four days to complete.
The cluster will be automatically shut down after the jobs are completed.
  
  ```bash
  # starting from the exploration/ae-cluster directory
  bash batch_run.sh batch_plan_pipeline.txt
  ```
  
  - Step 1 **[Minimal]**: To reproduce the `pipe` part of Figure 10(e),
execute the following commands.

  ```bash
  # starting from the exploration/ae-cluster directory
  bash batch_run.sh batch_plan_pipeline_minimal.txt
  ```

  - Step 2 **[Full]**: Process the collected data using the following commands.
This step will only take **several minutes** to complete. After completing this
step, you should generate the entire shape of Figure 10 with **similar details**.

  ```bash
  bash batch_plot_pipeline.sh
  cd pipeline-time
  # after which you should replicate results similar to the ones in Figure 10
  ```
  
  - Step 2 **[Minimal]**: Process the collected data using the following commands.
This step will only take **several seconds** to complete. After completing this
step, you should generate the entire shape of Figure 10(e) with **similar details**.

  ```bash
  bash batch_plot_pipeline_minimal.sh
  cd pipeline-time
  # after which you should replicate results similar to the ones in Figure 10(e)
  ```

## 6. Repo Structure

```
Repo Root
|---- dordis                           # Core implementation
|---- exploration                      # Evaluation
    |---- cluster_folder_template      # Necessities for cluster deployment
    |---- simulation_folder_template   # Necessities for single-node simulation
    |---- dev                          # Backend for experiment manager
    |---- analysis                     # Backend for resulting data processing
```

## 7. Support
If you need any help, please submit a Github issue, or contact Zhifeng Jiang via zjiangaj@cse.ust.hk.

## 8. License

The code included in this project is licensed under the [Apache 2.0 license](LICENSE). If you wish to use the codes and models included in this project for commercial purposes, please sign this [document](https://docs.google.com/forms/d/e/1FAIpQLSeJRoGkDtmN5LP5MS_xQFa2nerlcQT8gTEnEdiKmUVu3s2DWA/viewform?usp=sf_link) to obtain authorization.

## 9. Citation

If you find this repository useful, please consider giving ⭐ and citing our paper (preprint available [here](https://arxiv.org/pdf/2209.12528.pdf)):

```bibtex
@inproceedings{jiang2024efficient,
  author={Jiang, Zhifeng and Wang, Wei and Ruichuan, Chen},
  title={Dordis: Efficient Federated Learning with Dropout-Resilient Differential Privacy},
  year={2024},
  booktitle={EuroSys},
}
```

## [Recent activity [![Time period](https://images.repography.com/44587806/SamuelGong/Dordis/recent-activity/8XPPH8JArUAvVdkGA8K5SCbumi1AfsBrGcyYp_yZFVY/6XDpH_F_-jcSljG9RSH5DA69zYgOAX9WlHd86A0iXcY_badge.svg)](https://repography.com)
[![Timeline graph](https://images.repography.com/44587806/SamuelGong/Dordis/recent-activity/8XPPH8JArUAvVdkGA8K5SCbumi1AfsBrGcyYp_yZFVY/6XDpH_F_-jcSljG9RSH5DA69zYgOAX9WlHd86A0iXcY_timeline.svg)](https://github.com/SamuelGong/Dordis/commits)
[![Issue status graph](https://images.repography.com/44587806/SamuelGong/Dordis/recent-activity/8XPPH8JArUAvVdkGA8K5SCbumi1AfsBrGcyYp_yZFVY/6XDpH_F_-jcSljG9RSH5DA69zYgOAX9WlHd86A0iXcY_issues.svg)](https://github.com/SamuelGong/Dordis/issues)
[![Pull request status graph](https://images.repography.com/44587806/SamuelGong/Dordis/recent-activity/8XPPH8JArUAvVdkGA8K5SCbumi1AfsBrGcyYp_yZFVY/6XDpH_F_-jcSljG9RSH5DA69zYgOAX9WlHd86A0iXcY_prs.svg)](https://github.com/SamuelGong/Dordis/pulls)
[![Activity map](https://images.repography.com/44587806/SamuelGong/Dordis/recent-activity/8XPPH8JArUAvVdkGA8K5SCbumi1AfsBrGcyYp_yZFVY/6XDpH_F_-jcSljG9RSH5DA69zYgOAX9WlHd86A0iXcY_map.svg)](https://github.com/SamuelGong/Dordis/commits)
