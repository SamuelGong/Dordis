# Dordis

This repository contains the evaluation artifacts of our EuroSys'24 paper *Efficient Federated Learning with Dropout-Resilient
  Differential Privacy*.

[(preprint available here)](https://arxiv.org/pdf/2209.12528.pdf).

## 1. Overview

The system supports two mode:

1. **Simulation**, where one can have his/her experiments running in a local machine/GPU server,
This mode is usually for validating *functionality*, *privacy*, or *utility*.
2. **Cluster Deployment**, where one can have his/her experiments running in an AWS EC2 cluster 
(funded by his/her account). One can also choose to run atop an existing cluster of Ubuntu nodes
(but this is not documented yet). 
This mode is usually for evaluating *runtime performance*.

The rest of the README is organized as follows:

### Table of Contents

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

## 2. Prerequisites

One should be able to directly work in a **Python 3** Anaconda environment with some dependencies installed.
For ease of use, we provide a shortcut for doing so:

```bash
# starting from [project folder]
cd exploration/dev
bash controller_install.sh
# please exit your current shell and log in again for conda to take effect
```

## 3. Simulation

### 3.1 Install Dependencies

Think of a name for the working directory, e.g., `ae-simulator` 
(this section uses this example name througout).

```bash
# starting from [project folder]
cd exploration
cp -r simulation_folder_template ae-simulator
cd ae-simulator
bash setup.sh install
```

**Remark**

1. All the dependencies will be installed to a newly added environment 
called `dordis` and thus will not mess with the original system.

### 3.2 Run Experiments

Suppose that an experiment is configured by 
`[target folder]/[target configuration file]` (relative path), 
then one should be able to run the following commands to start the task in the background:

```bash
# starting from [project folder]/exploration/ae-simulator
bash simulator_run.sh start_a_task [target folder]/[target configuration file]
```

Then the mainly logged information will be output to the following file, where 
`[some timestamp]` will be prompted as soon as you enter the above command:

```
exploration/ae-simulator/[target folder]/[some timestamp]/dordis-coordinator/log.txt
```

**Remarks**

1. One can use `simulator_run.sh` for task-related control (no need to remember, 
because the prompt will inform you of them whenever you start a task):
    ```bash
    # for killing the task halfway
    bash simulator_run.sh kill_a_task [target folder]/[some timestamp]
    # for analyzing the output to generate some insightful figures/tables
    bash simulator_run.sh analyze_a_task [target folder]/[some timestamp]
    ```
   
### 3.3 Batch Tasks to Run (Optional)

The simulator supports batching tasks to run. After writing what tasks to run 
in the background into `exploration/ae-simulator/batch_plan.txt` like:

```
some_folder_relative_path/configuration-a.yml
some_folder_relative_path/configuration-b.yml
some_folder_relative_path/configuration-c.yml
```

Then one can sequentially run the tasks in a batch via

```
# Starting at exploration/ae-simulator/
bash batch_run.sh
```

The execution log will be available at `exploration/ae-simulator/batch_log.txt`.

**Remark**

1. Simply use `kill -9 [pid]` for killing the batching logic halfway (so that it does not issue any new task). 
The `pid` is available at the beginning of the log.
2. After that, for the currently running task that one wants to stop halfway,
please kill it with `bash simulator_run.sh kill_a_task [...]`, as in the above subsection,
where `[...]` for job killing is also available at the log.

## 4. Cluster Deployment

One can start with his/her local machine (but it should have stable network connection), 
or a remote node which is dedicated to coordination purpose 
(thus it does not have to be a powerful machine).

### 4.1 Install and Configure AWS CLI

One should have an **AWS account**. Also, at the coordinator node, 
one should have installed the latest **aws-cli** with credentials well configured (so that
we can manipulate all the nodes in the cluster remotely via command line tools.).

**Reference**

1. [Install AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html).
    * Example command for installing into Linux x86 (64-bit):
    ```bash
    # done anywhere, e.g., at your home directory
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
    sudo apt install unzip
    unzip awscliv2.zip
    sudo ./aws/install
    ```
2. [Configure AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-quickstart.html).
    * Example command for configuring one's AWS CLI:
    ```bash
    # done anywhere, e.g., at your home directory
    aws configure
    ```
   where one sequentially input AWS Access Key ID, AWS Secret Access Key, Default region name and default output format.

### 4.2 Launch a Cluster

Think of a name for the working directory, e.g., `ae-cluster` 
(this section uses this example name througout; do NOT mess with the above simulator folder).

```bash
# starting from [project folder]
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
#     count and (just planning, not applied upon starting) bandwidth for each client
# p.s. the provided images are specifically of Ubuntu 18.04
vim ec2_cluster_config.yml
# edit the value for LOCAL_PRIVATE_KEY
# the same as KeyName mentioned above
vim manage_cluster.sh
```

Then one can launch a cluster from scratch:

```bash
# starting from [project folder]/exploration/ae-cluster
bash manage_cluster.sh launch
```

**Remark**

1. One can use `manage_cluster.sh` for a wide range of cluster-related control:

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

### 4.3 Configure the Cluster

```bash
# starting from [project folder]/exploration/ae-cluster
bash setup.sh install
bash setup.sh deploy_cluster
```

**Remark** 
1. Make sure that your `~/.ssh/` has the correct key file that you specified in `../dev/ec2_node_template.yml` 
as well as `manage_cluster.sh`.
2. One can use `setup.sh` for a wide-range of app-related control:
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
# starting from [project folder]/exploration/ae-simulator
bash cluster_run.sh start_a_task [target folder]/[target configuration file]
```

**Remark**
1. One can use `cluster_run.sh` for task-related control (no need to remember, 
because the prompt will inform you of them whenever you start a task):
    ```bash
    # for killing the task halfway
    bash cluster_run.sh kill_a_task [target folder]/[some timestamp]
    # for fetching logs from all running nodes to the coordinator
    # (i.e., the machine where you type this command)
    bash cluster_run.sh conclude_a_task [target folder]/[some timestamp]
    # for analyzing the collected log to generate some insightful figures/tables
    # Do it only after the command ... conclude_a_task ... has been executed successfully
    bash cluster_run.sh analyze_a_task [target folder]/[some timestamp]
    ```

### 4.5 Batch Tasks to Run (Optional)

Similarly, the cluster mode also supports batching tasks to run. After writing what tasks to run 
in the background into `exploration/ae-cluster/batch_plan.txt` like:

```
some_folder_relative_path/configuration-a.yml
some_folder_relative_path/configuration-b.yml
some_folder_relative_path/configuration-c.yml
```

Then one can sequentially run the tasks in a batch via

```
# Starting at exploration/ae-cluster/
bash batch_run.sh
```

The execution log will be available at `exploration/ae-cluster/batch_log.txt`.

**Remark**

1. Simply use `kill -9 [pid]` for killing the batching logic halfway (so that it does not issue any new task). 
The `pid` is available at the beginning of the log.
2. After that, for the currently running task that one wants to stop halfway,
please kill it with `bash simulator_run.sh kill_a_task [...]`, as in the above subsection,
where `[...]` for job killing is also available at the log.

## 5. Reproducing Experimental Results

### 5.1 Major Claims

* Our noise enforcement scheme, *XNoise*, guarantees the consistent achievement of the desired privacy level, even in the presence of client dropout, while maintaining the model utility. This claim is supported by the simulation experiment (**E1**) outlined in Section 6.2 of the paper, with the corresponding results presented in Figure 8, Table 2, and Figure 9.
* The *XNoise* scheme introduces acceptable runtime overhead, with the network overhead being scalable with respect to the model's expanding size. This can be proven by the cluster experiment (**E2**) described in Section 6.3 of the paper, whose results are reported in Figure 10 and Table 3.
* The pipeline-parallel aggregation design employed by Dordis significantly boosts training speed, leading to a remarkable improvement of up to 2.4X in the training round time. This finding is supported by the cluster experiment (**E3**) discussed in Section 6.4, and the corresponding results are depicted in Figure 10.

### 5.2 Experiments

#### E1 [*Effectiveness of Noise Enforcement*]
* **Preparation** Please make sure that you have followed the instructions in Section [Simulation](#3-simulation) and enter the folder `exploration/ae-simulator`.
* **Execution**
  - Step 1: Reproducing Figure 8 only requires to run the following commands, which only takes **several seconds** to finish:
  
   ```bash
   # for Figure 8
   cd privacy
   bash run.sh
   cd ..
  ```
  - Step 2: Reproducing Figure 9 and Table 2 requires to execute the following commands for batch processing, which may require an extensive amount of time to finish (e.g., when we used a node (Intel Xeon 16 Cores 32 Threads E5-2683V4 2.1GHz) with 8 NVIDIA GeForce RTX 2080 Ti GPUs exclusively for this simulation, it took us **one to two months**). Should you have either faster CPUs or faster GPUs, the time cost should be reduced. As the jobs are launched and processed in the background, you should refer to the generated file `batch_log.txt` to see the up-to-date status of the batch processing.
  
   ```bash
   bash batch_run.sh
   ```

  - Step 3: When the above step ends with success (i.e., the `batch_log.txt` ends with a line `Batch job ended`), further run the following fast commands (**several seconds**) to visualize the collected data.

   ```bash
   # for Table 2
   cd utility
   bash run.sh
   cd ..
   # for Figure 9
   cd experiments
   bash batch_plot.sh
   ```

* **Expected Results**
  - After Step 1, you are expected to replicate the **exact** images as the ones illustrated in Figure 8. 
  - After Step 3, you should be able to replicate **similar** results as the ones presented by Table 2 and Figure 9.

#### E2 [*Efficiency of Noise Enforcement*]

* **Preparation** Please make sure that you have followed the instructions in Section [Cluster Deployment](#4-cluster-deployment) and enter the folder `exploration/ae-cluster`.
* **Execution**
  - Step 1: Start the allocated AWS EC2 cluster and set the bandwidth limit using the following commands, which takes **two to three minutes** to finish:

  ```bash
  bash manage_cluster.sh start
  # after the above command returns and wait for approximately one more minute
  bash manage_cluster.sh limit_bandwidth
  ```

  - Step 2: Reproducing the related part of Figure 10 needs to run the following commands, which may takes **four to five** days to complete. Similar to simulation, a line `Batch jobs ended.` will be appended to the log file `batch_log.txt` when all the jobs finished. It is also noteworthy that the cluster will be automatically stopped after the completion of all jobs, and thus there is no need for waiting at the table.
  
  ```bash
  # for part of Figure 10
  bash batch_run.sh batch_plan.txt
  ```

  - Step 3: Reproducing Table 3 only needs calculation using the following commands within **one second**:
  
  ```bash
  # for Table 3
  python network/main.py
  ```

* **Expected Results**
  - After Step 3, you are expected to see the **exact** data of Table 3 in the command line.
  - The processing of the collected data for Figure 10 is deferred to the next experiment (**E3**).

#### E3 [*Efficiency of Pipeline Acceleration*]
* **Preparation** Similar to **E2**, Please make sure that you have followed the instructions in Section [Cluster Deployment](#4-cluster-deployment) and enter the folder `exploration/ae-cluster`.
* **Execution**
  - Step 1: Again, start and configure the AWS EC2 cluster with the following commands in **two to three** minutes

  ```bash
  bash manage_cluster.sh start
  # after the above command returns and wait for approximately one more minute
  bash manage_cluster.sh limit_bandwidth
  ```

  - Step 2: Reproducing the remaining part of Figure 10 needs to run the following commands, which may takes **three to four** days to complete. Similarly, a successful completion of jobs will be indicated by the final line `Batch jobs ended.` appended to the log file `batch_log.txt`. The cluster will also be then shut down automatically.
  
  ```bash
  # for the other part of Figure 10
  bash batch_run.sh batch_plan_2.txt
  ```

  - Step 3: Processing the collected data using the following commands, which only takes **several minutes**.
  ```bash
  cd performance
  bash batch_plot.sh
  ```
  
* **Expected Results**
  - After Step 2, you are expected to generated images **similar** to the ones presented in Figure 10.

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

## Notes

Please consider citing our paper if 
you use the code or data in your research project.

```bibtex
@inproceedings{jiang2024efficient,
  author={Jiang, Zhifeng and Wang, Wei and Ruichuan, Chen},
  title={Efficient Federated Learning with Dropout-Resilient Differential Privacy},
  year={2024},
  booktitle={EuroSys},
}
```

## Contact
Zhifeng Jiang (zjiangaj@cse.ust.hk).
