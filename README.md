# Hyades

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
5. [Repo Structure](#5-repo-structure)
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
called `hyades` and thus will not mess with the original system.

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
exploration/ae-simulator/[target folder]/[some timestamp]/hyades-coordinator/log.txt
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

## 5. Repo Structure

```
Repo Root
|---- hyades                           # Core implementation
|---- exploration                      # Evaluation
    |---- cluster_folder_template      # Basic tools for cluster deployment
    |---- simulation_folder_template   # Basic tools for single-node simulation
    |---- dev                          # Shared experiment backend
    |---- analysis                     # Shared data processing backend
```

## Notes

Please consider citing our paper if 
you use the code or data in your research project.

```bibtex
@inproceedings{efficient2024jiang,
  author={Jiang, Zhifeng and Wang, Wei and Ruichuan, Chen},
  title={Efficient Federated Learning with Dropout-Resilient Differential Privacy},
  year={2024},
  booktitle={EuroSys},
}
```

## Contact
Zhifeng Jiang (zjiangaj@cse.ust.hk).
