#!/bin/bash

bash cluster_run.sh analyze_tasks pipeline-time/femnist_0_plot_plan.yml
bash cluster_run.sh analyze_tasks pipeline-time/femnist_10_plot_plan.yml
bash cluster_run.sh analyze_tasks pipeline-time/femnist_20_plot_plan.yml
bash cluster_run.sh analyze_tasks pipeline-time/femnist_30_plot_plan.yml
bash cluster_run.sh analyze_tasks pipeline-time/femnist_0_big_plot_plan.yml
bash cluster_run.sh analyze_tasks pipeline-time/femnist_10_big_plot_plan.yml
bash cluster_run.sh analyze_tasks pipeline-time/femnist_20_big_plot_plan.yml
bash cluster_run.sh analyze_tasks pipeline-time/femnist_30_big_plot_plan.yml
bash cluster_run.sh analyze_tasks pipeline-time/cifar10_0_plot_plan.yml
bash cluster_run.sh analyze_tasks pipeline-time/cifar10_10_plot_plan.yml
bash cluster_run.sh analyze_tasks pipeline-time/cifar10_20_plot_plan.yml
bash cluster_run.sh analyze_tasks pipeline-time/cifar10_30_plot_plan.yml
bash cluster_run.sh analyze_tasks pipeline-time/cifar10_0_big_plot_plan.yml
bash cluster_run.sh analyze_tasks pipeline-time/cifar10_10_big_plot_plan.yml
bash cluster_run.sh analyze_tasks pipeline-time/cifar10_20_big_plot_plan.yml
bash cluster_run.sh analyze_tasks pipeline-time/cifar10_30_big_plot_plan.yml