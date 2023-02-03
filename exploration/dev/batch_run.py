import time
import json
import sys
import os
import logging
from utils import execute_remotely, \
    execute_locally, calc_sleep_time
import torch

project_name = "hyades"
setup_handler_rel = "setup.sh"
cluster_run_handler_rel = "cluster_run.sh"
simulator_run_handler_rel = "simulator_run.sh"
manage_cluster_hanlder_rel = "manage_cluster.sh"
catelog_rel = "catelog.yml"


def poll_if_stop(coordinator_public_ip, local_private_key_path):
    username = 'ubuntu'
    commands = [f'ps -ef | grep {project_name} | grep python']
    try:
        raw_resp = execute_remotely(commands, coordinator_public_ip,
                                    username, local_private_key_path)
        resp = raw_resp[0]['stdout']
        stopped = True
        for line in resp:
            if 'config' in line.split('\n')[0]:
                stopped = False
                break
        if stopped:  # for debugging only, should be deleted later
            logging.info(f"It seems that the program has stopped. "
                         f"The responses are {resp}")
        return stopped
    except Exception as e:  # probably SSH errors
        return str(e)


def local_poll_if_stop():
    commands = [f"ps -ef | grep -E '{project_name}|run' | grep python"]
    try:
        raw_resp = execute_locally(commands, shell=True)
        resp = raw_resp[0]['resp']
        resp = resp.split('\n')

        stopped = True
        for line in resp:
            if 'config' in line:
                stopped = False
                break
        if stopped:  # for debugging only, should be deleted later
            logging.info(f"It seems that the program has stopped. "
                         f"The responses are {resp}")
        return stopped
    except Exception as e:  # probably SSH errors
        return str(e)


def start_tasks(launch_result_path, local_private_key_path,
                batch_plan_path, shutdown_after_completion=True):
    logging.basicConfig(
        format='[%(levelname)s][%(asctime)s]: %(message)s',
        level=logging.INFO,
        datefmt='(%m-%d) %H:%M:%S')
    logging.getLogger("paramiko").setLevel(logging.WARNING)
    logging.info(f'Batch jobs started. '
                 f'The pid of this coordinator is {os.getpid()}. '
                 f'Will shut down the cluster after completion: '
                 f'{shutdown_after_completion}.')

    # Step 1: find the coordinator's public IP
    with open(launch_result_path, 'r') as fin:
        launch_result = json.load(fin)
    for node in launch_result:
        if 'coordinator' in node['name']:
            coordinator_public_ip = node['public_ip']
            break
    logging.info(f'The public IP of the coordinator is {coordinator_public_ip}')

    # Step 2: launching and polling
    with open(batch_plan_path, 'r') as fin:
        tasks = fin.readlines()
    tasks = [task.split('\n')[0] for task in tasks]

    # have a small test to avoid waste of time
    should_stop = False
    for task in tasks:
        if not os.path.exists(task):
            logging.info(f"File {task} does not exist! "
                         f"Please proofread the plan.")
            should_stop = True
    if should_stop:
        exit(0)

    poll_interval = 20
    for idx, task in enumerate(tasks):
        if not task:
            continue

        start_cmd = f'bash {setup_handler_rel} clean_memory'
        try:
            _ = execute_locally([start_cmd])
        except Exception as e:
            logging.info(f'Failed to clean memory. '
                  f'Messages: {e} ({type(e)}))')
            continue

        start_cmd = f'bash {cluster_run_handler_rel} start_a_task {task}'
        try:
            raw_resp = execute_locally([start_cmd])
        except Exception as e:
            logging.info(f'Failed to start {task}. '
                  f'Messages: {e} ({type(e)}))')
            continue

        resp_list = raw_resp[0]['resp'].split('\n')

        # highly dependent on the code in plato_related.py
        try:
            kill_cmd = resp_list[-8].replace('\t', '')
            kill_cmd_2 = resp_list[-7].replace('\t', '')
            kill_cmd += ' && ' + kill_cmd_2
            conclude_cmd = resp_list[-5].replace('\t', '')
            analyze_cmd = resp_list[-3].replace('\t', '')
            vim_cmd = resp_list[-1].replace('\t', '')
        except Exception as e:  # error case 1
            logging.info(f'Failed to start {task}. '
                         f'Messages: {resp_list}')
            continue
        else:
            if 'bash' not in kill_cmd:  # error case 2
                logging.info(f'Failed to start Task {idx}: {task}. '
                             f'Responses: {resp_list}')
                continue

        logging.info(f'Task {idx}: {task} started.\n'
                     f'Can be killed halfway by \n\t{kill_cmd}\n'
                     f'Can be concluded by \n\t{conclude_cmd}\n'
                     f'Can be analyzed by \n\t{analyze_cmd}\n'
                     f'Can be viewed at a computing node by \n\t{vim_cmd}')

        poll_step = 0
        time.sleep(3)  # TODO: avoid hard-coding
        start_time = time.perf_counter()
        logging.info(f'Polling its status...')
        while True:
            stopped = poll_if_stop(coordinator_public_ip, local_private_key_path)
            if isinstance(stopped, bool):
                logging.info(f"\tElapsed time: "
                             f"{round(time.perf_counter() - start_time)}, "
                             f"task {task} has stopped: {stopped}.")
                if stopped:
                    break
            else:
                logging.info(f"\tElapsed time: "
                             f"{round(time.perf_counter() - start_time)}. "
                             f"Status cannot be probed due to {stopped}. "
                             f"Retry in {poll_interval} seconds.")

            sleep_time = calc_sleep_time(poll_interval,
                                         poll_step, start_time)
            time.sleep(sleep_time)
            poll_step += 1

        try:
            _ = execute_locally([conclude_cmd])
        except Exception as e:
            logging.info(f'Failed to conclude {task}. '
                         f'Messages: {e} ({type(e)}))')
            continue
        logging.info(f'Task {idx}: {task} concluded.')

        try:
            _ = execute_locally([analyze_cmd])
        except Exception as e:
            logging.info(f'Failed to analyze {task}. '
                         f'Messages: {e} ({type(e)}))')
            continue
        logging.info(f'Task {idx}: {task} analyzed.')

        # for ease of analysis
        # assume an item in batch_plan has exactly two components, i.e,
        # [folder_name]/[config_file_name (ended with .yml)]
        config_file_name = task.split('/')[1][:-4]
        timestamp_str = conclude_cmd.split(' ')[-1].split('/')[1]
        catelog_line = f"{timestamp_str}: {config_file_name}"

        parent_folder = task.split('/')[0]
        catelog_path = os.path.join(parent_folder, catelog_rel)
        if not os.path.exists(catelog_path):
            with open(catelog_path, 'w+') as fout:
                fout.writelines([catelog_line])
        else:
            with open(catelog_path, 'a') as fout:
                fout.writelines(['\n' + catelog_line])

    logging.info(f'Batch jobs ended.')

    if shutdown_after_completion:
        shutdown_cmd = f'bash {manage_cluster_hanlder_rel} stop'
        try:
            _ = execute_locally([shutdown_cmd])
        except Exception as e:
            logging.info(f'Failed to shut down the cluster. '
                         f'Messages: {e} ({type(e)}))')
        else:
            logging.info(f'The cluster shut down.')


def local_start_tasks(batch_plan_path):
    logging.basicConfig(
        format='[%(levelname)s][%(asctime)s]: %(message)s',
        level=logging.INFO,
        datefmt='(%m-%d) %H:%M:%S')
    logging.getLogger("paramiko").setLevel(logging.WARNING)
    logging.info(f'Batch jobs started. '
                 f'The pid of this coordinator is {os.getpid()}. ')

    with open(batch_plan_path, 'r') as fin:
        tasks = fin.readlines()
    tasks = [task.split('\n')[0] for task in tasks]

    should_stop = False
    for task in tasks:
        if not os.path.exists(task):
            logging.info(f"File {task} does not exist! "
                         f"Please proofread the plan.")
            should_stop = True
    if should_stop:
        exit(0)

    simulator_uses_gpu = torch.cuda.is_available()
    poll_interval = 20
    for idx, task in enumerate(tasks):
        if not task:
            continue

        start_cmd = f'bash {simulator_run_handler_rel} ' \
                    f'start_a_task {task}'
        try:
            raw_resp = execute_locally([start_cmd])
        except Exception as e:
            logging.info(f'Failed to start {task}. '
                  f'Messages: {e} ({type(e)}))')
            continue

        resp_list = raw_resp[0]['resp'].split('\n')

        # highly dependent on the code in plato_related.py
        try:
            kill_cmd = resp_list[-6].replace('\t', '')
            kill_cmd_2 = resp_list[-5].replace('\t', '')
            kill_cmd += ' && ' + kill_cmd_2
            analyze_cmd = resp_list[-3].replace('\t', '')
            vim_cmd = resp_list[-1].replace('\t', '')
        except Exception as e:  # error case 1
            logging.info(f'Failed to start {task}. '
                         f'Messages: {resp_list}')
            continue
        else:
            if 'bash' not in kill_cmd:  # error case 2
                logging.info(f'Failed to start Task {idx}: {task}. '
                             f'Responses: {resp_list}')
                continue

        logging.info(f'Task {idx}: {task} started.\n'
                     f'Can be killed halfway by \n\t{kill_cmd}\n'
                     f'Can be analyzed by \n\t{analyze_cmd}\n'
                     f'Can be viewed at a computing node by \n\t{vim_cmd}')

        poll_step = 0

        if simulator_uses_gpu:
            # may need more cold-start time
            time.sleep(30)  # TODO: avoid hard-coding
        else:
            time.sleep(3)  # TODO: avoid hard-coding
        start_time = time.perf_counter()
        logging.info(f'Polling its status...')
        while True:
            stopped = local_poll_if_stop()
            if isinstance(stopped, bool):
                logging.info(f"\tElapsed time: "
                             f"{round(time.perf_counter() - start_time)}, "
                             f"task {task} has stopped: {stopped}.")
                if stopped:
                    break
            else:
                logging.info(f"\tElapsed time: "
                             f"{round(time.perf_counter() - start_time)}. "
                             f"Status cannot be probed due to {stopped}. "
                             f"Retry in {poll_interval} seconds.")

            sleep_time = calc_sleep_time(poll_interval,
                                         poll_step, start_time)
            time.sleep(sleep_time)
            poll_step += 1

        try:
            _ = execute_locally([analyze_cmd])
        except Exception as e:
            logging.info(f'Failed to analyze {task}. '
                         f'Messages: {e} ({type(e)}))')
            continue
        logging.info(f'Task {idx}: {task} analyzed.')

        # for ease of analysis
        # assume an item in batch_plan has exactly two components, i.e,
        # [folder_name]/[config_file_name (ended with .yml)]
        config_file_name = task.split('/')[-1][:-4]
        timestamp_str = analyze_cmd.split(' ')[-1].split('/')[1]
        catelog_line = f"{timestamp_str}: {config_file_name}"

        simulator_path = '/'.join(batch_plan_path.split('/')[:-1])
        parent_folder = task.split('/')[-2]
        parent_folder = os.path.join(simulator_path, parent_folder)
        catelog_path = os.path.join(parent_folder, catelog_rel)
        if not os.path.exists(catelog_path):
            with open(catelog_path, 'w+') as fout:
                fout.writelines([catelog_line])
        else:
            with open(catelog_path, 'a') as fout:
                fout.writelines(['\n' + catelog_line])

    logging.info(f'Batch jobs ended.')


def main(args):
    command = args[0]
    if command == 'start_tasks':
        launch_result_path = args[1]
        local_private_key_path = args[2]
        batch_plan_path = args[3]
        start_tasks(launch_result_path,
                    local_private_key_path, batch_plan_path)
    elif command == "local_start_tasks":
        batch_plan_path = args[1]
        local_start_tasks(batch_plan_path)
        # start_tasks(launch_result_path,
        #             local_private_key_path, batch_plan_path, True)


if __name__ == '__main__':
    main(sys.argv[1:])
