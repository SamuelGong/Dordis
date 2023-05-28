import sys
import json
import os
import yaml
from utils import ExecutionEngine
from shutil import copy as shutil_copy
import copy

from utils import get_chunks_idx, get_chunks_idx_with_mod

project_name = "SAStream"
exploration_dir_rel = "exploration"
dev_dir_rel = "dev"
conda_env_name = project_name.lower()

# clients are distributed to machines
# in chunks where each chunk is of size mod_used_in_chunk_idx
mod_used_in_chunk_idx = 1


def copy_config(config_path, task_folder, config_rel):
    new_config_path = os.path.join(task_folder, config_rel)
    shutil_copy(config_path, new_config_path)
    return new_config_path


def my_insert(cur, tup, val):
    if len(tup) == 1:
        cur[tup[0]] = val
        return
    if tup[0] not in cur:
        cur[tup[0]] = {}
    my_insert(cur[tup[0]], tup[1:], val)


def edit_config(config_path, keys, value):
    with open(config_path, 'r') as fin:
        dictionary = yaml.load(fin, Loader=yaml.FullLoader)
    my_insert(dictionary, keys, value)
    with open(config_path, 'w') as fout:
        yaml.dump(dictionary, fout)


def extract_coordinator_address(launch_result):
    for idx, simplified_response in enumerate(launch_result):
        name = simplified_response['name']
        public_ip = simplified_response['public_ip']
        if 'coordinator' in name:
            return public_ip
    return None


def get_client_launch_plan(config_path, launch_result):
    with open(config_path, 'r') as fin:
        config = yaml.load(fin, Loader=yaml.FullLoader)

    if 'num_physical_clients' in config['clients'] \
            and 'resource_saving' in config['clients'] \
            and config['clients']['resource_saving']:
        num_physical_clients = config['clients']['num_physical_clients']
    else:
        num_physical_clients = config['clients']['total_clients']

    num_workers = len(launch_result) - 1
    range_generator = get_chunks_idx_with_mod(
        l=num_physical_clients,
        n=num_workers,
        mod=mod_used_in_chunk_idx
    )

    client_launch_plan = {}
    for node in launch_result:
        name = node['name']
        if 'worker' in name:
            begin, end = next(range_generator)
            num_clients = end - begin
            client_launch_plan[name] = num_clients
    return client_launch_plan


def remote_execution(command, args):
    launch_result_path = args[1]
    local_private_key_path = args[2]
    last_response_path = args[3]
    with open(launch_result_path, 'r') as fin:
        launch_result = json.load(fin)

    if command in ["remote_start", "remote_kill", "collect_result"]:
        last_response_rel = last_response_path
        issue_result_rel = 'pid.txt'  # TODO: avoid hard-coding
        log_rel = 'log.txt'
        config_rel = 'config.yml'
        result_rel = 'result.csv'

        if command == "remote_start":
            start_time = args[4]
            config_path = args[5]
            client_port = 80
            coordinator_address = extract_coordinator_address(launch_result)

            task_parent_folder = '/'.join(config_path.split('/')[:-1])
            task_folder = os.path.join(task_parent_folder, start_time)
            os.makedirs(task_folder)
            last_response_path = os.path.join(task_folder, last_response_rel)

            idx = task_parent_folder.find(project_name)
            remote_task_parent_folder = task_parent_folder[idx:]
            remote_task_folder = os.path.join(remote_task_parent_folder, start_time)
            client_launch_plan = get_client_launch_plan(
                config_path, launch_result)
            new_config_path = copy_config(config_path, task_folder, config_rel)
            edit_config(new_config_path, ['results', 'results_dir'],
                        '/home/ubuntu/' + remote_task_folder + '/')
        else:
            task_folder = args[4]
            idx = task_folder.find(project_name)
            remote_task_folder = task_folder[idx:]
            last_response_path = os.path.join(task_folder, last_response_rel)
            if command in ["remote_kill", "collect_result"]:
                config_path = os.path.join(task_folder, config_rel)
                client_launch_plan = get_client_launch_plan(
                    config_path, launch_result)

        remote_task_folder_short = '/'.join(remote_task_folder.split('/')[-2:])
    elif command in ["add_pip_dependency", "add_apt_dependency"]:
        dependency = args[4]

    execution_plan_list = []
    remote_template = {
        'username': 'ubuntu',
        'key_filename': local_private_key_path
    }
    client_idx = 1  # cannot start from 0 if using FEMNIST!
    for idx, simplified_response in enumerate(launch_result):
        name = simplified_response['name']
        if 'coordinator' not in name \
                and command in ["remote_start", "remote_kill", "collect_result"]:
            num_logical_clients = client_launch_plan[name]
            if num_logical_clients == 0:
                continue

        public_ip = simplified_response['public_ip']
        execution_sequence = []

        if command == "standalone":
            remote_template.update({
                'commands': [
                    f"cd ~/{project_name}/{exploration_dir_rel}/{dev_dir_rel} "
                    "&& source standalone_install.sh"
                ]
            })
            execution_sequence = [
                ('remote', remote_template),
                ('prompt', [f'Standalone installation finished on node '
                            f'{name} ({public_ip}).'])
            ]

        elif command == "clean_memory":
            remote_template.update({
                'commands': [
                    "sudo sh -c 'echo 3 >  /proc/sys/vm/drop_caches'"
                ]
            })
            execution_sequence = [
                ('remote', remote_template),
                ('prompt', [f'Memory clean on node '
                            f'{name} ({public_ip}).'])
            ]

        elif command == "deploy_cluster":
            if 'coordinator' in name:
                remote_template.update({
                    'commands': [
                        f"cd ~/{project_name}/{exploration_dir_rel}/{dev_dir_rel} "
                        "&& source cluster_install.sh"
                    ]
                })
            else:
                continue

            execution_sequence = [
                ('remote', remote_template),
                ('prompt', [f'Cluster server deployed on '
                            f'{name} ({public_ip}).'])
            ]

        elif command == "start_cluster":
            if 'coordinator' in name:
                remote_template.update({
                    'commands': [
                        "sudo systemctl start nginx"
                    ]
                })
                execution_sequence = [
                    ('remote', remote_template),
                    ('prompt', [f'Cluster server started on '
                                f'{name} ({public_ip}).'])
                ]

        elif command == "add_pip_dependency":
            remote_template.update({
                'commands': [
                    "source ~/anaconda3/etc/profile.d/conda.sh && "
                    f"conda activate {conda_env_name} && "
                    f"pip install {dependency} && "
                    "conda deactivate"
                ]
            })
            execution_sequence = [
                ('remote', remote_template),
                ('prompt', [f'Pip dependency {dependency} installed on '
                            f'{name} ({public_ip}).'])
            ]

        elif command == "add_apt_dependency":
            remote_template.update({
                'commands': [
                    f"sudo apt install {dependency} -y"
                ]
            })
            execution_sequence = [
                ('remote', remote_template),
                ('prompt', [f'Apt dependency {dependency} installed on '
                            f'{name} ({public_ip}).'])
            ]

        elif command == "remote_start":
            if 'coordinator' in name:
                before_copy = copy.deepcopy(remote_template)
                after_copy = copy.deepcopy(remote_template)
                before_copy.update({
                    'commands': [
                        f"cd {remote_task_parent_folder} && mkdir -p {start_time}"
                    ]
                })
                after_copy.update({
                    'commands': [
                        f"source ~/anaconda3/etc/profile.d/conda.sh && "
                        f"cd ~/{project_name} && conda activate {conda_env_name} && "
                        f"cat ~/{remote_task_folder}/{name}-{config_rel} "
                        f"> ~/{remote_task_folder}/{log_rel} && ./run "
                        f"--config=$HOME/{remote_task_folder}/{name}-{config_rel} "
                        f">> ~/{remote_task_folder}/{log_rel} 2>&1 &"
                    ]
                })
                execution_sequence = [
                    ('remote', before_copy),
                    ('local', [
                        f"scp -q -i {local_private_key_path} "
                        f"-o StrictHostKeyChecking=no "
                        f"-o UserKnownHostsFile=/dev/null "
                        f"{new_config_path} "
                        f"ubuntu@{public_ip}:"
                        f"~/{remote_task_folder}/{name}-{config_rel}"
                    ]),
                    ('remote', after_copy),
                    ('prompt', [f'Task {start_time} started on '
                                f'{name} ({public_ip}).'])
                ]
            else:
                before_copy = copy.deepcopy(remote_template)
                after_copy = copy.deepcopy(remote_template)
                before_copy.update({
                    'commands': [
                        # "sleep 10",  # allow some time for the server to start
                        f"cd {remote_task_parent_folder} && mkdir -p {start_time}"
                    ]
                })
                commands = []
                num_clients = client_launch_plan[name]
                commands.append(f"cat ~/{remote_task_folder}/{name}-{config_rel} "
                                f"> ~/{remote_task_folder}/{log_rel}")
                for i in range(num_clients):
                    actual_client_idx = client_idx + i
                    commands.append(f"source ~/anaconda3/etc/profile.d/conda.sh && "
                                    f"cd ~/{project_name} && conda activate {conda_env_name} && ./run_client "
                                    f"--config=$HOME/{remote_task_folder}/{name}-{config_rel} "
                                    f"-i {actual_client_idx} "
                                    f"--server {coordinator_address}:{client_port} "
                                    f">> ~/{remote_task_folder}/{log_rel} 2>&1 &")
                client_idx += num_clients

                after_copy.update({ 'commands': commands })
                execution_sequence = [
                    ('remote', before_copy),
                    ('local', [
                        f"scp -q -i {local_private_key_path} "
                        f"-o StrictHostKeyChecking=no "
                        f"-o UserKnownHostsFile=/dev/null "
                        f"{new_config_path} "
                        f"ubuntu@{public_ip}:"
                        f"~/{remote_task_folder}/{name}-{config_rel}"
                    ]),
                    ('remote', after_copy),
                    ('prompt', [f'Task {start_time} started on '
                                f'{name} ({public_ip}).'])
                ]

        elif command == "remote_kill":  # only work when there is only one task running
            remote_template.update({
                'commands': [
                    f"ps -ef | grep {conda_env_name} | grep python > "
                    f"{remote_task_folder}/{issue_result_rel}",
                    # f"ps -ef | grep {conda_env_name} | grep python >> "
                    # f"{remote_task_folder}/{issue_result_rel}",
                    f"cat {remote_task_folder}/{issue_result_rel} | awk '{{print $2}}' "
                    f"| xargs kill -9 1>/dev/null 2>&1"
                ]
            })
            execution_sequence = [
                ('remote', remote_template),
                ('prompt', [f'Task {remote_task_folder_short} killed on '
                            f'{name} ({public_ip}).'])
            ]

        elif command == "collect_result":
            if 'coordinator' in name:
                remote_template.update({  # so that git pull does not alert of inconsistency
                    'commands': [
                        f"rm -rf ~/{remote_task_folder}/{result_rel}"
                    ]
                })
                execution_sequence = [
                    ('local', [
                        f"mkdir -p {task_folder}/{name}",
                        f"scp -q -i {local_private_key_path} "
                        f"-o StrictHostKeyChecking=no "
                        f"-o UserKnownHostsFile=/dev/null "
                        f"ubuntu@{public_ip}:~/{remote_task_folder}/{log_rel} "
                        f"{task_folder}/{name}",
                        f"scp -q -i {local_private_key_path} "
                        f"-o StrictHostKeyChecking=no "
                        f"-o UserKnownHostsFile=/dev/null "
                        f"ubuntu@{public_ip}:~/{remote_task_folder}/{result_rel} "
                        f"{task_folder}",
                    ]),
                    ('remote', remote_template),
                    ('prompt', [f'Partial results for {remote_task_folder_short} '
                                f'retrieved from {name} ({public_ip}).'])
                ]
            else:
                execution_sequence = [
                    ('local', [
                        f"mkdir -p {task_folder}/{name}",
                        f"scp -q -i {local_private_key_path} "
                        f"-o StrictHostKeyChecking=no "
                        f"-o UserKnownHostsFile=/dev/null "
                        f"ubuntu@{public_ip}:~/{remote_task_folder}/{log_rel} "
                        f"{task_folder}/{name}"
                    ]),
                    ('prompt', [f'Partial results for {remote_task_folder_short} '
                                f'retrieved from {name} ({public_ip}).'])
                ]
        else:
            execution_sequence = []

        execution_plan = {
            'name': name,
            'public_ip': public_ip,
            'execution_sequence': execution_sequence
        }
        execution_plan_list.append((execution_plan,))

    engine = ExecutionEngine()
    last_response = engine.run(execution_plan_list, 4)
    with open(last_response_path, 'w') as fout:
        json.dump(last_response, fout, indent=4)

    if command == "remote_start":
        print(f"\nUse the command at local host if you want to stop it:\n"
              f"\tbash cluster_run.sh kill_a_task {remote_task_folder_short}\n"
              f"\trm -rf {remote_task_folder_short}")
        print(f"Use the command at local host if you want to retrieve the log:\n"
              f"\tbash cluster_run.sh conclude_a_task {remote_task_folder_short}")
        print(f"Use the command at local host if you want to analyze the log:\n"
              f"\tbash cluster_run.sh analyze_a_task {remote_task_folder_short}")
        print(f"Use the command at a computing node if you want to watch its state:\n"
              f"\tvim ~/{remote_task_folder}/{log_rel}")


def simulation(command, args):
    config_rel = 'config.yml'
    issue_result_rel = 'pid.txt'
    log_rel = 'log.txt'

    if command == "local_start":
        working_dir = args[1]
        last_response_path = args[2]
        start_time = args[3]
        template_config_path = args[4]

        target_project = template_config_path.split('/')[-2]
        task_folder = os.path.join(working_dir, target_project, start_time)
        os.makedirs(task_folder)

        # Add simulation attributes
        config_path = copy_config(template_config_path, task_folder, config_rel)
        edit_config(config_path, ['simulation', 'type'], 'simple')

        # Throttle the number of server's ports (especially for runinng in a laptop)
        edit_config(config_path, ['server', 'port'], [8001, 8002, 8003, 8004])

        # Turn on debugging function
        edit_config(config_path, ['app', 'debug'], {
            'client': {
                'sketch_num': 6
            },
            'server': {
                'sketch_num': 6,
                'test': True
            },
        })

        task_folder_short = '/'.join(task_folder.split('/')[-2:])
        with open(config_path, 'r') as fin:
            config = yaml.load(fin, Loader=yaml.FullLoader)
        # print(config)
        # exit(0)
    elif command in ["local_kill", "local_collect_results"]:
        task_folder = args[1]
        last_response_path = args[2]
        config_path = os.path.join(task_folder, config_rel)
    elif command in ["local_standalone"]:
        last_response_path = args[1]
    else:
        raise ValueError(f"Command {command} is not supported.")

    if command in ["local_standalone", "local_clean_memory"]:
        names = ["hyades-coordinator"]
    else:
        task_folder_short = '/'.join(task_folder.split('/')[-2:])
        with open(config_path, 'r') as fin:
            config = yaml.load(fin, Loader=yaml.FullLoader)

        if 'num_physical_clients' in config['clients'] \
                and 'resource_saving' in config['clients'] \
                and config['clients']['resource_saving']:
            num_physical_clients = config['clients']['num_physical_clients']
        else:
            num_physical_clients = config['clients']['total_clients']
        names = [f"hyades-worker-{client_idx}" for client_idx
                 in range(1, num_physical_clients + 1)] + ["hyades-coordinator"]

        if command in ["local_start"]:
            client_ports = config["server"]['port']

    execution_plan_list = []
    client_idx = 1
    for name in names:
        execution_sequence = []

        if command == "local_standalone":
            execution_sequence = [
                ('local-shell', [
                    f"cd ../{dev_dir_rel} && pwd "
                    "&& source ./standalone_install.sh"
                ]),
                ('prompt', [f'Standalone installation locally finished.'])
            ]
        elif command == "local_start":
            entity_dir = f"{task_folder}/{name}"
            entity_log_path = f"{entity_dir}/{log_rel}"
            project_path = f"{working_dir}/../.."
            execution_sequence = [
                ('local-shell', [
                    f"cd {task_folder} && mkdir -p {entity_dir} "
                    f"&& cat {config_path} > {entity_log_path}",
                ]),
            ]

            if 'coordinator' in name:
                execution_sequence += [
                    # if not shell, cannot use redirect features
                    ('local-shell', [
                        f"cd {project_path} && "
                        f"source ~/anaconda3/etc/profile.d/conda.sh && "
                        f"conda activate {conda_env_name} && "
                        f"export OPENBLAS_NUM_THREADS=2 && "
                        f"./run --config={config_path} >> {entity_log_path} 2>&1 &"
                    ]),
                    # ('local-shell', [
                    #     f"echo $-; shopt login_shell"
                    # ]),
                    ('prompt', [f'Task {start_time} locally started.'])
                ]
            else:
                client_port = client_ports[(client_idx - 1) % len(client_ports)]
                execution_sequence += [
                    # if not shell, cannot use redirect features
                    ('local-shell', [
                        f"cd {project_path} && "
                        f"source ~/anaconda3/etc/profile.d/conda.sh && "
                        f"conda activate {conda_env_name} && "
                        f"export OPENBLAS_NUM_THREADS=2 && "
                        f"./run_client -i {client_idx} --config={config_path} "
                        f"--server 127.0.0.1:{client_port} "
                        f">> {entity_log_path} 2>&1 &"
                    ])
                    # ('local-shell', [
                    #     f"date"
                    # ])
                ]
                client_idx += 1
        elif command == "local_kill":
            if 'coordinator' in name:
                execution_sequence = [
                    ('local-shell', [
                        f"ps -ef | grep -E '{conda_env_name}|run' | grep python > "
                        f"{task_folder}/{issue_result_rel}",
                        f"cat {task_folder}/{issue_result_rel} | awk '{{print $2}}' "
                        f"| xargs kill -9 1>/dev/null 2>&1"
                    ]),
                    ('prompt', [f'Task {task_folder_short} locally killed.'])
                ]
            else:
                pass

        execution_plan = {
            'name': name,
            'execution_sequence': execution_sequence
        }
        execution_plan_list.append((execution_plan,))

    engine = ExecutionEngine()
    last_response = engine.run(execution_plan_list, 4)
    with open(last_response_path, 'w') as fout:
        json.dump(last_response, fout, indent=4)

    if command == "local_start":
        print(f"\nUse the command if you want to stop it:\n"
              f"\tbash simulator_run.sh kill_a_task {task_folder_short}\n"
              f"\trm -rf {task_folder_short}")
        print(f"Use the command at local host if you want to analyze the log:\n"
              f"\tbash simulator_run.sh analyze_a_task {task_folder_short}")
        print(f"Use the command if you want to watch the server's state:\n"
              f"\tvim {entity_log_path}")

def main(args):
    command = args[0]

    if command in ["standalone", "clean_memory", "deploy_cluster", "start_cluster",
                   "remote_start", "remote_kill", "collect_result",
                   "add_pip_dependency", "add_apt_dependency"]:
        remote_execution(command, args)
    else:  # simulation
        simulation(command, args)


if __name__ == '__main__':
    main(sys.argv[1:])
