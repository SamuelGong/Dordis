import sys
import json
from utils import ExecutionEngine


def initialize(launch_result_path, local_private_key_path,
               last_response_path, node_private_key_path, github_repo):
    with open(launch_result_path, 'r') as fin:
        launch_result = json.load(fin)
    project_name = github_repo.split('/')[-1].split('.')[0]

    execution_plan_list = []
    for idx, simplified_response in enumerate(launch_result):
        name = simplified_response['name']
        public_ip = simplified_response['public_ip']
        execution_sequence = []

        copy_node_private_key = f"scp -q -i {local_private_key_path} " \
                                f"-o StrictHostKeyChecking=no " \
                                f"-o UserKnownHostsFile=/dev/null " \
                                f"{node_private_key_path} " \
                                f"ubuntu@{public_ip}:/home/ubuntu/.ssh/"
        execution_sequence.append((
            'local', [copy_node_private_key]
        ))
        execution_sequence.append((
            'remote',
            {
                'username': 'ubuntu',
                'key_filename': local_private_key_path,
                # only initialize if project does not exist
                # (compatible for scaling)
                'commands': [
                    f'[ ! -d "{project_name}" ] '
                    '&& ssh-keyscan -t rsa github.com >> ~/.ssh/known_hosts '
                    f'&& git clone {github_repo}',
                ]
            }
        ))
        execution_sequence.append((
            'prompt', [f"Initialized node {name} ({public_ip})."]
        ))

        execution_plan = {
            'name': name,
            'public_ip': public_ip,
            'execution_sequence': execution_sequence
        }
        execution_plan_list.append((execution_plan,))

    engine = ExecutionEngine()
    last_response = engine.run(execution_plan_list)
    with open(last_response_path, 'w') as fout:
        json.dump(last_response, fout, indent=4)


def general(launch_result_path, local_private_key_path,
            last_response_path, github_repo, core_command):
    with open(launch_result_path, 'r') as fin:
        launch_result = json.load(fin)
    project_name = github_repo.split('/')[-1].split('.')[0]

    execution_plan_list = []
    for idx, simplified_response in enumerate(launch_result):
        name = simplified_response['name']
        public_ip = simplified_response['public_ip']
        execution_sequence = [
            ('remote', {
                'username': 'ubuntu',
                'key_filename': local_private_key_path,
                'commands': [f'cd {project_name} && git {core_command}']
            }),
            ('prompt', [f"Updated the repo {project_name} on {name} ({public_ip})."])
        ]

        execution_plan = {
            'name': name,
            'public_ip': public_ip,
            'execution_sequence': execution_sequence
        }
        execution_plan_list.append((execution_plan,))

    engine = ExecutionEngine()
    # otherwise c5.xlarge only has 4 cores, too slow for 100 clients!
    last_response = engine.run(execution_plan_list, 4)
    with open(last_response_path, 'w') as fout:
        json.dump(last_response, fout, indent=4)


def main(args):
    command = args[0]

    if command == "initialize":
        launch_result_path, local_private_key_path, last_response_path, \
            node_private_key_path, github_repo = args[1:6]
        initialize(launch_result_path, local_private_key_path,
                   last_response_path, node_private_key_path, github_repo)

    elif command == "pull":
        launch_result_path, local_private_key_path, last_response_path, \
            github_repo = args[1:5]
        general(launch_result_path, local_private_key_path,
                last_response_path, github_repo, "pull")

    elif command == "checkout":
        launch_result_path, local_private_key_path, last_response_path, \
            github_repo, repo_branch = args[1:6]
        general(launch_result_path, local_private_key_path,
                last_response_path, github_repo, f"checkout {repo_branch}")
    else:
        print('Unknown commands!')


if __name__ == '__main__':
    main(sys.argv[1:])