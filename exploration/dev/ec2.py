import sys
import os.path
import json
import yaml
import boto3
import botocore
import numpy as np
from datetime import date, datetime
from utils import ExecutionEngine, my_random_zipfian


class ComplexEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(obj, date):
            return obj.strftime('%Y-%m-%d')
        else:
            return json.JSONEncoder.default(self, obj)


def launch_a_node(client, node_config):
    return client.run_instances(BlockDeviceMappings=node_config['BlockDeviceMappings'],
                                InstanceType=node_config['InstanceType'],
                                TagSpecifications=node_config['TagSpecifications'],
                                NetworkInterfaces=node_config['NetworkInterfaces'],
                                ImageId=node_config['ImageId'],
                                KeyName=node_config['KeyName'],
                                MinCount=1, MaxCount=1)


def generate_network_speed(data):
    type = data["type"]
    if type == "constant":
        value = data["args"]["value"]
        return value
    elif type == "zipf":
        helping_number = 100000000
        seed = data["args"]["seed"]
        a = data["args"]["a"]
        n = data["args"]["n"]
        amin = helping_number / data["args"]["max"]
        amax = helping_number / data["args"]["min"]

        np.random.seed(seed)
        res = my_random_zipfian(
            a=a,
            n=n,
            amin=amin,
            amax=amax
        )
        res = sorted((helping_number / np.array(res))
                     .astype(int).tolist())
        return res
    else:
        raise NotImplementedError


def generate_client_config(data):
    node_list = []
    type_list = data["type_list"]
    total_count = 1  # start from 1 instead of 0
    for type_dict in type_list:
        name_prefix = type_dict["name_prefix"]
        count = type_dict["count"]
        type = type_dict["type"]
        region = type_dict["region"]
        upload_kbps = generate_network_speed(type_dict["upload_kbps"])
        download_kbps = generate_network_speed(type_dict["download_kbps"])

        l = []
        for idx in range(count):
            name = name_prefix + str(total_count + idx)
            l.append({
                "name": name,
                "type": type,
                "region": region,
                "upload_kbps": upload_kbps[idx],
                "download_kbps": download_kbps[idx]
            })

        total_count += count
        node_list += l
    return node_list


def parse_cluster_config(cluster_config_path, scale=False):
    with open(cluster_config_path, 'r') as fin:
        cluster_config = yaml.load(fin, Loader=yaml.FullLoader)

    if 'clients' in cluster_config:  # backward compatible
        node_list = []
        for client_dict in cluster_config['clients']:
            the_only_key = list(client_dict.keys())[0]
            node_list.append(client_dict[the_only_key])
    else:  # then need to generate
        template = cluster_config["client_template"]
        node_list = generate_client_config(data=template)

    if not scale:  # the server is not changed during scaling
        node_list.append(cluster_config['server'])

    subnet_dict = cluster_config['subnets']
    image_dict = cluster_config['images']
    return node_list, subnet_dict, image_dict


def open_inbound_ports(client, security_group, ports):
    response = []
    for port in ports:
        try:
            resp = client.authorize_security_group_ingress(
                GroupId=security_group,
                IpPermissions = [{
                    'IpProtocol': 'all',
                    'IpRanges': [
                        {
                            'CidrIp': '0.0.0.0/0'
                        },
                    ],
                    'Ipv6Ranges': [
                        {
                            'CidrIpv6': '::/0'
                        },
                    ],
                    'FromPort': port,
                    'ToPort': port,
                }]
            )
        except botocore.exceptions.ClientError as e:  # ignore if inbound rules exist
            resp = e.response['Error']['Code']
        finally:
            response.append({port: resp})
    return response


def launch_a_cluster(cluster_config_path, node_template_path,
                     last_response_path, launch_result_path, scale=False):
    with open(node_template_path, 'r') as fin:
        node_template = yaml.load(fin, Loader=yaml.FullLoader)
    node_list, subnet_dict, image_dict = parse_cluster_config(
        cluster_config_path=cluster_config_path,
        scale=scale
    )

    nodes_to_terminate = []
    nodes_to_remain = []
    if scale and os.path.exists(launch_result_path):
        name_planned_to_run = []
        for node in node_list:
            if "coordinator" not in node["name"]:
                name_planned_to_run.append(node["name"])

        name_exists = []
        with open(launch_result_path, 'r') as fin:
            old_launch_result = json.load(fin)
        coordinator_node = None
        for node in old_launch_result:
            if "coordinator" not in node["name"]:
                name_exists.append(node["name"])
            else:
                coordinator_node = node

        name_to_launch = list(set(name_planned_to_run) - set(name_exists))
        name_to_terminate = list(set(name_exists) - set(name_planned_to_run))
        name_to_remain = list(set(name_exists)
                              .intersection(set(name_planned_to_run)))
        print(f"Already launched: {name_exists}, "
              f"nodes to launch: {name_to_launch}, "
              f"nodes to terminate: {name_to_terminate}")

        nodes_to_launch = [node for node in node_list
                           if node["name"] in name_to_launch]
        nodes_to_terminate = [node for node in old_launch_result
                              if node["name"] in name_to_terminate]
        nodes_to_remain = [node for node in old_launch_result
                           if node["name"] in name_to_remain]
        # do not forget the coordinator!
        nodes_to_remain.append(coordinator_node)
    else:
        nodes_to_launch = node_list

    launch_result = []
    if nodes_to_launch:
        last_response = []
        region_to_boto3_client_mapping = {}  # only create one client for a region
        region_to_instance_id_mapping = {}

        print(f"Launching {len(nodes_to_launch)} nodes ...")
        for node_base_config in nodes_to_launch:
            region = node_base_config['region']
            instance_type = node_base_config['type']
            name = node_base_config['name']

            if region in region_to_boto3_client_mapping:
                client = region_to_boto3_client_mapping[region]
            else:
                client = boto3.client('ec2', region_name=region)
                region_to_boto3_client_mapping[region] = client

            node_template["InstanceType"] = instance_type
            node_template["TagSpecifications"][0]["Tags"][0]["Value"] = name
            node_template["NetworkInterfaces"][0]["SubnetId"] = subnet_dict[region]
            node_template["ImageId"] = image_dict[region]

            launch_response = launch_a_node(client, node_template)
            useful_part = launch_response['Instances'][0]
            instance_id = useful_part['InstanceId']
            private_ip = useful_part['NetworkInterfaces'][0]['PrivateIpAddress']
            security_group = useful_part['SecurityGroups'][0]["GroupId"]
            if "coordinator" in name:
                ports = [22, 80]
            else:
                ports = [22]
            security_response = open_inbound_ports(client, security_group, ports)
            last_response.append({
                'name': name,
                'launch_response': launch_response,
                'allow_ingress_response': security_response
            })

            simplified_response = {
                "name": name,
                "id": instance_id,
                "region": region,
                "private_ip": private_ip,
                "security_group": security_group
            }
            launch_result.append(simplified_response)

            if region not in region_to_instance_id_mapping:
                region_to_instance_id_mapping[region] = [instance_id]
            else:
                region_to_instance_id_mapping[region].append(instance_id)

        print(f"All {len(nodes_to_launch)} nodes are launched! Waiting for ready ...")
        with open(last_response_path, 'w') as fout:
            json.dump(last_response, fout, indent=4, cls=ComplexEncoder)

        for region, instance_ids in region_to_instance_id_mapping.items():
            client = region_to_boto3_client_mapping[region]
            waiter = client.get_waiter('instance_running')
            waiter.wait(
                InstanceIds=instance_ids,
                WaiterConfig={
                    'Delay': 1,
                    'MaxAttempts': 120
                }
            )

        print(f"All {len(nodes_to_launch)} nodes are ready. "
              f"Collecting public IP addresses ...")
        for idx, simplified_response in enumerate(launch_result):
            instance_id = simplified_response['id']
            region = simplified_response['region']
            client = region_to_boto3_client_mapping[region]
            description = client.describe_instances(InstanceIds=[instance_id])
            public_ip = description['Reservations'][0]['Instances'][0]['PublicIpAddress']
            launch_result[idx]['public_ip'] = public_ip

        # show by the way
        for node in launch_result:
            print(f'{node["name"]}: {node["public_ip"]}')

    if nodes_to_terminate:
        print(f"Terminating {len(nodes_to_terminate)} nodes ...")
        selected_ids = [node["id"] for node in nodes_to_terminate]
        ec2_actions_on_a_cluster(
            action="terminate",
            last_response_path=last_response_path,
            launch_result_path=launch_result_path,
            selected_ids=selected_ids,
            write_last_response_mode='a+'  # append last response
        )

    launch_result = nodes_to_remain + launch_result
    with open(launch_result_path, 'w') as fout:
        json.dump(launch_result, fout, indent=4, cls=ComplexEncoder)


def merge_instances_by_region(launch_result, selected_ids=None):
    region_to_instance_ids_mapping = {}
    for simplified_response in launch_result:
        instance_id = simplified_response['id']
        # if has a selection intention, then ignore unselected instances
        if selected_ids and instance_id not in selected_ids:
            continue

        region = simplified_response['region']
        if region not in region_to_instance_ids_mapping:
            region_to_instance_ids_mapping[region] = [instance_id]
        else:
            region_to_instance_ids_mapping[region].append(instance_id)

    return region_to_instance_ids_mapping


def narrow_scope(launch_result, num_nodes):
    num_nodes = int(num_nodes)
    assert num_nodes > 0

    new_launch_result = []
    for node_dict in launch_result:
        name = node_dict['name']
        if 'coordinator' in name:  # TODO: avoid hard-coding
            new_launch_result.append(node_dict)
        elif 'worker' in name:
            worker_idx = int(name.split('-')[-1])
            if worker_idx <= num_nodes:
                new_launch_result.append(node_dict)
    return new_launch_result


def ec2_actions_on_a_cluster(action, last_response_path, launch_result_path,
                             num_nodes=None,
                             selected_ids=None, write_last_response_mode='w'):
    with open(launch_result_path, 'r') as fin:
        launch_result = json.load(fin)

    if num_nodes is not None:
        launch_result = narrow_scope(launch_result, num_nodes)

    region_to_instance_ids_mapping = merge_instances_by_region(
        launch_result=launch_result,
        selected_ids=selected_ids
    )
    region_to_boto3_clients_mapping = {}
    last_response = {}
    for region, instance_ids in region_to_instance_ids_mapping.items():
        client = boto3.client('ec2', region_name=region)
        region_to_boto3_clients_mapping[region] = client

        if action == 'start':
            response = client.start_instances(InstanceIds=instance_ids)
        elif action == 'reboot':
            response = client.reboot_instances(InstanceIds=instance_ids)
        elif action == 'stop':
            response = client.stop_instances(InstanceIds=instance_ids)
        elif action == 'terminate':
            response = client.terminate_instances(InstanceIds=instance_ids)
        else:
            response = None

        last_response[region] = response

    # if it is a start action,
    # we additionally need to record the new public addresses
    if action == "start":
        print("Started and waiting for ready ...")
        for region, instance_ids in region_to_instance_ids_mapping.items():
            client = region_to_boto3_clients_mapping[region]
            waiter = client.get_waiter('instance_running')
            waiter.wait(
                InstanceIds=instance_ids,
                WaiterConfig={
                    'Delay': 1,
                    'MaxAttempts': 120,
                }
            )

        print("All are ready. Collecting public IP addresses ...")
        for idx, node_config in enumerate(launch_result):
            region = node_config['region']
            client = region_to_boto3_clients_mapping[region]

            instance_id = node_config['id']
            description = client.describe_instances(InstanceIds=[instance_id])
            public_ip = description['Reservations'][0]['Instances'][0]['PublicIpAddress']
            launch_result[idx]['public_ip'] = public_ip

        with open(launch_result_path, 'w') as fout:
            json.dump(launch_result, fout, indent=4)

        # show by the way
        for node in launch_result:
            print(f'{node["name"]}: {node["public_ip"]}')

    with open(last_response_path, write_last_response_mode) as fout:
        json.dump(last_response, fout, indent=4)

    print('Done.')


def main(args):
    command = args[0]

    if command in ["launch", "scale"]:
        cluster_config_path = args[1]
        node_template_path = args[2]
        last_response_path = args[3]
        launch_result_path = args[4]
        if command == "launch":
            scale = False
        else:
            scale = True
        launch_a_cluster(cluster_config_path, node_template_path,
                         last_response_path, launch_result_path, scale)
    elif command in ["start", "stop", "reboot", "terminate"]:
        last_response_path = args[1]
        launch_result_path = args[2]
        if len(args) >= 4:
            num_nodes = args[3]
        else:
            num_nodes = None
        ec2_actions_on_a_cluster(command, last_response_path,
                                 launch_result_path, num_nodes)
    elif command == 'show':
        launch_result_path = args[1]
        with open(launch_result_path, 'r') as fin:
            launch_result = json.load(fin)
        for node in launch_result:
            print(f'{node["name"]}: {node["public_ip"]}')

    elif command in ['free_bandwidth', 'limit_bandwidth']:
        last_response_path = args[1]
        launch_result_path = args[2]
        local_private_key_path = args[3]
        dev_path = args[4]
        if command == "limit_bandwidth":
            ec2_config_path = args[5]
            node_list, _, _ = parse_cluster_config(
                cluster_config_path=ec2_config_path,
                scale=False
            )
            node_dict = {n['name']: n for n in node_list}

        with open(launch_result_path, 'r') as fin:
            launch_result = json.load(fin)

        execution_plan_list = []
        for idx, simplified_response in enumerate(launch_result):
            remote_template = {
                'username': 'ubuntu',
                'key_filename': local_private_key_path
            }
            name = simplified_response['name']
            public_ip = simplified_response['public_ip']

            execution_sequence = []

            if command == "free_bandwidth":
                remote_template.update({
                    'commands': [
                        f"cd {dev_path} "
                        "&& bash bandwidth.sh clean_limit"
                    ]
                })
                execution_sequence = [
                    ('remote', remote_template),
                    ('prompt', [f'Removed limits on bandwidth of node '
                                f'{name} ({public_ip}).'])
                ]
            elif command == "limit_bandwidth":
                instance_dict = node_dict[name]
                if not "upload_kbps" in instance_dict \
                        and not "download_kbps" in instance_dict:
                    continue
                if "upload_kbps" in instance_dict:
                    upload_kbps = instance_dict["upload_kbps"]
                    if "download_kbps" in instance_dict:
                        download_kbps = instance_dict["download_kbps"]
                        remote_template.update({
                            'commands': [
                                f"cd {dev_path} "
                                f"&& bash bandwidth.sh limit_both "
                                f"{upload_kbps} {download_kbps}"
                            ]
                        })
                    else:
                        remote_template.update({
                            'commands': [
                                f"cd {dev_path} "
                                f"&& bash bandwidth.sh limit_upload "
                                f"{upload_kbps} {upload_kbps}"
                            ]
                        })
                else:
                    download_kbps = instance_dict["download_kbps"]
                    remote_template.update({
                        'commands': [
                            f"cd {dev_path} "
                            f"&& bash bandwidth.sh limit_upload "
                            f"{download_kbps}"
                        ]
                    })
                execution_sequence = [
                    ('remote', remote_template),
                    ('prompt', [f'Set limits on bandwidth of node '
                                f'{name} ({public_ip}).'])
                ]

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


if __name__ == '__main__':
    main(sys.argv[1:])
