import subprocess


def git_status():
    process = subprocess.Popen(['git', 'rev-parse', 'HEAD'],
                               shell=False,
                               stdout=subprocess.PIPE)
    git_head_hash = process.communicate()[0].strip().decode("utf-8")
    git_head_hash_short = git_head_hash[:7]

    process = subprocess.Popen(['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                               shell=False,
                               stdout=subprocess.PIPE)
    git_branch_name = process.communicate()[0].strip().decode("utf-8")

    return git_branch_name, git_head_hash_short