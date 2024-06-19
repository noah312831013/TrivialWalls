import os
import subprocess
from tqdm import tqdm  # Assuming you want to use tqdm for progress tracking
import json

root = "/media/user/WD_BLACK/noah/zind/datasets"
indices = os.listdir(root)
with open(root+'zind_partition.json') as file:
    data = json.load(file)
test = data['test']
for index in tqdm(test):
    data_dir = os.path.join(root, index, 'panos_aligned')
    otuput_dir = os.path.join('src/output',index)
    # Constructing the command as a list of strings
    command = [
        'python', 'inference.py',
        '--cfg', 'src/config/zind.yaml',
        '--data_glob', data_dir,
        '--output_dir', otuput_dir,
        '--post_processing', 'manhattan'
    ]
    
    try:
        # Running the subprocess command
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running subprocess for {data_dir}: {e}")
