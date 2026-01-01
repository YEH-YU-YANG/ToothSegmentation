import os
import shutil
import subprocess
import sys

from src.config import load_config

config = load_config('configs/config.toml')

log_dir = os.path.join('logs', config.experiment)

if os.path.exists(log_dir):
    raise FileExistsError(f'log folder "{log_dir}" already exists')

os.mkdir(log_dir)
split_filename = f'{config.split_filename}.json'
shutil.copyfile(os.path.join('splits', split_filename), os.path.join(log_dir, os.path.basename(split_filename)))
shutil.copyfile('configs/config.toml', os.path.join(log_dir, 'config.toml'))

for fold in range(1, config.num_folds + 1):
    try:
        command = [sys.executable, '-m', 'scripts.train', '--fold', str(fold)]
        subprocess.run(command, check=True)
    except (Exception, KeyboardInterrupt):
        exit(1)
