import os
import shutil
import subprocess
import sys

from src.config import Config
from src.utils import save_config

config = Config()

log_dir = os.path.join('logs', config.EXPERIMENT)

if os.path.exists(log_dir):
    raise FileExistsError(f'log folder "{log_dir}" already exists')

os.mkdir(log_dir)
shutil.copyfile(config.SPLIT_FILENAME, os.path.join(log_dir, os.path.basename(config.SPLIT_FILENAME)))
save_config(config, os.path.join(log_dir, 'config.json'))

for fold in range(1, config.NUM_FOLDS + 1):
    try:
        command = [sys.executable, '-m', 'scripts.train', '--fold', str(fold)]
        subprocess.run(command, check=True)
    except (Exception, KeyboardInterrupt):
        exit(1)
