import fnmatch
import os
import stat
import posixpath

from dotenv import load_dotenv
from paramiko import SSHClient, SSHException
from src.console import DownloadProgress

class SFTPDownloader:
    def __init__(self, hostname, port, username, password):
        self.hostname = hostname
        self.port = port
        self.username = username
        self.password = password

        self.client = None
        self.sftp = None

    def _walk_dir(self, remote_dir='.', relative_dir='', ignore_files=None):
        ignore_files = ignore_files or []

        total_files = []
        total_size = 0
        for item in self.sftp.listdir_attr(remote_dir):
            path = posixpath.join(remote_dir, item.filename)
            relative_path = posixpath.join(relative_dir, item.filename)

            if stat.S_ISLNK(item.st_mode):
                print(f'\033[33m[WARNING]\033[0m {path} is a symbolic link, will be ignored.')
                continue
            if any(fnmatch.fnmatch(item.filename, pattern) for pattern in ignore_files):
                continue

            if stat.S_ISDIR(item.st_mode):
                files, size = self._walk_dir(path, relative_path, ignore_files)
                total_files.extend(files)
                total_size += size
            else:
                size = item.st_size or 0
                total_files.append((path, relative_path))
                total_size += size

        return total_files, total_size

    def download_dir(self, source_log_dir, experiment_name, output_dir, ignore_files=None):
        source_dir = posixpath.join(source_log_dir, experiment_name)
        output_dir = os.path.join(output_dir, experiment_name)
        files, size = self._walk_dir(source_dir, ignore_files=ignore_files)

        with DownloadProgress() as progress:
            task = progress.add_task('download', total=size)
            for source_path, relative_path in files:
                local_path = os.path.join(output_dir, *relative_path.split('/'))
                os.makedirs(os.path.dirname(local_path), exist_ok=True)

                progress.update(task, current=relative_path)

                current = {'size': 0}
                def callback(transferred, total):
                    delta = transferred - current['size']
                    if delta > 0:
                        progress.update(task, advance=delta)
                        current['size'] = transferred

                self.sftp.get(source_path, local_path, callback=callback)

    def __enter__(self):
        self.client = SSHClient()
        self.client.load_system_host_keys()
        self.client.connect(self.hostname, self.port, self.username, self.password)
        self.sftp = self.client.open_sftp()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.sftp is not None:
            self.sftp.close()
        if self.client is not None:
            self.client.close()
        return False

def get_env():
    if not os.path.exists('.env'):
        raise FileNotFoundError(
            '.env file not found.\033[0m\n'
            'This file is required for SFTP connection.\n\n'
            'Copy ".env.example" to ".env" and fill in your credentials.\n'
            'Example:\n'
            '\033[90mSFTP_HOSTNAME = your.server.address\n'
            'SFTP_PORT = 22\n'
            'SFTP_USERNAME = your_username\n'
            'SFTP_PASSWORD = your_password\033[0m\n'
            'Please refer to README.md for more details.'
        )

    load_dotenv()
    return {
        'hostname': os.getenv('SFTP_HOSTNAME'),
        'port': int(os.getenv('SFTP_PORT')),
        'username': os.getenv('SFTP_USERNAME'),
        'password': os.getenv('SFTP_PASSWORD')
    }

def download_experiment(experiment_name, *, source_log_dir='tooth_segmentation/logs', output_dir='logs', ignore_files=None):
    ignore_files = ignore_files or ['events.out.tfevents.*', 'last.pth']

    env = get_env()
    hostname = env['hostname']
    port = env['port']
    username = env['username']
    password = env['password']

    try:
        with SFTPDownloader(hostname, port, username, password) as downloader:
            downloader.download_dir(source_log_dir, experiment_name, output_dir, ignore_files)
    except SSHException as exception:
        raise RuntimeError(
            f'\033[1;35m{exception.__class__.__name__}\033[0m: \033[35m{exception}\033[0m\n'
            'Try using the following command to connect to the server first:\n'
            f'\033[36mssh {username}@{hostname} -p {port}\033[0m'
        ) from None
    except FileNotFoundError:
        raise FileNotFoundError(f'\033[35mExperiment "{experiment_name}" not found in server.\033[0m') from None

def ensure_experiment_exists(experiment_name, *, source_log_dir='tooth_segmentation/logs', output_dir='logs', ignore_files=None):
    if not os.path.exists(os.path.join(output_dir, experiment_name)):
        print(f'Experiment "{experiment_name}" not found. Downloading from the server...')
        download_experiment(experiment_name, source_log_dir=source_log_dir, output_dir=output_dir, ignore_files=ignore_files)
