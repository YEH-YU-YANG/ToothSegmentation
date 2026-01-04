from argparse import ArgumentParser
from src.downloader import download_experiment

parser = ArgumentParser()
parser.add_argument('experiment', type=str)
parser.add_argument('--source-log-dir', default='tooth_segmentation/logs')
parser.add_argument('--output', default='logs')
group = parser.add_mutually_exclusive_group()
group.add_argument('--ignore', nargs='*', default=['events.out.tfevents.*', 'last.pth'])
group.add_argument('--no-ignore', action='store_true')
args = parser.parse_args()

source_log_dir = args.source_log_dir
experiment_name = args.experiment
output_dir = args.output
ignore_files = args.ignore if not args.no_ignore else []

download_experiment(experiment_name, source_log_dir=source_log_dir, output_dir=output_dir, ignore_files=ignore_files)
