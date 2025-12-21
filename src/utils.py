import pandas
import torch

from tbparse import SummaryReader
from torch import optim
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TimeElapsedColumn, TimeRemainingColumn

class Table:
    def __init__(self, headers=None, *rows):
        if headers is not None:
            headers = [self.convert_str(header) for header in headers]
        self.headers = headers
        rows = [[self.convert_str(data) for data in row] for row in rows]
        self.rows = rows
    def add_row(self, row):
        row = [self.convert_str(data) for data in row]
        self.rows.append(row)
    def convert_str(self, data):
        if isinstance(data, torch.Size):
            return str(tuple(data))
        if isinstance(data, (float, torch.Tensor)):
            return f'{data:.6f}'
        if not isinstance(data, str):
            return str(data)
        return data
    def display_line(self, tokens, widths):
        left, line, center, right = tokens
        left = left + line
        center = line + center + line
        right = line + right
        print(f'{left}{center.join(line * width for width in widths)}{right}')
    def display_rows(self, row, widths, align_center=False):
        align = '^' if align_center else '<'
        print(f'│ {" │ ".join(f"{data:{align}{width}}" for data, width in zip(row, widths))} │')
    def display(self):
        columns = zip(*self.rows) if self.headers is None else zip(self.headers, *self.rows)
        widths = [max(len(data) for data in column) for column in columns]
        self.display_line('┌─┬┐', widths)
        if self.headers is not None:
            self.display_rows(self.headers, widths, '^')
            self.display_line('├─┼┤', widths)
        for row in self.rows:
            self.display_rows(row, widths)
        self.display_line('└─┴┘', widths)

class Summary:
    def __init__(self, scalars):
        self.scalars = scalars
    def __getitem__(self, query):
        if query == 'step' or query == 'value':
            return self.scalars[query]
        if query.startswith('Fold'):
            mask = self.scalars['fold'] == query
            drop_column = 'fold'
        elif query == 'Train' or query == 'Val':
            mask = self.scalars['mode'] == query
            drop_column = 'mode'
        else:
            mask = self.scalars['metric'] == query
            drop_column = 'metric'
        scalars = self.scalars[mask]
        scalars = scalars.drop(columns=drop_column)
        return Summary(scalars)
    def __repr__(self):
        return repr(self.scalars)
    def __getattr__(self, attribute):
        return getattr(self.scalars, attribute)
    def __len__(self):
        return len(self.scalars)

def load_summary(log_dir):
    reader = SummaryReader(log_dir, extra_columns={'dir_name'})
    scalars = reader.scalars

    dir_names = scalars['dir_name'].unique()
    dir_names = pandas.DataFrame({'dir_name': dir_names})

    pattern = r'(?P<fold>Fold_\d+)/(?P<metric>.+)_(?P<mode>[^/_]+)$'
    extracted = dir_names['dir_name'].str.extract(pattern)

    dir_names = pandas.concat([dir_names, extracted], axis=1)
    scalars = scalars.merge(dir_names, on='dir_name')

    scalars = scalars[['fold', 'metric', 'mode', 'step', 'value']]
    scalars[['fold', 'metric', 'mode']] = scalars[['fold', 'metric', 'mode']].astype('category')

    return Summary(scalars)

def track(sequence, desc=''):
    columns = [
        '[progress.description]{task.description}',
        '[progress.percentage]{task.percentage:>3.0f}%',
        BarColumn(),
        MofNCompleteColumn(),
        '[',
        TimeElapsedColumn(),
        '<',
        TimeRemainingColumn(),
        ']'
    ]
    with Progress(*columns) as progress:
        task = progress.add_task(desc, total=len(sequence))
        for item in sequence:
            yield item
            progress.update(task, advance=1)

OPTIMIZERS = {
    'Adam': optim.Adam
}

def get_optimizer(model, config):
    return OPTIMIZERS[config.optimizer.name](model.parameters(), **config.optimizer.parameters)

if __name__ == '__main__':
    import time

    for _ in track(range(20)):
        time.sleep(0.25)

    summary = load_summary('logs/UNet_baseline')
    table = Table(['Fold', 'mIoU'])
    for fold in range(1, 5):
        val_metric = summary[f'Fold_{fold}']['mIoU']['Val']['value'].max()
        table.add_row([fold, val_metric])
    table.display()
