import torch

from rich.progress import BarColumn, DownloadColumn, MofNCompleteColumn, Progress, ProgressColumn, TimeElapsedColumn, TimeRemainingColumn, TextColumn, TransferSpeedColumn
from rich.text import Text

class SpeedColumn(TransferSpeedColumn):
    def render(self, task):
        transfer_speed = super().render(task)
        return Text.assemble(
            Text('( ', 'dim'),
            transfer_speed,
            Text(' )', 'dim')
        )

class RuntimeColumn(ProgressColumn):
    def __init__(self):
        super().__init__()
        self.elapsed = TimeElapsedColumn()
        self.remaining = TimeRemainingColumn()
    def render(self, task):
        elapsed = self.elapsed.render(task)
        remaining = self.remaining.render(task)
        return Text.assemble(
            Text('[ ', 'dim'),
            elapsed,
            Text(' < ', 'dim'),
            remaining,
            Text(' ]', 'dim')
        )

class PercentageColumn(TextColumn):
    def __init__(self):
        super().__init__('[progress.percentage]{task.percentage:>3.0f}%')

class DownloadProgress(Progress):
    def __init__(self):
        super().__init__(
            PercentageColumn(),
            BarColumn(),
            DownloadColumn(),
            SpeedColumn(),
            RuntimeColumn()
        )

class IterationProgress(Progress):
    def __init__(self):
        super().__init__(
            '[progress.description]{task.description}',
            PercentageColumn(),
            BarColumn(),
            MofNCompleteColumn(),
            RuntimeColumn()
        )

def track(sequence, desc=''):
    with IterationProgress() as progress:
        task = progress.add_task(desc, total=len(sequence))
        for item in sequence:
            yield item
            progress.update(task, advance=1)

class Table:
    def __init__(self, headers=None, *rows):
        if headers is not None:
            headers = [self._convert_str(header) for header in headers]
        self.headers = headers
        rows = [[self._convert_str(data) for data in row] for row in rows]
        self.rows = rows
    def add_row(self, row):
        row = [self._convert_str(data) for data in row]
        self.rows.append(row)
    def _convert_str(self, data):
        if isinstance(data, torch.Size):
            return str(tuple(data))
        if isinstance(data, (float, torch.Tensor)):
            return f'{data:.6f}'
        if not isinstance(data, str):
            return str(data)
        return data
    def _display_line(self, tokens, widths):
        left, line, center, right = tokens
        left = left + line
        center = line + center + line
        right = line + right
        print(f'{left}{center.join(line * width for width in widths)}{right}')
    def _display_rows(self, row, widths, align_center=False):
        align = '^' if align_center else '<'
        print(f'│ {" │ ".join(f"{data:{align}{width}}" for data, width in zip(row, widths))} │')
    def display(self):
        columns = zip(*self.rows) if self.headers is None else zip(self.headers, *self.rows)
        widths = [max(len(data) for data in column) for column in columns]
        self._display_line('┌─┬┐', widths)
        if self.headers is not None:
            self._display_rows(self.headers, widths, '^')
            self._display_line('├─┼┤', widths)
        for row in self.rows:
            self._display_rows(row, widths)
        self._display_line('└─┴┘', widths)
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.display()
        return False
