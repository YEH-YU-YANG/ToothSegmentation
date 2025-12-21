import os
import numpy

from PySide6.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QComboBox
from PySide6.QtCore import Signal, QThread
from PySide6.QtGui import QVector3D
from pyqtgraph.opengl import GLViewWidget, GLVolumeItem
from scipy import ndimage
from src.dataset import get_fold

class Label:
    TOOTH = 1
    BONE = 2

class Color:
    TOOTH = (235, 223, 180, 255)
    BONE = (212, 161, 230, 75)

class ViewSetting:
    VIEW_WIDTH = 640
    VIEW_HEIGHT = 640

    BACKGROUND = 'white'

class DataManager:
    def __init__(self, experiment_name, patient_mapping, base_output_dir='outputs'):
        self.experiment_name = experiment_name
        self.patient_mapping = {
            patient: fold
            for patient, fold in sorted(patient_mapping.items(), key=lambda x: int(x[0][5:]))
        }
        self.patients = list(self.patient_mapping.keys())
        self.base_output_dir = base_output_dir

    def load_data(self, patient):
        fold = self.patient_mapping[patient]
        base_dir = os.path.join(self.base_output_dir, self.experiment_name, f'Fold_{fold}', patient)

        predict_volume = numpy.load(os.path.join(base_dir, 'volume.npy'))
        ground_truth_volume = numpy.load(os.path.join(base_dir, 'ground_truth.npy'))

        return predict_volume, ground_truth_volume

class VolumeLoader(QThread):
    finished = Signal(object, object) # predict volume, ground truth volume
    def __init__(self, data_manager, patient):
        super().__init__()
        self.data_manager = data_manager
        self.patient = patient

    def run(self):
        predict_volume, ground_truth_volume = self.data_manager.load_data(self.patient)
        predict_volume = self.process_volume(predict_volume)
        ground_truth_volume = self.process_volume(ground_truth_volume)
        self.finished.emit(predict_volume, ground_truth_volume)

    def process_volume(self, volume):
        volume = volume.transpose(2, 1, 0) # (W, H, Z)
        volume = numpy.flip(volume, 2) # upside down

        tooth_mask = volume == Label.TOOTH # (W, H, Z)
        bone_mask = volume == Label.BONE # (W, H, Z)
        erosion = ndimage.binary_erosion(bone_mask)
        bone_surface = bone_mask & (~erosion) # (W, H, Z)

        rgba = numpy.zeros((*volume.shape, 4), dtype=numpy.uint8) # (W, H, Z, 4)
        rgba[tooth_mask] = Color.TOOTH
        rgba[bone_surface] = Color.BONE

        return rgba

class SyncGLView(GLViewWidget):
    viewChanged = Signal(object)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setBackgroundColor(ViewSetting.BACKGROUND)
        self.setFixedSize(ViewSetting.VIEW_WIDTH, ViewSetting.VIEW_HEIGHT)

        self.item = None
        self.volume_shape = None

    def update_volume(self, volume):
        self.volume_shape = volume.shape[:3]
        width, height, depth = self.volume_shape

        if self.item is not None:
            self.removeItem(self.item)

        self.item = GLVolumeItem(volume, smooth=True)
        self.item.translate(-width / 2, -height / 2, -depth / 2)
        self.item.rotate(90, 0, 0, 1)
        self.addItem(self.item)

        self.reset_camera()

    def reset_camera(self):
        width, _, depth = self.volume_shape
        fov = self.opts['fov']
        self.setCameraPosition(
            pos=QVector3D(0, 0, 0),
            distance=max(width, depth) / 2 / numpy.tan(numpy.radians(fov / 2)),
            elevation=0,
            azimuth=0
        )

    def sync_change(self):
        self.viewChanged.emit(self.opts)

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        self.sync_change()

    def wheelEvent(self, event):
        super().wheelEvent(event)
        self.sync_change()

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        self.sync_change()

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        self.sync_change()

    def contextMenuEvent(self, event):
        self.reset_camera()
        self.sync_change()

    def apply_opts(self, options):
        for key in ['azimuth', 'elevation', 'distance', 'center']:
            self.opts[key] = options[key]
        self.update()

class MainWindowUI(QMainWindow):
    def __init__(self):
        super().__init__()

        self.move(0, 0)

        widget = QWidget()
        layout = QHBoxLayout(widget)
        self.setCentralWidget(widget)

        self.patient_selector = QComboBox()
        self.predict_view = SyncGLView()
        self.ground_truth_view = SyncGLView()

        layout.addWidget(self.patient_selector)
        layout.addWidget(self.predict_view)
        layout.addWidget(self.ground_truth_view)

        self.predict_view.viewChanged.connect(self.ground_truth_view.apply_opts)
        self.ground_truth_view.viewChanged.connect(self.predict_view.apply_opts)

class MainWindow(MainWindowUI):
    def __init__(self, data_manager):
        super().__init__()
        self.data_manager = data_manager

        self.setWindowTitle(self.data_manager.experiment_name)

        self.patient_selector.addItems(self.data_manager.patients)
        self.patient_selector.setCurrentIndex(-1)
        self.patient_selector.currentIndexChanged.connect(self.load_patient)

    def load_patient(self, index):
        patient = self.data_manager.patients[index]
        self.patient_selector.setEnabled(False)
        self.thread = VolumeLoader(self.data_manager, patient)
        self.thread.finished.connect(self.on_volume_loaded)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def on_volume_loaded(self, predict_volume, ground_truth_volume):
        self.predict_view.update_volume(predict_volume)
        self.ground_truth_view.update_volume(ground_truth_volume)
        self.patient_selector.setEnabled(True)

def get_patient_fold_mapping(config):
    return {
        patient: fold
        for fold in range(1, config.num_folds + 1)
        for patient in get_fold(config.split_filename, fold)[1]
    }

if __name__ == '__main__':
    from argparse import ArgumentParser
    from PySide6.QtWidgets import QApplication
    from src.config import load_config

    parser = ArgumentParser()
    parser.add_argument('exp', type=str)
    args = parser.parse_args()

    experiment_name = args.exp

    config = load_config(os.path.join('logs', experiment_name, 'config.toml'))
    patient_fold_map = get_patient_fold_mapping(config)

    app = QApplication([])

    data_manager = DataManager(experiment_name, patient_fold_map)
    window = MainWindow(data_manager)

    window.show()

    app.exec()
