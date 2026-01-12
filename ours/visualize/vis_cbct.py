import sys
import os
import argparse
import time
import gc
import numpy as np
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QFrame,
    QProgressBar
)
from PySide6.QtCore import Qt, QThread, Signal, QObject, QSemaphore
import pyqtgraph.opengl as gl

# 嘗試匯入 PyTorch 進行加速
try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    from scipy import ndimage  # Fallback

# =================配置與參數處理=================
def parse_args():
    parser = argparse.ArgumentParser(description="3D NPY Viewer for Dental Data")
    parser.add_argument(
        "--patient", 
        type=str, 
        default="52730449", 
        help="輸入病人編號 (Patient ID)"
    )
    return parser.parse_args()

class Label:
    TOOTH = 1
    BONE = 2

class Color:
    TOOTH = (235, 223, 180, 255)
    BONE = (212, 161, 230, 75)

class ViewSetting:
    VIEW_WIDTH = 800
    VIEW_HEIGHT = 800
    BACKGROUND = "white"

# ================= GPU 加速運算邏輯 =================

def torch_binary_erosion_3d(input_tensor, kernel_size=3):
    """
    使用 PyTorch Conv3d 模擬 3D Binary Erosion
    """
    device = input_tensor.device
    kernel = torch.ones((1, 1, kernel_size, kernel_size, kernel_size), device=device, dtype=torch.float32)
    padding = kernel_size // 2
    
    # 使用 no_grad 節省梯度計算的記憶體
    with torch.no_grad():
        x = input_tensor.unsqueeze(0).unsqueeze(0).float()
        out = F.conv3d(x, kernel, padding=padding, stride=1)
        target_val = kernel_size ** 3
        result = out >= (target_val - 0.1)
    
    return result.squeeze(0).squeeze(0)

def process_volume_gpu(volume_np):
    """使用 PyTorch 進行 GPU 加速處理 (優化記憶體版)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 這裡不直接 .to(device)，避免在轉型前就佔用顯存
    # 先在 CPU 轉好型別或策略性移動
    
    try:
        with torch.no_grad(): # 關鍵：不計算梯度，大幅節省記憶體
            volume_t = torch.from_numpy(volume_np).to(device)

            # 1. 處理 4D 輸入
            if volume_t.ndim == 4:
                volume_t = volume_t[..., 0]

            # 2. 座標轉換
            volume_t = volume_t.permute(2, 1, 0)
            volume_t = torch.flip(volume_t, [0])

            is_raw_image = torch.max(volume_t) > 10

            rgba_np = None

            if is_raw_image:
                # 為了省記憶體，使用 float16 運算如果硬體支援，這裡保守用 float32
                vol_float = volume_t.float()
                
                # 計算 Percentile
                v_min = torch.quantile(vol_float, 0.01)
                v_max = torch.quantile(vol_float, 0.99)
                
                vol_norm = torch.clamp(vol_float, v_min, v_max)
                # 釋放原始 float tensor
                del vol_float
                
                vol_norm = (vol_norm - v_min) / (v_max - v_min + 1e-5) * 255
                vol_norm = vol_norm.to(torch.uint8)

                # 構建 RGBA
                shape = volume_t.shape
                rgba = torch.zeros((shape[0], shape[1], shape[2], 4), dtype=torch.uint8, device=device)
                
                rgba[..., 0] = vol_norm
                rgba[..., 1] = vol_norm
                rgba[..., 2] = vol_norm
                
                # Alpha 通道
                threshold = 80
                alpha = vol_norm # 引用，不複製
                
                # 建立暫存 mask
                high_mask = alpha >= threshold
                
                # 先將 alpha 設為 0
                rgba[..., 3] = 0 
                
                # 只處理高亮區域
                # 注意：這裡的操作要小心記憶體
                alpha_vals = alpha[high_mask].float() * 0.4
                
                # 寫回 RGBA
                # 利用 masked_scatter 或索引賦值
                rgba[..., 3].masked_scatter_(high_mask, alpha_vals.to(torch.uint8))
                
                # 轉回 CPU Numpy
                rgba_np = rgba.cpu().numpy()

            else:
                # Segmentation Mask 模式
                tooth_mask = (volume_t == 1)
                bone_mask = (volume_t == 2)

                erosion = torch_binary_erosion_3d(bone_mask, kernel_size=3)
                bone_surface = bone_mask & (~erosion)

                shape = volume_t.shape
                rgba = torch.zeros((shape[0], shape[1], shape[2], 4), dtype=torch.uint8, device=device)

                rgba[tooth_mask] = torch.tensor(Color.TOOTH, dtype=torch.uint8, device=device)
                rgba[bone_surface] = torch.tensor(Color.BONE, dtype=torch.uint8, device=device)

                rgba_np = rgba.cpu().numpy()

            # 清理 GPU 記憶體
            del volume_t
            if 'rgba' in locals(): del rgba
            if 'vol_norm' in locals(): del vol_norm
            torch.cuda.empty_cache()
            
            return rgba_np

    except Exception as e:
        # 如果 GPU 出錯，嘗試清理後拋出，讓外面接手切換 CPU
        torch.cuda.empty_cache()
        raise e

def process_volume_cpu(volume):
    """(備用) 原始 CPU 處理邏輯"""
    from scipy import ndimage
    print(" -> 切換至 CPU 模式處理...")
    if volume.ndim == 4:
        volume = volume[..., 0]
    volume = volume.transpose(2, 1, 0)
    volume = np.flip(volume, 0)
    is_raw_image = np.max(volume) > 10

    if is_raw_image:
        vol_norm = volume.astype(np.float32)
        v_min, v_max = np.percentile(vol_norm, [1, 99]) 
        vol_norm = np.clip(vol_norm, v_min, v_max)
        vol_norm = (vol_norm - v_min) / (v_max - v_min + 1e-5) * 255
        vol_norm = vol_norm.astype(np.uint8)
        rgba = np.zeros((*volume.shape, 4), dtype=np.uint8)
        rgba[..., 0:3] = vol_norm[..., np.newaxis]
        threshold = 80 
        alpha = vol_norm.copy()
        alpha[alpha < threshold] = 0
        alpha[alpha >= threshold] = (alpha[alpha >= threshold] * 0.4).astype(np.uint8)
        rgba[..., 3] = alpha
        return rgba
    else:
        tooth_mask = volume == 1
        bone_mask = volume == 2
        erosion = ndimage.binary_erosion(bone_mask)
        bone_surface = bone_mask & (~erosion)
        rgba = np.zeros((*volume.shape, 4), dtype=np.uint8)
        rgba[tooth_mask] = Color.TOOTH
        rgba[bone_surface] = Color.BONE
        return rgba

def process_volume_dispatch(volume):
    """根據環境選擇加速版或 CPU 版"""
    if HAS_TORCH:
        try:
            return process_volume_gpu(volume)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("[GPU OOM] 顯存不足，清理後切換至 CPU。")
                torch.cuda.empty_cache()
                return process_volume_cpu(volume)
            else:
                print(f"[GPU Error] {e}")
                return process_volume_cpu(volume)
    else:
        return process_volume_cpu(volume)

# ================= 多執行緒載入器 =================

class DataLoaderWorker(QThread):
    data_ready = Signal(str, object, tuple) 

    def __init__(self, path, label, semaphore):
        super().__init__()
        self.path = path
        self.label = label
        self.semaphore = semaphore # 接收信號量

    def run(self):
        # 關鍵修改：獲取許可證 (Acquire)
        # 只有獲得許可的執行緒才能開始載入大檔案與使用 GPU
        self.semaphore.acquire()
        
        try:
            # print(f"[{self.label}] 開始處理...")
            # 1. IO Load
            raw_data = np.load(self.path)
            
            # 2. Processing (GPU/CPU)
            processed_data = process_volume_dispatch(raw_data)
            
            # 3. Emit Result
            shape = processed_data.shape[:3]
            self.data_ready.emit(self.label, processed_data, shape)
            
            # 主動刪除大變數並回收
            del raw_data
            del processed_data
            gc.collect()

        except Exception as e:
            print(f"Error loading {self.label}: {e}")
            self.data_ready.emit(self.label, None, (0,0,0))
            
        finally:
            # 關鍵修改：釋放許可證 (Release)
            # 讓下一個排隊的執行緒開始工作
            if HAS_TORCH:
                torch.cuda.empty_cache()
            self.semaphore.release()
            # print(f"[{self.label}] 處理完成，釋放資源。")

# ================= 視圖元件 =================

class SyncGLView(gl.GLViewWidget):
    def __init__(self, title, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setBackgroundColor(ViewSetting.BACKGROUND)
        self.setMinimumSize(400, 400)
        self.sync_targets = [] 
        self.item = None
        self.volume_shape = None
        self.title = title

    def set_data(self, processed_data, shape):
        if processed_data is None:
            return

        self.volume_shape = shape
        width, height, depth = self.volume_shape

        if self.item is not None:
            self.removeItem(self.item)

        self.item = gl.GLVolumeItem(processed_data, smooth=True)
        self.item.translate(-width / 2, -height / 2, -depth / 2)
        self.addItem(self.item)
        self._reset_camera()
        self.update()

    def _reset_camera(self):
        if self.volume_shape:
            width, _, depth = self.volume_shape
            distance = max(width, depth) * 1.5
            self.setCameraPosition(distance=distance, elevation=90, azimuth=90)

    def sync_camera_from(self, opts):
        self.opts["center"] = opts["center"]
        self.opts["distance"] = opts["distance"]
        self.opts["azimuth"] = opts["azimuth"]
        self.opts["elevation"] = opts["elevation"]
        self.update()

    def mouseMoveEvent(self, ev):
        super().mouseMoveEvent(ev)
        for target in self.sync_targets:
            target.sync_camera_from(self.opts)

    def wheelEvent(self, ev):
        super().wheelEvent(ev)
        for target in self.sync_targets:
            target.sync_camera_from(self.opts)

class DynamicViewerWindow(QMainWindow):
    def __init__(self, patient_id):
        super().__init__()
        self.patient_id = patient_id
        self.setWindowTitle(f"NPY 3D Viewer - Patient: {patient_id} (Serialized Loading)")
        self.resize(1600, 900)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.main_layout = QHBoxLayout(central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        self.views = {} 
        self.threads = [] 
        
        # 狀態列
        self.status_bar = self.statusBar()
        self.loading_progress = QProgressBar()
        self.loading_progress.setFixedWidth(200)
        self.status_bar.addPermanentWidget(self.loading_progress)
        
        # 關鍵：限制同時只能有 1 個執行緒進行「載入+GPU運算」
        # 設定為 1 可以最大程度保證不 OOM
        self.gpu_semaphore = QSemaphore(1) 

        self.init_views()

    def init_views(self):
        data_dir = os.path.join("..", "..", "data", self.patient_id)
        
        file_configs = [
            # {"label": "牙齒", "path": os.path.join(data_dir, "mask_binary.npy")},
            # {"label": "牙齒 (後處理)", "path": os.path.join(data_dir, "teeth_clean.npy")},
            {"label": "牙齒", "path": os.path.join(data_dir, "teeth_lower.npy")},
            {"label": "牙齒 + 骨頭", "path": os.path.join(data_dir, "teeth_upper.npy")}
        ]

        valid_configs = []
        for config in file_configs:
            if os.path.exists(config["path"]):
                valid_configs.append(config)
            else:
                print(f"[Warning] 找不到路徑: {config['path']}")

        if not valid_configs:
            lbl = QLabel("無有效資料")
            lbl.setAlignment(Qt.AlignCenter)
            self.main_layout.addWidget(lbl)
            return

        self.total_files = len(valid_configs)
        self.loaded_count = 0
        self.loading_progress.setRange(0, self.total_files)

        # 1. 建立 UI
        for i, config in enumerate(valid_configs):
            v_layout = QVBoxLayout()
            v_layout.setContentsMargins(0, 0, 0, 0)
            v_layout.setSpacing(0)

            lbl_title = QLabel(f"{config['label']} (排隊中...)")
            lbl_title.setAlignment(Qt.AlignCenter)
            lbl_title.setStyleSheet("font-weight: bold; font-size: 14px; padding: 4px; background-color: #eee; border-bottom: 1px solid #ccc;")
            lbl_title.setFixedHeight(35)

            view = SyncGLView(config["label"])
            
            v_layout.addWidget(lbl_title, 0)
            v_layout.addWidget(view, 1)
            self.main_layout.addLayout(v_layout)

            self.views[config["label"]] = {
                "view": view, 
                "label_widget": lbl_title,
                "config": config
            }

            if i < len(valid_configs) - 1:
                line = QFrame()
                line.setFrameShape(QFrame.VLine)
                line.setFrameShadow(QFrame.Plain)
                line.setLineWidth(1)
                self.main_layout.addWidget(line)

        # 2. 啟動背景執行緒 (帶有 Semaphore)
        print(f"開始排程載入 {self.total_files} 個檔案 (序列模式)...")
        for config in valid_configs:
            # 將 Semaphore 傳入 Worker
            worker = DataLoaderWorker(config["path"], config["label"], self.gpu_semaphore)
            worker.data_ready.connect(self.on_data_loaded)
            self.threads.append(worker)
            worker.start()

    def on_data_loaded(self, label, data, shape):
        if label in self.views:
            target = self.views[label]
            view = target["view"]
            lbl_widget = target["label_widget"]

            if data is not None:
                view.set_data(data, shape)
                lbl_widget.setText(f"{label} ({self.patient_id})")
                print(f"[完成] {label} 載入完畢。")
            else:
                lbl_widget.setText(f"{label} (載入失敗)")
        
        self.loaded_count += 1
        self.loading_progress.setValue(self.loaded_count)
        
        if self.loaded_count == self.total_files:
            self.loading_progress.hide()
            self.status_bar.showMessage("所有資料載入完成", 5000)
            self.setup_sync()

    def setup_sync(self):
        view_list = [v["view"] for v in self.views.values()]
        if len(view_list) >= 2:
            print("開啟多視窗視角同步。")
            for i in range(len(view_list)):
                for j in range(len(view_list)):
                    if i != j:
                        view_list[i].sync_targets.append(view_list[j])

if __name__ == "__main__":
    # 設定 PyTorch 記憶體分配策略 (可選，防止碎片化)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    args = parse_args()
    app = QApplication(sys.argv)
    
    if HAS_TORCH and torch.cuda.is_available():
        print(f"GPU 加速已啟用: {torch.cuda.get_device_name(0)}")
    else:
        print("注意: 未偵測到 PyTorch 或 CUDA，將使用 CPU 模式。")

    window = DynamicViewerWindow(args.patient)
    window.showMaximized()
    sys.exit(app.exec())