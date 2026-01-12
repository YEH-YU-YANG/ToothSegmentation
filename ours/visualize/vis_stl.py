import pyvista as pv
import os
import math
import sys
import argparse

# 嘗試匯入 PyTorch (符合您的環境習慣)
try:
    import torch
except ImportError:
    pass

# ==========================================
# 1. 參數解析與配置 (Argument Parsing)
# ==========================================

def parse_args():
    parser = argparse.ArgumentParser(description="STL 3D Visualizer for Dental Data")
    parser.add_argument(
        "--patient", 
        type=str, 
        default="52730449", 
        help="輸入病人編號 (Patient ID)"
    )
    return parser.parse_args()

def get_stl_configs(patient_id):
    """
    根據 patient_id 動態生成檔案路徑配置
    """
    # 定義基礎路徑
    data_dir = os.path.join("..", "..", "data", patient_id, "ios")
    
    # 【使用說明】
    # 在這裡定義您想看的 .stl 檔案。
    files_to_view = [
        {
            "label": f"Lower Jaw ({patient_id})",
            "path": os.path.join(data_dir, "LowerJawScan.stl")
        },
        {
            "label": f"Upper Jaw ({patient_id})",
            "path": os.path.join(data_dir, "UpperJawScan.stl")
        },
        # 如果有其他檔案需求，可以在此處擴充
        # {
        #     "label": "Bite Scan",
        #     "path": os.path.join(data_dir, f"{patient_id} BiteScan.stl")
        # },
    ]
    return files_to_view

# ==========================================
# 2. 視覺化邏輯 (Visualization Logic)
# ==========================================

class STLConfigVisualizer:
    def __init__(self, config_list):
        self.config_list = config_list
        self.valid_items = self._validate_inputs()

    def _validate_inputs(self):
        """
        檢查設定清單中的檔案是否存在。
        """
        valid = []
        print(f"正在檢查檔案路徑...")
        for item in self.config_list:
            path = item['path']
            label = item['label']
            
            if os.path.exists(path):
                valid.append(item)
                print(f"  [O] 載入: {label}")
            else:
                print(f"  [X] 找不到檔案: {path}")
        return valid

    def _calculate_grid(self, n):
        """計算子視窗排列 (Rows x Cols)"""
        if n == 0: return 0, 0
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        return rows, cols

    def show(self):
        num_files = len(self.valid_items)
        if num_files == 0:
            print("\n錯誤：沒有任何有效的 STL 檔案可供顯示。")
            return

        rows, cols = self._calculate_grid(num_files)
        print(f"\n開啟視窗中... 共有 {num_files} 個模型，排列方式: {rows}x{cols}")

        plotter = pv.Plotter(shape=(rows, cols), window_size=(1400, 900))
        
        for i, item in enumerate(self.valid_items):
            r = i // cols
            c = i % cols
            plotter.subplot(r, c)
            
            path = item['path']
            label = item['label']

            try:
                # 載入模型
                mesh = pv.read(path)
                
                # 加入網格 (PBR 渲染增加質感)
                plotter.add_mesh(
                    mesh, 
                    color='white', 
                    pbr=True, 
                    metallic=0.1, 
                    roughness=0.6
                )
                
                # 加入標籤文字 (左上)
                plotter.add_text(label, font_size=12, position='upper_left')
                
                # 加入檔名資訊
                filename = os.path.basename(path)
                plotter.add_text(filename, font_size=8, position='lower_left', color='grey')
                
                plotter.reset_camera()
                
            except Exception as e:
                print(f"讀取錯誤 {label}: {e}")
                plotter.add_text(f"Error: {e}", position='upper_left', color='red')

        # 同步所有子視窗視角
        if num_files > 1:
            plotter.link_views()
        
        print("視窗已啟動。")
        plotter.show()

# ==========================================
# 3. 執行程式
# ==========================================

if __name__ == "__main__":
    # 1. 解析命令列參數
    args = parse_args()
    
    # 2. 根據參數獲取配置清單
    target_configs = get_stl_configs(args.patient)
    
    # 3. 執行視覺化
    viz = STLConfigVisualizer(target_configs)
    viz.show()