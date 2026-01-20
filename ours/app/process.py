import numpy as np
import pyvista as pv
import os
import argparse
import warnings

# 忽略 PyVista 警告
warnings.filterwarnings("ignore", module="pyvista")

DATA_ROOT = "../../data"
CONTOUR_VALUE = 0.5 

def process_patient_geometry(patient_id):
    print(f"--- Processing Patient: {patient_id} (Geometry Only) ---")
    patient_folder = os.path.join(DATA_ROOT, patient_id)
    output_dir = os.path.join(patient_folder, "vtp")
    os.makedirs(output_dir, exist_ok=True)

    # 定義要處理的檔案清單
    tasks = [
        ("mask_multi.npy", "mask_multi.vtp"),
        ("mask_multi_clean.npy", "mask_multi_clean.vtp"),      
        ("mask_binary.npy", "mask_binary.vtp"),
        ("mask_binary_clean.npy", "mask_binary_clean.vtp"),
        ("teeth_clean.npy", "teeth_clean.vtp"),
        ("teeth_lower.npy", "teeth_lower.vtp"),
        ("teeth_upper.npy", "teeth_upper.vtp")
    ]
    
    for npy_name, vtp_name in tasks:
        npy_path = os.path.join(patient_folder, npy_name)
        vtp_path = os.path.join(output_dir, vtp_name)
        
        if not os.path.exists(npy_path):
            continue

        try:
            # 1. 讀取資料
            vol = np.load(npy_path)
            if vol.ndim == 4: vol = vol[..., 0]
            vol = np.flip(vol.transpose(2, 1, 0), 0) # 座標轉換
            
            # 2. 準備合併的容器
            meshes_to_merge = []
            unique_vals = np.unique(vol)

            # 遍歷感興趣的類別 (Class 1: 骨頭, Class 2: 牙齒)
            # 這裡不再區分顏色，僅提取幾何形狀
            target_classes = [1, 2]
            
            for class_id in target_classes:
                if class_id in unique_vals:
                    # 提取等值面
                    grid = pv.wrap((vol == class_id).astype(np.uint8))
                    mesh = grid.contour([CONTOUR_VALUE], method='marching_cubes')
                    
                    if mesh.n_points > 0:
                        # [關鍵] 清除所有資料欄位，只保留點與面
                        mesh.clear_data()
                        meshes_to_merge.append(mesh)

            # 3. 合併網格
            if not meshes_to_merge:
                print(f"  [Warning] No mesh generated for {npy_name}")
                continue
                
            final_mesh = meshes_to_merge[0]
            if len(meshes_to_merge) > 1:
                for i in range(1, len(meshes_to_merge)):
                    final_mesh = final_mesh + meshes_to_merge[i]

            # 4. 後處理 (法線計算，確保光影正確)
            final_mesh.compute_normals(inplace=True, auto_orient_normals=True)
            
            # 5. 存檔 (純幾何)
            final_mesh.save(vtp_path, binary=True)
            print(f"  [Success] Saved: {vtp_name}")

        except Exception as e:
            print(f"  [Error] {npy_name}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient", type=str, help="指定病人ID")
    args = parser.parse_args()

    if args.patient:
        process_patient_geometry(args.patient)
    else:
        patients = [d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))]
        patients.sort()
        for p in patients:
            process_patient_geometry(p)