import streamlit as st
import pyvista as pv
from stpyvista import stpyvista
import os
import numpy as np

# ================= 系統設定 =================
# 嘗試啟動虛擬螢幕，解決無頭伺服器渲染問題
try:
    pv.start_xvfb()
except Exception:
    pass

st.set_page_config(layout="wide", page_title="3D Dental Viewer (Robust)")
DATA_ROOT_DIR = os.path.join("..", "..", "data")

# ================= 核心函數 (防彈版讀取) =================

@st.cache_resource(show_spinner="Loading Model...")
def load_mesh_file(path, simplify=True):
    """
    防彈版讀取：
    1. 自動修復壞掉的網格 (clean/triangulate)
    2. 簡化失敗時自動回滾
    3. 確保有點才回傳
    """
    if not os.path.exists(path): return None
    try:
        # 1. 基礎讀取
        mesh = pv.read(path)
        
        # [修復] 清除孤立點、NaN點，並強制轉為三角網格 (解決顯示不出來的主因)
        mesh = mesh.clean()
        mesh = mesh.triangulate()
        
        # 檢查是否為空
        if mesh.n_points == 0 or mesh.n_cells == 0:
            print(f"[Warning] Empty mesh: {path}")
            return None

        # 2. 補法線 (確保光影)
        if mesh.point_data.get("Normals") is None:
            mesh.compute_normals(inplace=True, auto_orient_normals=True)

        # 3. 安全簡化 (Safe Decimation)
        target_cells = 50000 
        # 只有當面數真的太多的時候才簡化
        if simplify and mesh.n_cells > target_cells:
            try:
                # 備份
                original_mesh = mesh.copy()
                
                target_reduction = 1 - (target_cells / mesh.n_cells)
                # 限制最大簡化率，避免把模型縮沒了 (例如不超過 90%)
                target_reduction = min(target_reduction, 0.9)
                
                mesh.decimate(target_reduction, inplace=True)
                
                # [檢查] 如果簡化後點太少，認定失敗，還原
                if mesh.n_points < 10:
                    print(f"[Warning] Over-decimated {path}. Reverting.")
                    mesh = original_mesh
            except Exception as e:
                print(f"[Error] Decimation failed for {path}: {e}. Using original.")
                # 簡化報錯時，直接用原檔，不要回傳 None
                pass 
            
        return mesh
    except Exception as e:
        print(f"[Critical Error] Failed to load {path}: {e}")
        return None

def get_patient_list():
    if not os.path.exists(DATA_ROOT_DIR): return []
    p_list = [d for d in os.listdir(DATA_ROOT_DIR) if os.path.isdir(os.path.join(DATA_ROOT_DIR, d))]
    p_list.sort()
    return p_list

# ================= 繪圖邏輯 (安全渲染) =================

def plot_mesh_widget(mesh, key_id, title, color_style="default"):
    # [檢查] 再次確認網格有效性
    if mesh is None or mesh.n_points == 0:
        st.warning(f"{title}: 載入失敗或模型為空")
        return

    st.markdown(f"**{title}**")
    
    # 建立繪圖器
    plotter = pv.Plotter(window_size=[400, 400], off_screen=True)
    plotter.background_color = "#303030"
    
    # 關閉座標軸與邊框 (減少渲染錯誤機率)
    # plotter.show_axes() 
    
    lighting_kwargs = {
        "ambient": 0.4, "diffuse": 0.5, "specular": 0.1, "smooth_shading": True
    }

    # === 顏色設定 ===
    # if "mask" in key_id or color_style == "mask":
    #     c = "gold"
    #     o = 0.5
    # el
    if "stl" in key_id:
        c = "lightblue"
        o = 1.0
    else:
        c = "ivory"
        o = 1.0

    try:
        plotter.add_mesh(mesh, color=c, opacity=o, **lighting_kwargs)
        plotter.view_isometric()
        plotter.reset_camera()
        
        # [關鍵] 傳送給 Streamlit
        stpyvista(plotter, key=key_id)
    except Exception as e:
        st.error(f"Render Error: {e}")

# ================= 主畫面 =================
st.sidebar.title("Monitor")
patient_list = get_patient_list()
selected_patient = st.sidebar.selectbox("選擇病人 ID", patient_list) if patient_list else None

if st.sidebar.button("Reload"):
    st.cache_resource.clear()
    st.rerun()

if selected_patient:
    st.title(f"Patient: {selected_patient}")
    current_dir = os.path.join(DATA_ROOT_DIR, selected_patient)
    vtp_dir = os.path.join(current_dir, "vtp")

    tab1, tab2 = st.tabs(["STL", "CBCT"])

    # --- Tab 1: STL ---
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            m = load_mesh_file(os.path.join(current_dir, "ios", "LowerJawScan.stl"))
            plot_mesh_widget(m, f"stl_l_{selected_patient}", "Lower Jaw")
        with col2:
            m = load_mesh_file(os.path.join(current_dir, "ios", "UpperJawScan.stl"))
            plot_mesh_widget(m, f"stl_u_{selected_patient}", "Upper Jaw")

    # --- Tab 2: VTP ---
    with tab2:
        
        col_m1, col_m2 = st.columns(2)
        
        # with col_m1:
        #     path = os.path.join(vtp_dir, "mask_multi.vtp")
        #     m = load_mesh_file(path)
        #     plot_mesh_widget(m, f"vtp_multi_{selected_patient}", "Mask (Multi)", "mask")

        with col_m1:
            path = os.path.join(vtp_dir, "mask_binary.vtp")
            m = load_mesh_file(path)
            plot_mesh_widget(m, f"vtp_bin_{selected_patient}", "Mask (Binary)", "mask")

        st.markdown("---")
        st.markdown("#### 分離結果")
        c1, c2, c3 = st.columns(3)
        with c1:
            m = load_mesh_file(os.path.join(vtp_dir, "teeth_clean.vtp"))
            plot_mesh_widget(m, f"cln_{selected_patient}", "Clean")
        with c2:
            m = load_mesh_file(os.path.join(vtp_dir, "teeth_lower.vtp"))
            plot_mesh_widget(m, f"low_{selected_patient}", "Lower")
        with c3:
            m = load_mesh_file(os.path.join(vtp_dir, "teeth_upper.vtp"))
            plot_mesh_widget(m, f"up_{selected_patient}", "Upper")