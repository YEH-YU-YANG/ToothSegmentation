import streamlit as st
import pyvista as pv
from stpyvista import stpyvista
import os
import pandas as pd
from pathlib import Path

# ================= 系統設定 =================
try:
    pv.start_xvfb()
except Exception:
    pass

st.set_page_config(layout="wide", page_title="Professional Dental 3D Viewer")

# 設定資料根目錄
DATA_ROOT = Path("/home/p76144736/yeh/paper_related_project/ToothSegmentation/data")
LOG_FILE = "data_quality_log.csv"

# ================= 核心邏輯 =================

@st.cache_resource(show_spinner="正在優化網格...")
def load_and_preprocess_mesh(path, target_cells=60000):
    """
    專業級預處理：包含清理、三角化、法線計算與減面。
    """
    # 轉換為 Path 物件以確保相容性
    path = Path(path)
    
    if not path.exists():
        return None
    try:
        mesh = pv.read(str(path))
        # 清理無效幾何與孤立點
        mesh = mesh.clean().triangulate()
        
        if mesh.n_points == 0:
            return None

        # 自動校正法線方向
        mesh.compute_normals(inplace=True, auto_orient_normals=True)

        # 智能減面
        if mesh.n_cells > target_cells:
            reduction = min(1 - (target_cells / mesh.n_cells), 0.85)
            mesh = mesh.decimate(reduction)
            
        return mesh
    except Exception as e:
        st.error(f"讀取錯誤 {path.name}: {e}")
        return None

def save_quality_feedback(patient_id, status):
    """紀錄資料品質"""
    new_data = pd.DataFrame([[patient_id, status]], columns=["PatientID", "Status"])
    if os.path.exists(LOG_FILE):
        df = pd.read_csv(LOG_FILE)
        # 更新邏輯：保留最後一筆紀錄
        df = pd.concat([df, new_data]).drop_duplicates(subset=["PatientID"], keep="last")
    else:
        df = new_data
    df.to_csv(LOG_FILE, index=False)
    st.sidebar.success(f"已記錄: {patient_id} 為 {status}")

# ================= 渲染組件 =================

def render_3d_view(mesh, key, color="ivory", opacity=1.0):
    """封裝 Plotter 物件，確保資源及時釋放"""
    if mesh is None:
        st.info("暫無模型資料")
        return

    plotter = pv.Plotter(window_size=[400, 400], off_screen=True)
    plotter.background_color = "#1e1e1e"
    
    plotter.add_mesh(
        mesh, 
        color=color, 
        opacity=opacity, 
        smooth_shading=True,
        specular=0.5,
        ambient=0.3
    )
    plotter.view_isometric()
    plotter.reset_camera()
    
    stpyvista(plotter, key=key)
    plotter.close() # 釋放顯存

# ================= 介面佈局 =================

st.sidebar.title("Monitor")

# 獲取病人列表
if DATA_ROOT.exists():
    patient_list = sorted([d for d in os.listdir(DATA_ROOT) if (DATA_ROOT / d).is_dir()])
else:
    patient_list = []
    st.sidebar.error("找不到資料路徑")

selected_patient = st.sidebar.selectbox("患者編號 (Patient ID)", patient_list)

if selected_patient:
    p_path = DATA_ROOT / selected_patient
    vtp_path = p_path / "vtp"

    st.title(f"檢視中: {selected_patient}")
    
    # 快速標記功能
    col_btn1, col_btn2 = st.sidebar.columns(2)
    if col_btn1.button("標記正常"):
        save_quality_feedback(selected_patient, "Good")
    if col_btn2.button("標記損壞"):
        save_quality_feedback(selected_patient, "Bad/Artifact")

    tab_stl, tab_vtp = st.tabs(["原始 IOS 掃描", "分割結果"])

    # --- Tab 1: STL 顯示 ---
    with tab_stl:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Lower Jaw")
            m = load_and_preprocess_mesh(p_path / "ios" / "LowerJawScan.stl")
            render_3d_view(m, f"stl_l_{selected_patient}", color="lightblue")
        with c2:
            st.subheader("Upper Jaw")
            m = load_and_preprocess_mesh(p_path / "ios" / "UpperJawScan.stl")
            render_3d_view(m, f"stl_u_{selected_patient}", color="lightpink")

    # --- Tab 2: VTP 顯示 ---
    with tab_vtp:
        st.subheader("Mask 檔案比對")
        
        # 使用 Expander 讓介面更整潔，且修正了檔案讀取邏輯與 Key 重複問題
        with st.expander("展開查看：原始 vs 清洗後 Mask", expanded=True):
            r1_c1, r1_c2 = st.columns(2)
            
            # 原始資料區塊
            with r1_c1:
                st.markdown("**原始 Binary Mask**")
                m = load_and_preprocess_mesh(vtp_path / "mask_binary.vtp")
                # Key 加上 _raw 避免衝突
                render_3d_view(m, f"v_bin_raw_{selected_patient}", color="gold", opacity=0.7)
                
                st.markdown("**原始 Multi-class Mask**")
                m = load_and_preprocess_mesh(vtp_path / "mask_multi.vtp")
                render_3d_view(m, f"v_mul_raw_{selected_patient}", color="cyan", opacity=0.7)

            # 清洗後資料區塊
            with r1_c2:
                st.markdown("**清洗後 Binary Mask**")
                m = load_and_preprocess_mesh(vtp_path / "mask_binary_clean.vtp")
                # Key 加上 _clean 避免衝突
                render_3d_view(m, f"v_bin_clean_{selected_patient}", color="gold", opacity=0.7)

                st.markdown("**清洗後 Multi-class Mask**")
                m = load_and_preprocess_mesh(vtp_path / "mask_multi_clean.vtp")
                render_3d_view(m, f"v_mul_clean_{selected_patient}", color="cyan", opacity=0.7)

        st.markdown("---")
        st.subheader("牙齒分離結果")
        
        c1, c2, c3 = st.columns(3)
        
        vtp_files = [
            ("teeth_clean.vtp", "Total Clean", "ivory"),
            ("teeth_lower.vtp", "Lower Teeth", "lightblue"),
            ("teeth_upper.vtp", "Upper Teeth", "lightpink")
        ]
        
        cols = [c1, c2, c3]
        for i, (fname, label, color) in enumerate(vtp_files):
            with cols[i]:
                st.caption(label)
                m = load_and_preprocess_mesh(vtp_path / fname)
                # 使用唯一的 Key 格式
                render_3d_view(m, f"sep_{i}_{selected_patient}", color=color)