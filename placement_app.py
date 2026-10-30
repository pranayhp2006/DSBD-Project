"""
Placement Analysis System - Full-Stack Streamlit Application
Aligned with Data Science & Big Data Lab Curriculum (CO1, CO2, CO3, CO4)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO, StringIO
import os

# Import core system modules
import auth_manager
from data_processor import PreprocessingPipeline
from ml_models import PlacementClassifierSuite, PackageRegressionSuite
from clustering_engine import StudentClusteringEngine
from skill_gap_analyzer import SkillGapAnalyzer
import report_generator

CUSTOM_CSS = """
<style>
    /* Main container styling */
    .main { background-color: #f8fafc; }
    
    /* Header badges and titles */
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    
    /* Metric Cards */
    .metric-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
        border: 1px solid #e2e8f0;
        text-align: center;
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    }
    .metric-num {
        font-size: 2rem;
        font-weight: 700;
        color: #2563eb;
        margin-bottom: 0.2rem;
    }
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
        color: #64748b;
    }
    
    /* Status Badges */
    .badge-admin {
        background-color: #ede9fe;
        color: #6d28d9;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-student {
        background-color: #e0f2fe;
        color: #0369a1;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    
    /* Section callouts */
    .info-box {
        background-color: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 1rem;
        font-size: 0.92rem;
        color: #1e3a8a;
    }
</style>
"""


# Initialize Session State
def init_session():
    if "user" not in st.session_state:
        st.session_state["user"] = None
    if "raw_df" not in st.session_state:
        # Load default dataset if exists
        default_csv = os.path.join(os.path.dirname(__file__), "students_placement_dataset.csv")
        if os.path.exists(default_csv):
            st.session_state["raw_df"] = pd.read_csv(default_csv)
        else:
            st.session_state["raw_df"] = None
    if "clean_df" not in st.session_state:
        st.session_state["clean_df"] = None
    if "classifier_suite" not in st.session_state:
        st.session_state["classifier_suite"] = None
    if "cls_results" not in st.session_state:
        st.session_state["cls_results"] = None
    if "reg_suite" not in st.session_state:
        st.session_state["reg_suite"] = None
    if "reg_results" not in st.session_state:
        st.session_state["reg_results"] = None
    if "clustering_engine" not in st.session_state:
        st.session_state["clustering_engine"] = None
    if "clustering_results" not in st.session_state:
        st.session_state["clustering_results"] = None
    if "pipeline" not in st.session_state:
        st.session_state["pipeline"] = None
    if "feature_cols" not in st.session_state:
        st.session_state["feature_cols"] = []


# Auto-Train Baseline Pipeline helper
def ensure_models_trained():
    """Trains baseline models if not already present in session."""
    if st.session_state["cls_results"] is not None:
        return

    df = st.session_state["raw_df"]
    if df is None:
        return

    pipe = PreprocessingPipeline()
    # 1. Impute
    df_imp, _ = pipe.impute_missing_values(df, num_strategy="median", cat_strategy="mode")
    # 2. Outliers Cap
    df_cap, _ = pipe.handle_outliers(df_imp, method="iqr", action="cap")
    st.session_state["clean_df"] = df_cap

    # 3. Encode
    encoded_df = pipe.encode_features(df_cap, method="onehot")
    target_cls = "Placement_Status"
    target_reg = "Salary_Package_LPA"
    ignore_cols = ["Student_ID", "Name", target_cls, target_reg]
    feat_cols = [c for c in encoded_df.columns if c not in ignore_cols]
    st.session_state["feature_cols"] = feat_cols
    st.session_state["pipeline"] = pipe

    # 4. Scale
    X = encoded_df[feat_cols].values
    y_cls = (encoded_df[target_cls] == "Placed").astype(int).values
    X_scaled, _ = pipe.scale_features(pd.DataFrame(X, columns=feat_cols), method="standard")

    # 5. Train Classifiers
    cls_suite = PlacementClassifierSuite()
    cls_res = cls_suite.train_and_evaluate_all(X_scaled, y_cls, feat_cols)
    st.session_state["classifier_suite"] = cls_suite
    st.session_state["cls_results"] = cls_res

    # 6. Train Regression
    placed_mask = encoded_df[target_cls] == "Placed"
    if placed_mask.sum() > 20:
        X_reg = X_scaled[placed_mask]
        y_reg = encoded_df.loc[placed_mask, target_reg].values
        reg_suite = PackageRegressionSuite()
        reg_res = reg_suite.train_and_evaluate(X_reg, y_reg, feat_cols)
        st.session_state["reg_suite"] = reg_suite
        st.session_state["reg_results"] = reg_res

    # 7. Clustering
    clust_engine = StudentClusteringEngine()
    clust_engine.compute_elbow_curve(X_scaled, max_k=6)
    clust_res = clust_engine.fit_and_profile(X_scaled, df_cap, feat_cols, n_clusters=3)
    st.session_state["clustering_engine"] = clust_engine
    st.session_state["clustering_results"] = clust_res


# Sidebar Navigation & Authentication
def render_sidebar():
    st.sidebar.image("https://img.icons8.com/clouds/200/graduation-cap.png", width=90)
    st.sidebar.markdown("### Placement Analysis System")
    st.sidebar.caption("Data Science & Big Data Lab (CO1 - CO4)")
    st.sidebar.markdown("---")

    user = st.session_state["user"]

    if user is None:
        st.sidebar.subheader("🔐 Account Sign In")
        auth_mode = st.sidebar.radio("Select Action:", ["Login", "Register"], label_visibility="collapsed")

        if auth_mode == "Login":
            with st.sidebar.form("login_form"):
                username = st.text_input("Username").strip()
                password = st.text_input("Password", type="password")
                submit = st.form_submit_button("Sign In", use_container_width=True)

                if submit:
                    auth_user = auth_manager.authenticate_user(username, password)
                    if auth_user:
                        st.session_state["user"] = auth_user
                        st.sidebar.success(f"Welcome back, {auth_user['full_name']}!")
                        st.rerun()
                    else:
                        st.sidebar.error("Invalid username or password.")

            st.sidebar.markdown("#### ⚡ Quick Demo Logins")
            col1, col2 = st.sidebar.columns(2)
            with col1:
                if st.button("👨‍🏫 Admin", use_container_width=True, help="Prof. Sharma (Placement Cell)"):
                    st.session_state["user"] = auth_manager.authenticate_user("admin", "admin123")
                    st.rerun()
            with col2:
                if st.button("👨‍🎓 Student", use_container_width=True, help="Pranay Kumar (Student)"):
                    st.session_state["user"] = auth_manager.authenticate_user("pranay", "student123")
                    st.rerun()

        else:  # Register
            with st.sidebar.form("register_form"):
                new_username = st.text_input("Choose Username").strip()
                new_name = st.text_input("Full Name").strip()
                new_role = st.selectbox("Role", ["student", "admin"], format_func=lambda r: "Admin / Placement Cell" if r == "admin" else "Student")
                new_email = st.text_input("Email (optional)").strip()
                new_stu_id = st.text_input("Student Roll ID (if student)").strip() if new_role == "student" else ""
                new_password = st.text_input("Password", type="password")
                reg_submit = st.form_submit_button("Create Account", use_container_width=True)

                if reg_submit:
                    success, msg = auth_manager.register_user(
                        new_username, new_password, new_role, new_name, new_email, new_stu_id
                    )
                    if success:
                        st.sidebar.success(msg)
                    else:
                        st.sidebar.error(msg)

    else:
        # User is logged in
        role_badge = f"<span class='badge-admin'>ADMIN / PLACEMENT CELL</span>" if user["role"] == "admin" else f"<span class='badge-student'>STUDENT PORTAL</span>"
        st.sidebar.markdown(role_badge, unsafe_allow_html=True)
        st.sidebar.markdown(f"**{user['full_name']}**")
        st.sidebar.caption(f"Username: `{user['username']}` | Email: {user['email'] or 'N/A'}")
        if user.get("student_id"):
            st.sidebar.caption(f"Roll ID: `{user['student_id']}`")

        if st.sidebar.button("🚪 Logout", use_container_width=True):
            st.session_state["user"] = None
            st.rerun()

        st.sidebar.markdown("---")
        st.sidebar.subheader("📌 Navigation")

        if user["role"] == "admin":
            return st.sidebar.radio(
                "Go to Section:",
                [
                    "📊 Executive Dashboard",
                    "🧹 Data Preprocessing & EDA (CO1)",
                    "🤖 Classification & Regression (CO2 & CO4)",
                    "🎯 Student Segmentation & K-Means (CO3)",
                    "👥 User Management",
                    "📑 Reports & Data Export"
                ]
            )
        else:
            return st.sidebar.radio(
                "Go to Section:",
                [
                    "🎓 My Profile & Eligibility",
                    "🔮 Placement & Package Simulator",
                    "🎯 Skill Gap & Radar Benchmark",
                    "🚀 Targeted Improvement Plan"
                ]
            )

    return None


# Helper to convert student inputs into model-ready scaled feature vectors
def prepare_student_vector(input_dict: dict) -> np.ndarray:
    """Aligns student raw inputs to trained feature columns and standard scales it."""
    pipe = st.session_state.get("pipeline")
    feat_cols = st.session_state.get("feature_cols", [])

    if not pipe or not feat_cols:
        return np.zeros(len(feat_cols))

    # Build dummy single-row dataframe
    temp_df = pd.DataFrame([input_dict])
    # Encode with pipeline method
    enc_df = pipe.encode_features(temp_df, method=pipe.encoding_method)

    # Reindex so it has all columns used during training
    full_row = {}
    for col in feat_cols:
        full_row[col] = float(enc_df[col].iloc[0]) if col in enc_df.columns else 0.0

    row_df = pd.DataFrame([full_row])
    scaled = pipe.scaler.transform(row_df)[0]
    return scaled


# ==========================================
# ADMIN VIEW 1: EXECUTIVE DASHBOARD
# ==========================================
def render_admin_dashboard():
    st.markdown("<h1 class='main-title'>Placement Analytics & Batch Intelligence</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>High-level institutional metrics, recruitment distributions, and academic correlations.</p>", unsafe_allow_html=True)

    df = st.session_state.get("raw_df")
    if df is None or df.empty:
        st.warning("No placement dataset loaded. Please upload a dataset in the Preprocessing module.")
        return

    # High-level Metrics Cards
    total_students = len(df)
    placed_count = int((df["Placement_Status"] == "Placed").sum()) if "Placement_Status" in df.columns else 0
    placement_rate = round((placed_count / total_students) * 100, 1) if total_students > 0 else 0
    
    placed_df = df[df["Placement_Status"] == "Placed"] if "Placement_Status" in df.columns else df
    avg_pkg = round(float(placed_df["Salary_Package_LPA"].mean()), 2) if "Salary_Package_LPA" in placed_df.columns else 0.0
    max_pkg = round(float(placed_df["Salary_Package_LPA"].max()), 2) if "Salary_Package_LPA" in placed_df.columns else 0.0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-num">{total_students}</div>
            <div class="metric-label">Total Candidates</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-num" style="color: #10B981;">{placement_rate}%</div>
            <div class="metric-label">Placement Ratio</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-num" style="color: #6366F1;">₹ {avg_pkg} LPA</div>
            <div class="metric-label">Average Offer Package</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-num" style="color: #F59E0B;">₹ {max_pkg} LPA</div>
            <div class="metric-label">Highest Package</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Interactive Charts Grid
    row1_c1, row1_c2 = st.columns([1, 1])

    with row1_c1:
        st.subheader("Placement Ratio Overview")
        fig_pie = px.pie(
            df, 
            names="Placement_Status", 
            title="Overall Placed vs Unplaced Ratio",
            color="Placement_Status",
            color_discrete_map={"Placed": "#10B981", "Not Placed": "#EF4444"},
            hole=0.45
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie.update_layout(showlegend=False, margin=dict(t=40, b=20, l=20, r=20))
        st.plotly_chart(fig_pie, use_container_width=True)

    with row1_c2:
        st.subheader("Placement Performance by Engineering Branch")
        if "Branch" in df.columns:
            branch_summary = df.groupby(["Branch", "Placement_Status"]).size().reset_index(name="Count")
            fig_branch = px.bar(
                branch_summary, 
                x="Branch", 
                y="Count", 
                color="Placement_Status",
                barmode="group",
                title="Branch-wise Placement Breakdown",
                color_discrete_map={"Placed": "#10B981", "Not Placed": "#EF4444"}
            )
            fig_branch.update_layout(margin=dict(t=40, b=20, l=20, r=20))
            st.plotly_chart(fig_branch, use_container_width=True)

    row2_c1, row2_c2 = st.columns([1, 1])

    with row2_c1:
        st.subheader("CGPA vs Coding Competence Correlation")
        fig_scatter = px.scatter(
            df,
            x="CGPA",
            y="Coding_Score",
            color="Placement_Status",
            size="Salary_Package_LPA" if "Salary_Package_LPA" in df.columns else None,
            hover_data=["Name", "Branch", "Internships"],
            title="Impact of CGPA and Coding Score on Offers",
            color_discrete_map={"Placed": "#10B981", "Not Placed": "#EF4444"},
            opacity=0.75
        )
        fig_scatter.update_layout(margin=dict(t=40, b=20, l=20, r=20))
        st.plotly_chart(fig_scatter, use_container_width=True)

    with row2_c2:
        st.subheader("Salary Package Distribution (LPA)")
        if "Salary_Package_LPA" in df.columns and placed_count > 0:
            fig_hist = px.histogram(
                placed_df,
                x="Salary_Package_LPA",
                nbins=25,
                marginal="box",
                title="Salary Package Histogram & Boxplot (Placed Only)",
                color_discrete_sequence=["#6366F1"]
            )
            fig_hist.update_layout(margin=dict(t=40, b=20, l=20, r=20))
            st.plotly_chart(fig_hist, use_container_width=True)


# ==========================================
# ADMIN VIEW 2: DATA PREPROCESSING (CO1)
# ==========================================
def render_admin_preprocessing():
    st.markdown("<h1 class='main-title'>Data Preprocessing Module (CO1 Syllabus)</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Systematic data hygiene: Missing values imputation, Outlier handling, Categorical encoding, Scaling, and PCA dimensionality reduction.</p>", unsafe_allow_html=True)

    # Dataset Upload & Ingestion
    st.subheader("1. Dataset Ingestion & Historical Records")
    col_up, col_info = st.columns([2, 1])

    with col_up:
        uploaded_file = st.file_uploader(
            "Upload Student Placement Dataset (.csv or .xlsx)", 
            type=["csv", "xlsx", "xls"]
        )
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(".csv"):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                st.session_state["raw_df"] = df
                st.session_state["clean_df"] = None
                st.session_state["cls_results"] = None
                st.success(f"Successfully loaded '{uploaded_file.name}' with {len(df)} records and {df.shape[1]} columns!")
            except Exception as e:
                st.error(f"Error reading file: {e}")

    with col_info:
        if st.button("🔄 Reset to Default 650-Student Lab Dataset", use_container_width=True):
            def_path = os.path.join(os.path.dirname(__file__), "students_placement_dataset.csv")
            if os.path.exists(def_path):
                st.session_state["raw_df"] = pd.read_csv(def_path)
                st.session_state["clean_df"] = None
                st.session_state["cls_results"] = None
                st.rerun()

    df = st.session_state.get("raw_df")
    if df is None:
        st.info("Please upload a dataset or click 'Reset to Default Lab Dataset'.")
        return

    with st.expander("👀 View Raw Dataset Sample (First 5 Rows)", expanded=False):
        st.dataframe(df.head(), use_container_width=True)

    st.markdown("---")

    # Step A: Missing Value Detection & Imputation
    st.subheader("2. Missing Value Detection & Imputation")
    missing_df = PreprocessingPipeline.detect_missing_values(df)

    if not missing_df.empty:
        st.warning(f"⚠️ Missing values detected across {len(missing_df)} feature(s):")
        st.dataframe(missing_df, use_container_width=True)
    else:
        st.success("✅ No missing values detected in the dataset!")

    c_imp1, c_imp2, c_imp3 = st.columns([1, 1, 1])
    with c_imp1:
        num_strat = st.selectbox("Numeric Imputation Strategy:", ["median", "mean", "mode", "drop"], index=0)
    with c_imp2:
        cat_strat = st.selectbox("Categorical Imputation Strategy:", ["mode", "constant", "drop"], index=0)
    with c_imp3:
        st.write("")
        st.write("")
        if st.button("⚡ Execute Imputation Pipeline", use_container_width=True):
            clean_df, log = PreprocessingPipeline.impute_missing_values(df, num_strategy=num_strat, cat_strategy=cat_strat)
            st.session_state["raw_df"] = clean_df
            st.session_state["clean_df"] = clean_df
            st.success("Imputation complete!")
            if log:
                st.json(log)
            st.rerun()

    st.markdown("---")

    # Step B: Outlier Detection & Handling
    st.subheader("3. Outlier Detection & Handling (IQR Fence & Z-Score)")
    curr_df = st.session_state.get("clean_df") if st.session_state.get("clean_df") is not None else df

    col_out_cfg1, col_out_cfg2, col_out_cfg3 = st.columns(3)
    with col_out_cfg1:
        outlier_method = st.radio("Outlier Detection Method:", ["iqr", "zscore"], format_func=lambda m: "Interquartile Range (IQR Fence)" if m == "iqr" else "Z-Score (|z| > 3.0)")
    with col_out_cfg2:
        outlier_action = st.selectbox("Handling Action:", ["cap", "drop", "keep"], format_func=lambda a: "Winsorization / Cap to Fences" if a == "cap" else ("Drop Outlier Rows" if a == "drop" else "Keep Outliers"))
    with col_out_cfg3:
        iqr_mult = st.slider("IQR Multiplier:", 1.0, 3.0, 1.5, 0.1) if outlier_method == "iqr" else 1.5

    outliers_detected = PreprocessingPipeline.detect_outliers(curr_df, method=outlier_method, iqr_multiplier=iqr_mult)
    if not outliers_detected.empty and (outliers_detected["Outlier Count"] > 0).any():
        st.dataframe(outliers_detected[outliers_detected["Outlier Count"] > 0], use_container_width=True)
    else:
        st.success("No critical outliers detected with current thresholds.")

    if st.button("Apply Outlier Remediation", use_container_width=False):
        processed_df, out_log = PreprocessingPipeline.handle_outliers(
            curr_df, method=outlier_method, action=outlier_action, iqr_multiplier=iqr_mult
        )
        st.session_state["clean_df"] = processed_df
        st.success(f"Outlier remediation applied ({outlier_action}).")
        if out_log:
            st.write(out_log)

    st.markdown("---")

    # Step C: Encoding, Scaling & PCA
    st.subheader("4. Feature Scaling & Dimensionality Reduction (PCA)")
    active_df = st.session_state.get("clean_df") if st.session_state.get("clean_df") is not None else curr_df

    col_enc1, col_enc2, col_enc3 = st.columns(3)
    with col_enc1:
        enc_method = st.selectbox("Categorical Encoding:", ["onehot", "label"], format_func=lambda e: "One-Hot Encoding" if e == "onehot" else "Label Encoding")
    with col_enc2:
        scale_method = st.selectbox("Feature Scaling:", ["standard", "minmax"], format_func=lambda s: "StandardScaler (Mean=0, Std=1)" if s == "standard" else "MinMaxScaler (0 to 1)")
    with col_enc3:
        pca_components = st.slider("PCA Target Dimensions:", 2, 6, 2)

    pipe = PreprocessingPipeline()
    enc_df = pipe.encode_features(active_df, method=enc_method)
    feat_cols = [c for c in enc_df.columns if c not in ["Student_ID", "Name", "Placement_Status", "Salary_Package_LPA"]]
    X = enc_df[feat_cols].values
    X_scaled, scaled_df = pipe.scale_features(pd.DataFrame(X, columns=feat_cols), method=scale_method)
    pca_df, pca_metrics = pipe.compute_pca(X_scaled, feat_cols, n_components=pca_components)

    c_pca1, c_pca2 = st.columns([1, 1])
    with c_pca1:
        st.markdown(f"**Total Variance Explained by {pca_components} PCs:** `{pca_metrics['total_explained_variance']}%`")
        var_df = pd.DataFrame({
            "Principal Component": [f"PC {i+1}" for i in range(pca_components)],
            "Variance Ratio (%)": pca_metrics["explained_variance_ratio"]
        })
        fig_scree = px.bar(
            var_df, 
            x="Principal Component", 
            y="Variance Ratio (%)", 
            text="Variance Ratio (%)",
            title="Scree Plot: Explained Variance per Principal Component",
            color_discrete_sequence=["#3B82F6"]
        )
        fig_scree.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        st.plotly_chart(fig_scree, use_container_width=True)

    with c_pca2:
        pca_plot_df = pca_df.copy()
        pca_plot_df["Placement_Status"] = active_df["Placement_Status"].values
        pca_plot_df["Student_ID"] = active_df["Student_ID"].values
        fig_pca_scatter = px.scatter(
            pca_plot_df,
            x="PC1",
            y="PC2",
            color="Placement_Status",
            hover_data=["Student_ID"],
            title="2D PCA Dimensionality Reduction Projection",
            color_discrete_map={"Placed": "#10B981", "Not Placed": "#EF4444"}
        )
        st.plotly_chart(fig_pca_scatter, use_container_width=True)

    with st.expander("🔍 Inspect PCA Feature Eigenvector Loadings Matrix"):
        st.caption("Shows how much each original feature contributes to Principal Components PC1 and PC2.")
        st.dataframe(pca_metrics["loadings"].round(3), use_container_width=True)


# ==========================================
# ADMIN VIEW 3: CLASSIFICATION & REGRESSION (CO2 & CO4)
# ==========================================
def render_admin_models():
    st.markdown("<h1 class='main-title'>Predictive Modeling & Model Evaluation (CO2 & CO4)</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Train, benchmark, and compare 4 classification algorithms (KNN, Decision Tree, SVM, Logistic Regression) plus Salary Package Regression.</p>", unsafe_allow_html=True)

    df = st.session_state.get("clean_df")
    if df is None:
        df = st.session_state.get("raw_df")
    if df is None:
        st.warning("Please load a dataset first.")
        return

    # Hyperparameter & Training Configuration
    st.subheader("1. Model Training Hyperparameters")
    p1, p2, p3, p4, p5 = st.columns(5)
    with p1:
        test_size = st.slider("Test Split Size:", 0.1, 0.4, 0.2, 0.05)
    with p2:
        cv_folds = st.slider("Stratified CV Folds:", 3, 10, 5)
    with p3:
        knn_k = st.slider("KNN Neighbors (k):", 3, 15, 5, 2)
    with p4:
        dt_depth = st.slider("Decision Tree Max Depth:", 2, 12, 5)
    with p5:
        svm_kern = st.selectbox("SVM Kernel:", ["rbf", "linear", "poly"])

    if st.button("🚀 Retrain All Models & Re-evaluate", use_container_width=True):
        with st.spinner("Training classification algorithms & cross-validation folds..."):
            pipe = PreprocessingPipeline()
            enc_df = pipe.encode_features(df, method="onehot")
            feat_cols = [c for c in enc_df.columns if c not in ["Student_ID", "Name", "Placement_Status", "Salary_Package_LPA"]]
            X = enc_df[feat_cols].values
            y = (enc_df["Placement_Status"] == "Placed").astype(int).values
            X_scaled, _ = pipe.scale_features(pd.DataFrame(X, columns=feat_cols), method="standard")

            cls_suite = PlacementClassifierSuite()
            cls_res = cls_suite.train_and_evaluate_all(
                X_scaled, y, feat_cols,
                test_size=test_size,
                cv_folds=cv_folds,
                knn_neighbors=knn_k,
                dt_max_depth=dt_depth,
                svm_kernel=svm_kern
            )
            st.session_state["classifier_suite"] = cls_suite
            st.session_state["cls_results"] = cls_res
            st.session_state["pipeline"] = pipe
            st.session_state["feature_cols"] = feat_cols

            # Retrain Regression
            placed_mask = enc_df["Placement_Status"] == "Placed"
            if placed_mask.sum() > 20:
                X_reg = X_scaled[placed_mask]
                y_reg = enc_df.loc[placed_mask, "Salary_Package_LPA"].values
                reg_suite = PackageRegressionSuite()
                reg_res = reg_suite.train_and_evaluate(X_reg, y_reg, feat_cols, test_size=test_size)
                st.session_state["reg_suite"] = reg_suite
                st.session_state["reg_results"] = reg_res

            st.success("All models trained and evaluated successfully!")

    cls_res = st.session_state.get("cls_results")
    if not cls_res:
        st.info("Click the button above to train models.")
        return

    st.markdown("---")

    # Algorithm Comparison Table
    st.subheader("2. Classification Performance Benchmark")
    comp_df = cls_res["comparison_df"]
    best_model = cls_res["best_model_name"]
    st.markdown(f"🏆 **Top Performing Algorithm:** `{best_model}` (highest F1-score & balanced recall)")
    st.dataframe(comp_df, use_container_width=True)

    # Visual Benchmark Bar Chart
    fig_comp = px.bar(
        comp_df,
        x="Algorithm",
        y=["Accuracy (%)", "Precision (%)", "Recall (%)", "F1-Score (%)"],
        barmode="group",
        title="Multi-Metric Comparison Across Algorithms",
        color_discrete_sequence=["#3B82F6", "#10B981", "#F59E0B", "#8B5CF6"]
    )
    fig_comp.update_layout(yaxis=dict(range=[60, 100]), margin=dict(t=40, b=20, l=20, r=20))
    st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown("---")

    # Confusion Matrices and ROC Curves
    st.subheader("3. Model Diagnostics: Confusion Matrices & ROC Curves")
    tab_cm, tab_roc, tab_feat = st.tabs(["📊 Confusion Matrices", "📈 ROC & AUC Curves", "🎯 Feature Importances"])

    with tab_cm:
        cm_cols = st.columns(4)
        for idx, (m_name, m_data) in enumerate(cls_res["detailed_results"].items()):
            with cm_cols[idx]:
                st.markdown(f"**{m_name}**")
                cm = np.array(m_data["confusion_matrix"]["raw"])
                # Annotations for True/False Positives/Negatives
                fig_cm = px.imshow(
                    cm,
                    text_auto=True,
                    labels=dict(x="Predicted Class", y="Actual Class"),
                    x=["Not Placed", "Placed"],
                    y=["Not Placed", "Placed"],
                    color_continuous_scale="Blues"
                )
                fig_cm.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig_cm, use_container_width=True)
                st.caption(f"Acc: {m_data['accuracy']}% | F1: {m_data['f1_score']}%")

    with tab_roc:
        fig_roc = go.Figure()
        colors = ["#2563EB", "#10B981", "#F59E0B", "#8B5CF6"]
        for idx, (m_name, m_data) in enumerate(cls_res["detailed_results"].items()):
            fpr = m_data["roc_curve"]["fpr"]
            tpr = m_data["roc_curve"]["tpr"]
            auc_val = m_data["roc_auc"]
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode="lines",
                name=f"{m_name} (AUC = {auc_val})",
                line=dict(color=colors[idx % len(colors)], width=2.5)
            ))
        # Add 45 degree random guess baseline
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines",
            name="Random Baseline",
            line=dict(dash="dash", color="#94A3B8")
        ))
        fig_roc.update_layout(
            title="Receiver Operating Characteristic (ROC) Comparison",
            xaxis_title="False Positive Rate (1 - Specificity)",
            yaxis_title="True Positive Rate (Sensitivity / Recall)",
            height=460,
            margin=dict(t=40, b=20, l=20, r=20)
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    with tab_feat:
        dt_imp = cls_res["detailed_results"].get("Decision Tree", {}).get("feature_importance", {})
        if dt_imp:
            imp_df = pd.DataFrame(list(dt_imp.items()), columns=["Feature", "Importance"]).sort_values(by="Importance", ascending=True)
            fig_dt_imp = px.bar(
                imp_df.tail(10),
                x="Importance",
                y="Feature",
                orientation="h",
                title="Decision Tree Feature Gini Importances (Top Predictors)",
                color="Importance",
                color_continuous_scale="Viridis"
            )
            st.plotly_chart(fig_dt_imp, use_container_width=True)

    st.markdown("---")

    # Salary Package Regression
    st.subheader("4. Continuous Outcome: Salary Package Regression (LPA)")
    reg_res = st.session_state.get("reg_results")
    if reg_res:
        lr_met = reg_res["linear_regression"]
        rc1, rc2, rc3, rc4 = st.columns(4)
        rc1.metric("Linear Reg. R² Score", f"{lr_met['r2_score']}")
        rc2.metric("Mean Absolute Error (MAE)", f"₹ {lr_met['mae']} LPA")
        rc3.metric("Root Mean Sq Error (RMSE)", f"₹ {lr_met['rmse']} LPA")
        rc4.metric("5-Fold Cross-Val R²", f"{lr_met['cv_r2']}")

        # Actual vs Predicted Plot
        act_v_pred = reg_res["actual_vs_predicted"]
        scatter_reg_df = pd.DataFrame(act_v_pred)
        fig_reg = px.scatter(
            scatter_reg_df,
            x="actual",
            y="predicted",
            labels={"actual": "Actual Offer Package (LPA)", "predicted": "Model Predicted Package (LPA)"},
            title="Actual vs Predicted Packages (Validation Samples)"
        )
        # Add diagonal ideal line
        max_val = max(scatter_reg_df["actual"].max(), scatter_reg_df["predicted"].max())
        fig_reg.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode="lines", name="Ideal 1:1 Fit", line=dict(dash="dash", color="green")))
        st.plotly_chart(fig_reg, use_container_width=True)


# ==========================================
# ADMIN VIEW 4: STUDENT SEGMENTATION & CLUSTERING (CO3)
# ==========================================
def render_admin_clustering():
    st.markdown("<h1 class='main-title'>Student Segmentation & K-Means Clustering (CO3)</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Unsupervised grouping into performance tiers with the Elbow Method to target specialized placement training programs.</p>", unsafe_allow_html=True)

    df = st.session_state.get("clean_df")
    if df is None:
        df = st.session_state.get("raw_df")
    if df is None:
        st.warning("Please load a dataset first.")
        return

    pipe = st.session_state.get("pipeline")
    feat_cols = st.session_state.get("feature_cols")

    if not pipe or not feat_cols:
        st.info("Please train models first to prepare standardized features.")
        return

    enc_df = pipe.encode_features(df, method=pipe.encoding_method)
    X = enc_df[feat_cols].values
    X_scaled = pipe.scaler.transform(pd.DataFrame(X, columns=feat_cols))

    clust_engine = st.session_state.get("clustering_engine") or StudentClusteringEngine()

    # Step A: Elbow Method Visualizer
    st.subheader("1. Optimal Cluster Selection (The Elbow Method & Silhouette Scores)")
    if not clust_engine.elbow_data:
        clust_engine.compute_elbow_curve(X_scaled, max_k=8)

    elbow_data = clust_engine.elbow_data
    e1, e2 = st.columns(2)
    with e1:
        fig_elbow = px.line(
            x=elbow_data["k_values"],
            y=elbow_data["inertias"],
            markers=True,
            title="Elbow Curve: Within-Cluster Sum of Squares (Inertia)",
            labels={"x": "Number of Clusters (k)", "y": "Inertia (WCSS)"}
        )
        fig_elbow.update_traces(line=dict(color="#2563EB", width=3))
        st.plotly_chart(fig_elbow, use_container_width=True)

    with e2:
        fig_sil = px.line(
            x=elbow_data["k_values"],
            y=elbow_data["silhouettes"],
            markers=True,
            title="Silhouette Score vs Number of Clusters",
            labels={"x": "Number of Clusters (k)", "y": "Silhouette Score"}
        )
        fig_sil.update_traces(line=dict(color="#10B981", width=3))
        st.plotly_chart(fig_sil, use_container_width=True)

    st.markdown("---")

    # Step B: Fit Clusters & Display Semantic Tiers
    st.subheader("2. Performance Tier Segmentation")
    c_k, _ = st.columns([1, 2])
    with c_k:
        k_choice = st.slider("Select Target Clusters (k):", 2, 5, 3)

    if st.button("Apply K-Means Clustering", use_container_width=False) or st.session_state.get("clustering_results") is None:
        clust_res = clust_engine.fit_and_profile(X_scaled, df, feat_cols, n_clusters=k_choice)
        st.session_state["clustering_results"] = clust_res
        st.session_state["clustering_engine"] = clust_engine

    clust_res = st.session_state.get("clustering_results")
    if not clust_res:
        return

    # Render Tier Profiles Cards
    profiles = clust_res["profiles"]
    prof_cols = st.columns(len(profiles))
    for idx, prof in enumerate(profiles):
        with prof_cols[idx]:
            m = prof["metrics"]
            st.markdown(f"""
            <div style="background: #ffffff; border-top: 5px solid {prof['badge_color']}; padding: 16px; border-radius: 8px; border: 1px solid #e2e8f0; height: 100%;">
                <h3 style="margin: 0; font-size: 1.1rem; color: #1e293b;">{prof['tier_label']}</h3>
                <p style="color: #64748b; font-size: 0.85rem; margin: 4px 0 12px 0;">Candidates: <strong>{m['count']}</strong> ({round((m['count']/len(df))*100, 1)}%)</p>
                <div style="font-size: 0.9rem; line-height: 1.6; color: #334155;">
                    <div>Avg CGPA: <strong>{m['cgpa_mean']}</strong></div>
                    <div>Avg Coding Score: <strong>{m['coding_mean']}/100</strong></div>
                    <div>Avg Aptitude: <strong>{m['apt_mean']}/100</strong></div>
                    <div>Placement Ratio: <strong>{m['placed_pct']}%</strong></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 2D & 3D PCA Cluster Visualization
    clustered_df = clust_res["clustered_df"]
    v_tab1, v_tab2 = st.tabs(["🌐 2D PCA Cluster Map", "🔮 3D PCA Cluster Map"])

    with v_tab1:
        fig_clust2d = px.scatter(
            clustered_df,
            x="PCA_1",
            y="PCA_2",
            color="Performance_Tier",
            hover_data=["Student_ID", "Name", "Branch", "CGPA", "Coding_Score", "Placement_Status"],
            title=f"2D PCA Cluster Separation ({clust_res['pca_variance_2d'][0]}% + {clust_res['pca_variance_2d'][1]}% Variance)",
            color_discrete_sequence=["#10B981", "#F59E0B", "#EF4444", "#8B5CF6"]
        )
        st.plotly_chart(fig_clust2d, use_container_width=True)

    with v_tab2:
        if "PCA_3" in clustered_df.columns:
            fig_clust3d = px.scatter_3d(
                clustered_df,
                x="PCA_1",
                y="PCA_2",
                z="PCA_3",
                color="Performance_Tier",
                hover_data=["Student_ID", "Name", "Branch", "CGPA"],
                title="3D PCA Interactive Cluster Projection",
                color_discrete_sequence=["#10B981", "#F59E0B", "#EF4444", "#8B5CF6"]
            )
            fig_clust3d.update_layout(height=600)
            st.plotly_chart(fig_clust3d, use_container_width=True)

    st.markdown("---")

    # Actionable Training Interventions per Tier
    st.subheader("3. Placement Cell Intervention Blueprints")
    for prof in profiles:
        with st.expander(f"📌 Targeted Strategy for: {prof['tier_label']}"):
            st.markdown(f"**Target Company Segment:** {prof['focus_area']}")
            st.markdown("**Actionable Interventions:**")
            for rec in prof["recommendations"]:
                st.markdown(f"- {rec}")


# ==========================================
# ADMIN VIEW 5: USER MANAGEMENT
# ==========================================
def render_admin_users():
    st.markdown("<h1 class='main-title'>User Accounts & Role Administration</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Manage authorized student and staff credentials with salted SHA-256 password security.</p>", unsafe_allow_html=True)

    users = auth_manager.get_all_users()
    users_df = pd.DataFrame(users)
    st.dataframe(users_df[["id", "username", "role", "full_name", "email", "student_id", "created_at"]], use_container_width=True)

    st.markdown("---")
    st.subheader("Register New Account")
    with st.form("admin_create_user_form"):
        u1, u2, u3 = st.columns(3)
        with u1:
            u_name = st.text_input("Username").strip()
            u_pwd = st.text_input("Password", type="password")
        with u2:
            u_role = st.selectbox("Role", ["student", "admin"])
            u_fullname = st.text_input("Full Name").strip()
        with u3:
            u_email = st.text_input("Email").strip()
            u_stuid = st.text_input("Student Roll ID (e.g. STU202401)").strip() if u_role == "student" else ""

        btn_create = st.form_submit_button("Register User")
        if btn_create:
            ok, msg = auth_manager.register_user(u_name, u_pwd, u_role, u_fullname, u_email, u_stuid)
            if ok:
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)


# ==========================================
# ADMIN VIEW 6: REPORTS & EXPORT
# ==========================================
def render_admin_reports():
    st.markdown("<h1 class='main-title'>Export Center & Institutional Dossiers</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Download preprocessed datasets, cluster allocations, ML benchmark logs, and printable executive HTML dossiers.</p>", unsafe_allow_html=True)

    df = st.session_state.get("clean_df")
    if df is None:
        df = st.session_state.get("raw_df")

    cls_res = st.session_state.get("cls_results")
    clust_res = st.session_state.get("clustering_results")

    exp1, exp2 = st.columns(2)

    with exp1:
        st.subheader("1. Download Datasets & ML Logs (CSV)")

        if df is not None:
            csv_data = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Clean Preprocessed Dataset (CSV)",
                data=csv_data,
                file_name="clean_placement_dataset.csv",
                mime="text/csv",
                use_container_width=True
            )

        if clust_res is not None:
            clustered_df = clust_res["clustered_df"]
            clust_csv = clustered_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Clustered Student Cohort with Tiers (CSV)",
                data=clust_csv,
                file_name="student_clusters_tiers.csv",
                mime="text/csv",
                use_container_width=True
            )

        if cls_res is not None:
            comp_csv = cls_res["comparison_df"].to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download ML Model Performance Benchmark (CSV)",
                data=comp_csv,
                file_name="ml_model_benchmarks.csv",
                mime="text/csv",
                use_container_width=True
            )

    with exp2:
        st.subheader("2. Executive Placement Analytics Dossier (HTML)")
        st.caption("Formatted institutional report containing KPI summaries, algorithm benchmarking, and K-Means intervention plans.")

        if df is not None and cls_res is not None and clust_res is not None:
            html_content = report_generator.generate_placement_executive_summary_html(
                df,
                cls_res["comparison_df"],
                clust_res["profiles"],
                cls_res["best_model_name"]
            )

            st.download_button(
                label="📄 Download Printable Placement Dossier (HTML)",
                data=html_content.encode("utf-8"),
                file_name="placement_executive_summary_report.html",
                mime="text/html",
                use_container_width=True
            )

            with st.expander("👁️ Live Preview of Executive HTML Report"):
                st.components.v1.html(html_content, height=500, scrolling=True)


# ==========================================
# STUDENT PORTAL: VIEW 1 - PROFILE & ELIGIBILITY
# ==========================================
def render_student_profile():
    user = st.session_state["user"]
    st.markdown("<h1 class='main-title'>Student Placement Portal</h1>", unsafe_allow_html=True)
    st.markdown(f"<p class='sub-title'>Welcome, <strong>{user['full_name']}</strong> ({user.get('student_id') or 'ID: N/A'}). Review your corporate recruitment eligibility across tier brackets.</p>", unsafe_allow_html=True)

    # Locate student record in raw_df if exists, or use default starter
    df = st.session_state.get("raw_df")
    stu_record = None
    if df is not None and user.get("student_id"):
        match = df[df["Student_ID"] == user["student_id"]]
        if not match.empty:
            stu_record = match.iloc[0].to_dict()

    if stu_record is None:
        # Default mock profile
        stu_record = {
            "Student_ID": user.get("student_id", "STU202401"),
            "Name": user["full_name"],
            "Gender": "Male",
            "Branch": "Computer Science",
            "CGPA": 8.15,
            "Coding_Score": 72.0,
            "Aptitude_Score": 68.0,
            "Soft_Skills_Score": 74.0,
            "Internships": 1,
            "Projects_Count": 3,
            "Certifications_Count": 2,
            "Backlogs": 0,
            "Workshops_Attended": 2,
            "Hackathons_Attended": 1
        }

    # Profile Summary Cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Academic CGPA", f"{stu_record['CGPA']} / 10.0")
    c2.metric("Coding Assessment", f"{stu_record['Coding_Score']} / 100")
    c3.metric("Aptitude Score", f"{stu_record['Aptitude_Score']} / 100")
    c4.metric("Industry Internships", f"{stu_record['Internships']}")

    st.markdown("---")

    # Company Eligibility Assessment
    st.subheader("🏢 Recruitment Tier Eligibility Status")
    eligibility_list = SkillGapAnalyzer.evaluate_eligibility(stu_record)

    for item in eligibility_list:
        is_ok = item["eligible"]
        badge = "✅ ELIGIBLE" if is_ok else "❌ NOT CURRENTLY ELIGIBLE"
        color = "#10B981" if is_ok else "#EF4444"
        bg_col = "#F0FDF4" if is_ok else "#FEF2F2"

        with st.expander(f"{item['tier_name']} — {badge}", expanded=is_ok):
            st.markdown(f"""
            <div style="background: {bg_col}; border-left: 4px solid {color}; padding: 12px; border-radius: 6px; margin-bottom: 8px;">
                <p style="margin: 0; font-size: 0.9rem; color: #1e293b;"><strong>Benchmark Rule:</strong> {item['criteria']}</p>
                <p style="margin: 4px 0 0 0; font-size: 0.9rem; color: #475569;"><strong>Recruiting Companies:</strong> {', '.join(item['companies'])}</p>
            </div>
            """, unsafe_allow_html=True)
            if not is_ok:
                st.markdown("**Eligibility Shortfalls:**")
                for r in item["reasons"]:
                    st.markdown(f"- ⚠️ {r}")


# ==========================================
# STUDENT PORTAL: VIEW 2 - SIMULATOR
# ==========================================
def render_student_simulator():
    st.markdown("<h1 class='main-title'>Real-Time Placement & Package Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Simulate how adjusting your CGPA, coding drills, or internships impacts your probability of placement and expected salary package.</p>", unsafe_allow_html=True)

    cls_suite = st.session_state.get("classifier_suite")
    reg_suite = st.session_state.get("reg_suite")

    if not cls_suite or not reg_suite:
        st.warning("Models are not yet ready. Please contact the administrator.")
        return

    col_sim_in, col_sim_out = st.columns([1.2, 1])

    with col_sim_in:
        st.subheader("🎯 Customize Your Profile Parameters")
        s_cgpa = st.slider("Cumulative CGPA:", 5.0, 10.0, 8.2, 0.05)
        s_coding = st.slider("Coding Competence (0-100):", 30.0, 100.0, 75.0, 1.0)
        s_apt = st.slider("Aptitude Test Score (0-100):", 30.0, 100.0, 70.0, 1.0)
        s_soft = st.slider("Soft Skills & Interview Rating (0-100):", 30.0, 100.0, 72.0, 1.0)
        
        sim_c1, sim_c2 = st.columns(2)
        with sim_c1:
            s_intern = st.number_input("Internships Completed:", 0, 5, 1)
            s_proj = st.number_input("Major Projects Built:", 0, 8, 3)
        with sim_c2:
            s_cert = st.number_input("Certifications:", 0, 6, 2)
            s_backlogs = st.number_input("Active Backlogs:", 0, 4, 0)

        s_branch = st.selectbox("Engineering Branch:", [
            "Computer Science", "Information Technology", 
            "Electronics & Comm", "Electrical & Electronics", "Mechanical"
        ])
        s_gender = st.selectbox("Gender:", ["Male", "Female", "Other"])

    # Prepare vector
    sim_dict = {
        "Gender": s_gender,
        "Branch": s_branch,
        "CGPA": s_cgpa,
        "Coding_Score": s_coding,
        "Aptitude_Score": s_apt,
        "Soft_Skills_Score": s_soft,
        "Internships": s_intern,
        "Projects_Count": s_proj,
        "Certifications_Count": s_cert,
        "Backlogs": s_backlogs,
        "Workshops_Attended": 2,
        "Hackathons_Attended": 1
    }

    vec = prepare_student_vector(sim_dict)

    with col_sim_out:
        st.subheader("🔮 Predictive Intelligence Outcomes")

        # Classification prediction
        status_pred, conf = cls_suite.predict_single(None, vec)
        is_placed = status_pred == "Placed"

        if is_placed:
            st.markdown(f"""
            <div style="background: #ECFDF5; border: 2px solid #10B981; padding: 20px; border-radius: 12px; text-align: center; margin-bottom: 20px;">
                <h2 style="color: #065F46; margin: 0;">🎉 High Placement Probability</h2>
                <div style="font-size: 2.2rem; font-weight: 800; color: #10B981; margin: 8px 0;">{conf}% Confidence</div>
                <p style="color: #047857; margin: 0; font-size: 0.95rem;">Profile matches successful placed cohorts based on {cls_suite.best_model_name}.</p>
            </div>
            """, unsafe_allow_html=True)

            # Continuous package prediction
            pred_pkg, low_pkg, high_pkg = reg_suite.predict_package(vec)
            st.markdown(f"""
            <div style="background: #EFF6FF; border: 2px solid #3B82F6; padding: 20px; border-radius: 12px; text-align: center;">
                <h3 style="color: #1E40AF; margin: 0;">Predicted Salary Package</h3>
                <div style="font-size: 2.5rem; font-weight: 800; color: #2563EB; margin: 8px 0;">₹ {pred_pkg} LPA</div>
                <p style="color: #3B82F6; margin: 0; font-size: 0.95rem;">Expected Package Range: <strong>₹ {low_pkg} LPA — ₹ {high_pkg} LPA</strong></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background: #FEF2F2; border: 2px solid #EF4444; padding: 20px; border-radius: 12px; text-align: center;">
                <h2 style="color: #991B1B; margin: 0;">⚠️ Placement Readiness Alert</h2>
                <div style="font-size: 2.2rem; font-weight: 800; color: #EF4444; margin: 8px 0;">{conf}% Risk</div>
                <p style="color: #B91C1C; margin: 0; font-size: 0.95rem;">Profile currently exhibits high risk of not qualifying initial screening rounds. Check the Skill Gap tab for targeted remedies.</p>
            </div>
            """, unsafe_allow_html=True)


# ==========================================
# STUDENT PORTAL: VIEW 3 - SKILL GAP RADAR
# ==========================================
def render_student_skill_gap():
    st.markdown("<h1 class='main-title'>Personalized Skill Gap Benchmark</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Multivariate radar analysis comparing your technical, academic, and experiential competencies against successfully placed peers.</p>", unsafe_allow_html=True)

    df = st.session_state.get("raw_df")
    user = st.session_state["user"]

    # Student starter profile
    stu_record = {
        "CGPA": 8.1,
        "Coding_Score": 72.0,
        "Aptitude_Score": 68.0,
        "Soft_Skills_Score": 74.0,
        "Internships": 1,
        "Projects_Count": 3,
        "Certifications_Count": 2,
        "Backlogs": 0
    }

    if df is not None and user.get("student_id"):
        match = df[df["Student_ID"] == user["student_id"]]
        if not match.empty:
            stu_record = match.iloc[0].to_dict()

    gap_data = SkillGapAnalyzer.compute_gap_analysis(stu_record, df)

    col_radar, col_recs = st.columns([1.2, 1])

    with col_radar:
        st.subheader("Competency Radar Comparison")
        categories = gap_data["radar_categories"]
        stu_vals = gap_data["student_radar_scores"]
        bench_vals = gap_data["benchmark_radar_scores"]

        # Close the radar loop
        r_cats = categories + [categories[0]]
        r_stu = stu_vals + [stu_vals[0]]
        r_bench = bench_vals + [bench_vals[0]]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=r_bench,
            theta=r_cats,
            fill='toself',
            name='Placed Benchmark',
            line=dict(color='#94A3B8', dash='dash'),
            fillcolor='rgba(148, 163, 184, 0.15)'
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=r_stu,
            theta=r_cats,
            fill='toself',
            name='My Competency Profile',
            line=dict(color='#2563EB', width=2),
            fillcolor='rgba(37, 99, 235, 0.25)'
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
            height=480,
            margin=dict(t=30, b=20, l=40, r=40)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_recs:
        st.subheader("Prioritized Skill Enhancements")
        recs = gap_data["recommendations"]
        for r in recs:
            p_color = r["color"]
            st.markdown(f"""
            <div style="background: #ffffff; border-left: 4px solid {p_color}; padding: 12px 16px; border-radius: 6px; margin-bottom: 12px; border: 1px solid #e2e8f0;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                    <strong style="color: #1e293b; font-size: 0.95rem;">{r['area']}</strong>
                    <span style="background: {p_color}22; color: {p_color}; font-size: 0.75rem; font-weight: 700; padding: 2px 8px; border-radius: 4px;">{r['priority']} PRIORITY</span>
                </div>
                <p style="margin: 0; font-size: 0.88rem; color: #475569; line-height: 1.5;">{r['advice']}</p>
            </div>
            """, unsafe_allow_html=True)


# ==========================================
# STUDENT PORTAL: VIEW 4 - ACTION PLAN
# ==========================================
def render_student_action_plan():
    st.markdown("<h1 class='main-title'>4-Phase Placement Acceleration Roadmap</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Structured weekly milestones designed to elevate you into top-tier recruitment brackets.</p>", unsafe_allow_html=True)

    phases = [
        {
            "phase": "Phase 1: Academic & Coding Foundation (Weeks 1 - 4)",
            "icon": "📚",
            "tasks": [
                "Ensure zero active academic backlogs; request faculty remedial sessions if needed.",
                "Target 50 LeetCode Easy & Medium problems in Arrays, Strings, HashMaps.",
                "Maintain minimum CGPA target of 8.0+ for Dream company eligibility."
            ]
        },
        {
            "phase": "Phase 2: Project Portfolio & Open Source (Weeks 5 - 8)",
            "icon": "💻",
            "tasks": [
                "Build and deploy 1 end-to-end full stack / data science project with live URL.",
                "Write detailed GitHub READMEs with architecture diagrams and API docs.",
                "Participate in at least 1 departmental hackathon or open source contribution."
            ]
        },
        {
            "phase": "Phase 3: Industry Readiness & Internships (Weeks 9 - 12)",
            "icon": "🏢",
            "tasks": [
                "Complete a verified 6-8 week technical internship or industrial training.",
                "Practice speed quantitative aptitude on IndiaBIX (Time & Work, Permutations).",
                "Earn 1 industry recognized certification (AWS, Google Cloud, or Microsoft)."
            ]
        },
        {
            "phase": "Phase 4: Mock Interviews & Executive Presence (Weeks 13 - 16)",
            "icon": "🎙️",
            "tasks": [
                "Schedule 3 mock technical interviews with department alumni mentors.",
                "Prepare STAR format behavioral stories for HR rounds (Leadership, Conflict, Teamwork).",
                "Tailor resume using single-page ATS-friendly LaTeX / Word template."
            ]
        }
    ]

    for p in phases:
        with st.expander(f"{p['icon']} {p['phase']}", expanded=True):
            for t in p["tasks"]:
                st.checkbox(t, key=f"task_{t[:15]}")


# ==========================================
# MAIN ROUTING LOGIC
# ==========================================
def main():
    st.set_page_config(
        page_title="Placement Analysis & Intelligence System",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    init_session()
    auth_manager.init_auth_db()
    if st.session_state["raw_df"] is not None and st.session_state["cls_results"] is None:
        ensure_models_trained()

    selected_section = render_sidebar()
    user = st.session_state.get("user")

    if user is None:
        # Landing Page when not logged in
        st.markdown("<h1 class='main-title'>Placement Analysis & Intelligence System</h1>", unsafe_allow_html=True)
        st.markdown("<p class='sub-title'>Data Science & Big Data Lab Integrated Platform (CO1 - CO4 Curriculum Alignment)</p>", unsafe_allow_html=True)

        st.info("👈 Please **Sign In** or choose a **Quick Demo Login** (Admin or Student) in the sidebar to access the full system.")

        col_feat1, col_feat2 = st.columns(2)
        with col_feat1:
            st.markdown("""
            ### 🏫 For Placement Cell / Faculty
            - **Data Preprocessing (CO1):** Automated missing value imputation, outlier Winsorization, encoding & PCA.
            - **Predictive ML Classifiers (CO2 & CO4):** Train and benchmark **KNN, Decision Tree, SVM, and Logistic Regression**.
            - **Salary Package Regression:** Predict LPA packages using Linear Regression & Ensembles.
            - **Student Segmentation (CO3):** K-Means clustering with Elbow Method curve to segment student cohorts.
            - **Export Center:** Download preprocessed CSVs and printable executive HTML dossiers.
            """)
        with col_feat2:
            st.markdown("""
            ### 👨‍🎓 For Students
            - **Eligibility Tracker:** Instant corporate recruitment eligibility across Super Dream, Dream, and Core tiers.
            - **Interactive Simulator:** Test 'what-if' scenarios to see how CGPA or internships boost placement probability.
            - **Skill Gap Radar:** Multivariate radar chart benchmarking your profile against placed peers.
            - **Actionable Roadmap:** 4-Phase weekly milestone roadmap to ensure high placement readiness.
            """)

        # Display sample dataset preview
        df = st.session_state.get("raw_df")
        if df is not None:
            st.markdown("---")
            st.subheader("📊 Historical Benchmark Dataset Preview (650 Lab Records)")
            st.dataframe(df.head(6), use_container_width=True)

    else:
        # Route based on selected section
        if user["role"] == "admin":
            if selected_section == "📊 Executive Dashboard":
                render_admin_dashboard()
            elif selected_section == "🧹 Data Preprocessing & EDA (CO1)":
                render_admin_preprocessing()
            elif selected_section == "🤖 Classification & Regression (CO2 & CO4)":
                render_admin_models()
            elif selected_section == "🎯 Student Segmentation & K-Means (CO3)":
                render_admin_clustering()
            elif selected_section == "👥 User Management":
                render_admin_users()
            elif selected_section == "📑 Reports & Data Export":
                render_admin_reports()
            else:
                render_admin_dashboard()
        else:  # Student role
            if selected_section == "🎓 My Profile & Eligibility":
                render_student_profile()
            elif selected_section == "🔮 Placement & Package Simulator":
                render_student_simulator()
            elif selected_section == "🎯 Skill Gap & Radar Benchmark":
                render_student_skill_gap()
            elif selected_section == "🚀 Targeted Improvement Plan":
                render_student_action_plan()
            else:
                render_student_profile()


if __name__ == "__main__":
    main()
