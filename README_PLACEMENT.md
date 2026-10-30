# 🎓 Placement Analysis & Intelligence System

A full-stack, data-driven web application built with Python, Streamlit, Scikit-Learn, and Plotly. Designed to incorporate core concepts from the **Data Science and Big Data Lab** curriculum: **Data Preprocessing (CO1)**, **Predictive Classification & Regression (CO2 & CO4)**, and **Student Segmentation & Clustering (CO3)**.

---

## 🚀 Key Highlights & Course Outcome (CO) Mapping

| Module / Requirement | Syllabus Outcome | Implementation Architecture |
|---|---|---|
| **Role-Based Authentication** | System Security | SQLite database with salted SHA-256 password hashing. Multi-role access control (`Admin/Placement Cell` vs `Student`). |
| **Data Preprocessing** | **CO1** | Missing value imputation (Mean/Median/Mode), Outlier detection & Winsorization (IQR & Z-score), One-Hot/Label encoding, Feature scaling (StandardScaler/MinMax), and PCA dimensionality reduction. |
| **Predictive Modeling & Classification** | **CO2 & CO4** | 4-Algorithm Benchmark: **KNN, Decision Tree, Support Vector Machine (SVM), and Logistic Regression**. Full metrics: Confusion Matrix heatmaps, Accuracy, Precision, Recall, F1-Score, Cross-Validation, and ROC-AUC curves. |
| **Package Regression** | **CO2 & CO4** | Continuous salary prediction (in LPA) using Linear Regression, Ridge, and Random Forest with MAE, RMSE, $R^2$, and feature importance insights. |
| **Student Segmentation & Clustering** | **CO3** | Unsupervised **K-Means Clustering** with interactive **Elbow Method (WCSS)** and Silhouette scores. Automated assignment to **Tier 1 (Top Performers)**, **Tier 2 (Moderate)**, and **Tier 3 (Needs Foundation)** with institutional intervention blueprints. |
| **Student Self-Service Portal** | Applied Analytics | Personalized recruitment tier eligibility checker (Super Dream, Dream, Core IT, Mass), 'What-if' placement simulator, multivariate Skill Gap Radar Chart, and 4-phase milestone roadmap. |
| **Export & Reporting** | Reporting | 1-Click downloads of Cleaned CSVs, Clustered Student Segmentations with Tiers, Model Performance logs, and printable Executive Placement Dossiers (HTML). |

---

## 📂 Project Architecture

```
DBMS project/
├── placement_app.py              # Main Streamlit web application & routing
├── auth_manager.py               # SQLite user database, salted SHA-256 hashing & RBAC
├── data_processor.py             # PreprocessingPipeline: Imputation, Outliers, Encoding, PCA (CO1)
├── ml_models.py                  # Classification (KNN, DT, SVM, LR) & Salary Regression (CO2/CO4)
├── clustering_engine.py          # K-Means clustering, Elbow method, Tier profiling (CO3)
├── skill_gap_analyzer.py         # Student competency benchmarking, Radar scores & eligibility
├── report_generator.py           # Printable HTML dossier & CSV report exporter
├── dataset_generator.py          # Synthetic dataset generator for historical student data (650 records)
├── test_placement_system.py      # Automated unittest suite covering all CO modules (13 test cases)
├── run_placement_app.bat         # Windows double-click application launcher
├── placement_users.db            # SQLite user database
└── students_placement_dataset.csv# Pre-generated benchmark dataset (650 records)
```

---

## 🔑 Pre-seeded Login Credentials

| Role | Username | Password | Profile Description |
|---|---|---|---|
| **Admin / Placement Cell** | `admin` | `admin123` | Prof. K. Sharma (Head of Corporate Relations & TPO) |
| **Admin / Coordinator** | `coordinator` | `coord123` | Dr. A. Verma (Department TPO Coordinator) |
| **Student** | `pranay` | `student123` | Pranay Kumar (Roll ID: `STU202401`) |
| **Student** | `rahul` | `student123` | Rahul Gupta (Roll ID: `STU202402`) |
| **Student** | `sneha` | `student123` | Sneha Patil (Roll ID: `STU202403`) |

> 💡 **Quick Logins**: The sidebar provides 1-click **Quick Demo Login** buttons so you can switch between Admin and Student views instantly during presentations and evaluations!

---

## ⚡ How to Run the Application

### Method 1: Using the Batch Launcher (Windows)
Double-click `run_placement_app.bat` or run:
```cmd
run_placement_app.bat
```

### Method 2: Command Line
```powershell
python -m streamlit run placement_app.py
```
Then open your browser at `http://localhost:8501`.

---

## 🧪 Running Automated Unit Tests
To verify all 13 test cases across CO1 to CO4:
```powershell
python test_placement_system.py
```

Expected output:
```
.............
----------------------------------------------------------------------
Ran 13 tests in 2.7s

OK
```

---

## 📊 Modules & Workflow Walkthrough

### 1. 🏫 Admin & Placement Cell Capabilities
1. **Executive Dashboard:**
   - Real-time KPIs: Total Candidates, Placement Ratio (%), Average Offer Package (LPA), Highest Package.
   - Interactive Plotly figures: Placed vs Unplaced donut chart, Branch-wise distribution, CGPA vs Coding competence scatter plot, and Salary Package histogram.
2. **Data Preprocessing & EDA (CO1):**
   - Upload new historical datasets (CSV / Excel) or reset to the 650-record lab benchmark.
   - Missing value detector: Shows missing counts and percentages per feature; choose Mean, Median, Mode, or Drop strategy.
   - Outlier detector: Inspect box plots, toggle between **IQR Fence** and **Z-Score (|z| > 3.0)**, and choose **Winsorization (Cap)** or **Drop Rows**.
   - Categorical Encoding: Toggle between One-Hot and Label Encoding.
   - Feature Scaling & PCA: Select StandardScaler vs MinMaxScaler, adjust PCA component slider, view the **Scree Plot (Explained Variance %)**, and inspect the 2D PCA cluster projection and eigenvector loading matrix.
3. **Predictive Modeling & Model Evaluation (CO2 & CO4):**
   - Retrain models with custom hyperparameter sliders (Test Split, CV Folds, KNN $k$, Decision Tree Max Depth, SVM Kernel).
   - Side-by-side benchmark table: **Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Cross-Validation (Mean ± Std)**.
   - Diagnostics tabs:
     - **Confusion Matrices:** Interactive heatmaps displaying True Positives, True Negatives, False Positives, and False Negatives for all 4 models.
     - **ROC Curves:** Multi-model comparative ROC curves on a single canvas with AUC values.
     - **Feature Importances:** Gini importance bar chart highlighting top student placement determinants.
   - **Continuous Package Regression:** Linear Regression, Ridge, and Random Forest predicting packages in LPA with $R^2$, MAE, RMSE, and an Actual vs Predicted scatter plot.
4. **Student Segmentation & Clustering (CO3):**
   - **Elbow Curve:** Within-Cluster Sum of Squares (Inertia) and Silhouette Scores across $k=2$ to $k=8$.
   - **Performance Tiers:** Clusters automatically ranked into **Tier 1 (Top Performers)**, **Tier 2 (Moderate)**, and **Tier 3 (Needs Foundation)**.
   - **2D & 3D Interactive PCA Cluster Plots:** Hover over student markers to see Roll ID, Name, CGPA, and Tier.
   - **Placement Cell Intervention Blueprints:** Specific training recommendations tailored for each tier.
5. **User Management:**
   - View, create, and manage system accounts with encrypted credentials.
6. **Reports & Data Export:**
   - Download Clean Preprocessed Dataset (CSV).
   - Download Clustered Student Cohort with Tier Tags (CSV).
   - Download Model Performance Benchmark Table (CSV).
   - Download Printable Placement Executive Dossier (HTML).

### 2. 👨‍🎓 Student Portal Capabilities
1. **My Profile & Recruitment Eligibility:**
   - Instant eligibility status across 4 corporate brackets:
     - **Super Dream (20+ LPA):** Google, Microsoft, Amazon, Adobe, Uber.
     - **Dream (10 - 20 LPA):** Samsung R&D, Cisco, Qualcomm, Deloitte USI.
     - **Core & IT Services (4.5 - 10 LPA):** TCS Digital, Infosys DSE, Accenture.
     - **Mass Recruitment (3.5 - 4.5 LPA):** TCS Ninja, Wipro Turbo, HCL Tech.
   - Clear shortfall breakdown explaining exactly why an eligibility gate was not met (e.g. CGPA, active backlogs, coding threshold).
2. **Placement & Package Simulator:**
   - Interactive sliders for CGPA, Coding Score, Aptitude, Soft Skills, Internships, Projects, and Backlogs.
   - Instant prediction of placement probability and predicted offer package (e.g., `₹ 8.4 LPA (Range: ₹ 7.2 - ₹ 9.6 LPA)`).
3. **Personalized Skill Gap Radar:**
   - Interactive 7-axis Plotly Radar Chart comparing the student's profile directly against the benchmark of placed students.
   - Prioritized, color-coded improvement recommendations (High, Medium, Low priority).
4. **Targeted 4-Phase Roadmap:**
   - Interactive checklist with weekly milestones across 16 weeks to achieve top-tier placement readiness.
