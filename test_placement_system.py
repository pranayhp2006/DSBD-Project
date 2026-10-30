"""
Automated Comprehensive Test Suite for Placement Analysis System
Validates all Course Outcome (CO1 - CO4) modules, Authentication, ML Models, and Reporting.
"""

import warnings
warnings.filterwarnings("ignore")

import unittest
import os
import uuid
import pandas as pd
import numpy as np

import auth_manager
from data_processor import PreprocessingPipeline
from ml_models import PlacementClassifierSuite, PackageRegressionSuite
from clustering_engine import StudentClusteringEngine
from skill_gap_analyzer import SkillGapAnalyzer
import report_generator


class TestPlacementAnalysisSystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Generate clean synthetic test data if not already present
        cls.csv_path = "students_placement_dataset.csv"
        if not os.path.exists(cls.csv_path):
            from dataset_generator import generate_placement_dataset
            generate_placement_dataset(200, cls.csv_path)

        cls.df = pd.read_csv(cls.csv_path)
        auth_manager.init_auth_db()

    # 1. Authentication & Role-Based Access Tests
    def test_01_authentication_admin(self):
        user = auth_manager.authenticate_user("admin", "admin123")
        self.assertIsNotNone(user)
        self.assertEqual(user["role"], "admin")
        self.assertEqual(user["username"], "admin")

    def test_02_authentication_student(self):
        user = auth_manager.authenticate_user("pranay", "student123")
        self.assertIsNotNone(user)
        self.assertEqual(user["role"], "student")
        self.assertIsNotNone(user["student_id"])

    def test_03_invalid_login(self):
        user = auth_manager.authenticate_user("admin", "wrong_password_999")
        self.assertIsNone(user)

    def test_04_user_registration(self):
        uname = f"qa_{uuid.uuid4().hex[:6]}"
        ok, msg = auth_manager.register_user(uname, "qa_pass_123", "student", "QA Tester", "qa@edu.in", "STU9999")
        self.assertTrue(ok)
        auth = auth_manager.authenticate_user(uname, "qa_pass_123")
        self.assertIsNotNone(auth)
        self.assertEqual(auth["full_name"], "QA Tester")

    # 2. Data Preprocessing Tests (CO1)
    def test_05_missing_value_detection_and_imputation(self):
        missing_df = PreprocessingPipeline.detect_missing_values(self.df)
        self.assertIsInstance(missing_df, pd.DataFrame)

        # Impute
        clean_df, log = PreprocessingPipeline.impute_missing_values(self.df, num_strategy="mean", cat_strategy="mode")
        self.assertEqual(clean_df.isnull().sum().sum(), 0)

    def test_06_outlier_detection_and_capping(self):
        outliers = PreprocessingPipeline.detect_outliers(self.df, method="iqr")
        self.assertIsInstance(outliers, pd.DataFrame)

        capped_df, _ = PreprocessingPipeline.handle_outliers(self.df, method="iqr", action="cap")
        self.assertEqual(len(capped_df), len(self.df))

    def test_07_encoding_and_scaling(self):
        pipe = PreprocessingPipeline()
        encoded = pipe.encode_features(self.df, method="onehot")
        self.assertNotIn("Branch", encoded.columns)  # Was one-hot encoded
        self.assertTrue(any("Branch_" in col for col in encoded.columns))

        feat_cols = [c for c in encoded.columns if c not in ["Student_ID", "Name", "Placement_Status", "Salary_Package_LPA"]]
        X = encoded[feat_cols].values
        scaled_arr, scaled_df = pipe.scale_features(pd.DataFrame(X, columns=feat_cols), method="standard")
        self.assertEqual(scaled_arr.shape, X.shape)

    def test_08_pca_dimensionality_reduction(self):
        pipe = PreprocessingPipeline()
        clean_df, _ = pipe.impute_missing_values(self.df)
        encoded = pipe.encode_features(clean_df, method="onehot")
        feat_cols = [c for c in encoded.columns if c not in ["Student_ID", "Name", "Placement_Status", "Salary_Package_LPA"]]
        X = encoded[feat_cols].values
        X_scaled, _ = pipe.scale_features(pd.DataFrame(X, columns=feat_cols))

        pca_df, metrics = pipe.compute_pca(X_scaled, feat_cols, n_components=2)
        self.assertEqual(pca_df.shape[1], 2)
        self.assertIn("PC1", pca_df.columns)
        self.assertIn("PC2", pca_df.columns)
        self.assertGreater(metrics["total_explained_variance"], 0.0)

    # 3. Machine Learning Classification & Regression Tests (CO2 & CO4)
    def test_09_classification_benchmark_suite(self):
        pipe = PreprocessingPipeline()
        clean_df, _ = pipe.impute_missing_values(self.df)
        capped_df, _ = pipe.handle_outliers(clean_df, action="cap")
        encoded = pipe.encode_features(capped_df, method="onehot")
        feat_cols = [c for c in encoded.columns if c not in ["Student_ID", "Name", "Placement_Status", "Salary_Package_LPA"]]
        X = encoded[feat_cols].values
        y = (encoded["Placement_Status"] == "Placed").astype(int).values
        X_scaled, _ = pipe.scale_features(pd.DataFrame(X, columns=feat_cols))

        suite = PlacementClassifierSuite()
        res = suite.train_and_evaluate_all(X_scaled, y, feat_cols, test_size=0.2, cv_folds=3)

        self.assertIn("KNN", res["best_model_name"] + " " + " ".join(res["detailed_results"].keys()))
        self.assertIn("Decision Tree", res["detailed_results"])
        self.assertIn("Support Vector Machine (SVM)", res["detailed_results"])
        self.assertIn("Logistic Regression", res["detailed_results"])

        for model_name, m_data in res["detailed_results"].items():
            self.assertGreaterEqual(m_data["accuracy"], 50.0)
            self.assertGreaterEqual(m_data["f1_score"], 50.0)
            self.assertIn("confusion_matrix", m_data)
            self.assertIn("roc_curve", m_data)

        # Single prediction test
        single_label, conf = suite.predict_single(None, X_scaled[0])
        self.assertIn(single_label, ["Placed", "Not Placed"])
        self.assertGreaterEqual(conf, 0.0)

    def test_10_continuous_package_regression(self):
        pipe = PreprocessingPipeline()
        clean_df, _ = pipe.impute_missing_values(self.df)
        encoded = pipe.encode_features(clean_df, method="onehot")
        placed_df = encoded[encoded["Placement_Status"] == "Placed"]
        feat_cols = [c for c in encoded.columns if c not in ["Student_ID", "Name", "Placement_Status", "Salary_Package_LPA"]]

        X = placed_df[feat_cols].values
        y = placed_df["Salary_Package_LPA"].values
        X_scaled, _ = pipe.scale_features(pd.DataFrame(X, columns=feat_cols))

        reg_suite = PackageRegressionSuite()
        res = reg_suite.train_and_evaluate(X_scaled, y, feat_cols)

        self.assertIn("linear_regression", res)
        self.assertGreater(res["linear_regression"]["r2_score"], 0.4)
        self.assertLess(res["linear_regression"]["mae"], 5.0)

        # Predict package test
        pred, low, high = reg_suite.predict_package(X_scaled[0])
        self.assertGreater(pred, 0.0)
        self.assertLessEqual(low, pred)
        self.assertGreaterEqual(high, pred)

    # 4. Student Segmentation & K-Means Clustering Tests (CO3)
    def test_11_kmeans_clustering_and_tiers(self):
        pipe = PreprocessingPipeline()
        clean_df, _ = pipe.impute_missing_values(self.df)
        encoded = pipe.encode_features(clean_df, method="onehot")
        feat_cols = [c for c in encoded.columns if c not in ["Student_ID", "Name", "Placement_Status", "Salary_Package_LPA"]]
        X = encoded[feat_cols].values
        X_scaled, _ = pipe.scale_features(pd.DataFrame(X, columns=feat_cols))

        engine = StudentClusteringEngine()
        elbow = engine.compute_elbow_curve(X_scaled, max_k=4)
        self.assertEqual(len(elbow["k_values"]), 3)
        self.assertEqual(len(elbow["inertias"]), 3)

        c_res = engine.fit_and_profile(X_scaled, clean_df, feat_cols, n_clusters=3)
        self.assertEqual(len(c_res["profiles"]), 3)
        self.assertIn("Performance_Tier", c_res["clustered_df"].columns)
        self.assertIn("PCA_1", c_res["clustered_df"].columns)
        self.assertIn("PCA_2", c_res["clustered_df"].columns)

    # 5. Student Skill Gap Analyzer Tests
    def test_12_skill_gap_analysis(self):
        stu = {
            "CGPA": 7.2,
            "Coding_Score": 55.0,
            "Aptitude_Score": 60.0,
            "Soft_Skills_Score": 65.0,
            "Internships": 0,
            "Projects_Count": 2,
            "Certifications_Count": 1,
            "Backlogs": 1
        }
        gap_res = SkillGapAnalyzer.compute_gap_analysis(stu, self.df)
        self.assertIn("radar_categories", gap_res)
        self.assertIn("student_radar_scores", gap_res)
        self.assertGreater(len(gap_res["recommendations"]), 0)

        eligibility = SkillGapAnalyzer.evaluate_eligibility(stu)
        self.assertEqual(len(eligibility), 4)

    # 6. Report Generation Test
    def test_13_report_generator_html(self):
        pipe = PreprocessingPipeline()
        clean_df, _ = pipe.impute_missing_values(self.df)
        encoded = pipe.encode_features(clean_df, method="onehot")
        feat_cols = [c for c in encoded.columns if c not in ["Student_ID", "Name", "Placement_Status", "Salary_Package_LPA"]]
        X = encoded[feat_cols].values
        y = (encoded["Placement_Status"] == "Placed").astype(int).values
        X_scaled, _ = pipe.scale_features(pd.DataFrame(X, columns=feat_cols))

        cls_suite = PlacementClassifierSuite()
        cls_res = cls_suite.train_and_evaluate_all(X_scaled, y, feat_cols, cv_folds=2)

        engine = StudentClusteringEngine()
        c_res = engine.fit_and_profile(X_scaled, clean_df, feat_cols, n_clusters=3)

        html = report_generator.generate_placement_executive_summary_html(
            clean_df, cls_res["comparison_df"], c_res["profiles"], cls_res["best_model_name"]
        )
        self.assertIn("Placement Analysis & Intelligence System", html)
        self.assertIn("Machine Learning Classification Performance Benchmark", html)
        self.assertIn("Student Segmentation & K-Means Clustering Tiers", html)


if __name__ == "__main__":
    unittest.main()
