"""
Predictive Modeling and Machine Learning Suite (CO2 & CO4 Syllabus Alignment)
Covers:
1. Classification Algorithms:
   - K-Nearest Neighbors (KNN)
   - Decision Tree Classifier
   - Support Vector Machine (SVM)
   - Logistic Regression
2. Comprehensive Evaluation Metrics:
   - Accuracy, Precision, Recall, F1-Score (binary & macro)
   - Confusion Matrix (TP, TN, FP, FN)
   - ROC Curve and AUC Score
   - K-Fold Cross Validation (Mean & Std)
3. Continuous Outcome (Package) Prediction:
   - Linear Regression & Ridge/Random Forest
   - MAE, RMSE, R-squared evaluation metrics
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, mean_absolute_error,
    mean_squared_error, r2_score
)


class PlacementClassifierSuite:
    """Manages training, cross-validation, and benchmarking of placement classifiers."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models: Dict[str, Any] = {}
        self.results: Dict[str, Dict[str, Any]] = {}
        self.best_model_name: Optional[str] = None
        self.feature_names: List[str] = []
        self.classes: List[str] = ["Not Placed", "Placed"]

    def _init_models(
        self, 
        knn_neighbors: int = 5, 
        dt_max_depth: Optional[int] = 5, 
        svm_kernel: str = "rbf"
    ):
        self.models = {
            "Logistic Regression": LogisticRegression(
                max_iter=1000, 
                random_state=self.random_state,
                C=1.0
            ),
            "K-Nearest Neighbors (KNN)": KNeighborsClassifier(
                n_neighbors=knn_neighbors,
                weights="distance"
            ),
            "Decision Tree": DecisionTreeClassifier(
                max_depth=dt_max_depth, 
                criterion="gini", 
                random_state=self.random_state
            ),
            "Support Vector Machine (SVM)": CalibratedClassifierCV(
                SVC(
                    kernel=svm_kernel, 
                    random_state=self.random_state,
                    C=1.0
                ),
                cv=3
            )
        }

    def train_and_evaluate_all(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        feature_names: List[str],
        test_size: float = 0.2, 
        cv_folds: int = 5,
        knn_neighbors: int = 5,
        dt_max_depth: Optional[int] = 5,
        svm_kernel: str = "rbf"
    ) -> Dict[str, Any]:
        """
        Trains and benchmarks all 4 classification algorithms.
        Returns a comprehensive comparison dictionary and summary DataFrame.
        """
        self.feature_names = feature_names
        self._init_models(knn_neighbors, dt_max_depth, svm_kernel)

        # Stratified train/test split to preserve placement class ratio
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        comparison_list = []
        detailed_results = {}
        best_f1 = -1.0

        for name, model in self.models.items():
            # Fit model
            model.fit(X_train, y_train)

            # Predictions
            y_pred = model.predict(X_test)
            
            # Probability estimates for ROC/AUC
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)[:, 1]
            else:
                y_proba = model.decision_function(X_test)

            # Compute standard metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            # Confusion matrix: [[TN, FP], [FN, TP]]
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()

            # ROC Curve & AUC
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)

            # K-Fold Stratified Cross-Validation
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
            cv_f1_scores = cross_val_score(model, X, y, cv=cv, scoring="f1")

            # Feature importances / coefficients if applicable
            feature_imp = {}
            if name == "Decision Tree":
                feature_imp = dict(zip(feature_names, model.feature_importances_))
            elif name == "Logistic Regression":
                feature_imp = dict(zip(feature_names, model.coef_[0]))

            model_data = {
                "name": name,
                "accuracy": round(acc * 100, 2),
                "precision": round(prec * 100, 2),
                "recall": round(rec * 100, 2),
                "f1_score": round(f1 * 100, 2),
                "roc_auc": round(roc_auc, 3),
                "cv_mean_acc": round(cv_scores.mean() * 100, 2),
                "cv_std_acc": round(cv_scores.std() * 100, 2),
                "cv_mean_f1": round(cv_f1_scores.mean() * 100, 2),
                "confusion_matrix": {
                    "raw": cm.tolist(),
                    "tn": int(tn),
                    "fp": int(fp),
                    "fn": int(fn),
                    "tp": int(tp)
                },
                "roc_curve": {
                    "fpr": fpr.tolist(),
                    "tpr": tpr.tolist()
                },
                "feature_importance": feature_imp
            }
            detailed_results[name] = model_data

            comparison_list.append({
                "Algorithm": name,
                "Accuracy (%)": model_data["accuracy"],
                "Precision (%)": model_data["precision"],
                "Recall (%)": model_data["recall"],
                "F1-Score (%)": model_data["f1_score"],
                "ROC-AUC": model_data["roc_auc"],
                "CV Accuracy (%)": f"{model_data['cv_mean_acc']} ± {model_data['cv_std_acc']}%"
            })

            if f1 > best_f1:
                best_f1 = f1
                self.best_model_name = name

        self.results = detailed_results
        comparison_df = pd.DataFrame(comparison_list).sort_values(by="F1-Score (%)", ascending=False)

        return {
            "comparison_df": comparison_df,
            "detailed_results": detailed_results,
            "best_model_name": self.best_model_name,
            "test_data": (X_test, y_test),
            "train_data": (X_train, y_train)
        }

    def predict_single(
        self, 
        model_name: Optional[str], 
        features_vector: np.ndarray
    ) -> Tuple[str, float]:
        """
        Predicts placement for a single student feature vector.
        Returns ('Placed' or 'Not Placed', confidence_percentage)
        """
        chosen_name = model_name or self.best_model_name or list(self.models.keys())[0]
        model = self.models.get(chosen_name)
        if not model:
            raise ValueError(f"Model '{chosen_name}' is not trained yet.")

        vec = features_vector.reshape(1, -1)
        pred = model.predict(vec)[0]

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(vec)[0][1]
        else:
            proba = 1.0 if pred == 1 else 0.0

        label = "Placed" if pred == 1 else "Not Placed"
        conf = (proba if pred == 1 else (1.0 - proba)) * 100
        return label, round(conf, 1)


class PackageRegressionSuite:
    """Predicts continuous salary packages (LPA) for placed students using Linear Regression & Ensemble."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.linear_model: Optional[LinearRegression] = None
        self.ridge_model: Optional[Ridge] = None
        self.rf_model: Optional[RandomForestRegressor] = None
        self.metrics: Dict[str, Any] = {}
        self.feature_names: List[str] = []

    def train_and_evaluate(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        feature_names: List[str],
        test_size: float = 0.2
    ) -> Dict[str, Any]:
        """
        Trains Linear Regression, Ridge, and Random Forest models on salary packages.
        """
        self.feature_names = feature_names
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )

        self.linear_model = LinearRegression()
        self.linear_model.fit(X_train, y_train)
        y_pred_lr = self.linear_model.predict(X_test)

        self.ridge_model = Ridge(alpha=1.0)
        self.ridge_model.fit(X_train, y_train)
        y_pred_ridge = self.ridge_model.predict(X_test)

        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=self.random_state, max_depth=6)
        self.rf_model.fit(X_train, y_train)
        y_pred_rf = self.rf_model.predict(X_test)

        # Compute metrics for Linear Regression
        mae_lr = mean_absolute_error(y_test, y_pred_lr)
        rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
        r2_lr = r2_score(y_test, y_pred_lr)

        # Compute metrics for Ridge Regression
        mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
        rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
        r2_ridge = r2_score(y_test, y_pred_ridge)

        # Compute metrics for Random Forest
        mae_rf = mean_absolute_error(y_test, y_pred_rf)
        rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
        r2_rf = r2_score(y_test, y_pred_rf)

        # 5-Fold Cross Validation R2
        cv = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        cv_r2_lr = cross_val_score(self.linear_model, X, y, cv=cv, scoring="r2")
        cv_r2_rf = cross_val_score(self.rf_model, X, y, cv=cv, scoring="r2")

        # Coefficients for Linear Regression
        coef_dict = dict(zip(feature_names, self.linear_model.coef_))

        self.metrics = {
            "linear_regression": {
                "mae": round(float(mae_lr), 2),
                "rmse": round(float(rmse_lr), 2),
                "r2_score": round(float(r2_lr), 3),
                "cv_r2": round(float(cv_r2_lr.mean()), 3),
                "intercept": round(float(self.linear_model.intercept_), 2),
                "coefficients": coef_dict
            },
            "ridge_regression": {
                "mae": round(float(mae_ridge), 2),
                "rmse": round(float(rmse_ridge), 2),
                "r2_score": round(float(r2_ridge), 3)
            },
            "random_forest": {
                "mae": round(float(mae_rf), 2),
                "rmse": round(float(rmse_rf), 2),
                "r2_score": round(float(r2_rf), 3),
                "cv_r2": round(float(cv_r2_rf.mean()), 3),
                "feature_importances": dict(zip(feature_names, self.rf_model.feature_importances_))
            },
            "actual_vs_predicted": {
                "actual": y_test.tolist()[:50],
                "predicted": [round(float(p), 2) for p in y_pred_lr.tolist()[:50]]
            }
        }
        return self.metrics

    def predict_package(self, feature_vector: np.ndarray, model_type: str = "linear") -> Tuple[float, float, float]:
        """
        Predicts expected package (LPA) along with a realistic confidence interval.
        Returns (predicted_lpa, lower_bound, upper_bound)
        """
        vec = feature_vector.reshape(1, -1)
        if model_type == "rf" and self.rf_model is not None:
            pred = float(self.rf_model.predict(vec)[0])
            mae = self.metrics.get("random_forest", {}).get("mae", 1.2)
        else:
            if self.linear_model is None:
                raise ValueError("Regression model is not trained yet.")
            pred = float(self.linear_model.predict(vec)[0])
            mae = self.metrics.get("linear_regression", {}).get("mae", 1.2)

        pred = max(3.0, pred)
        low = max(3.0, pred - mae)
        high = pred + mae
        return round(pred, 2), round(low, 2), round(high, 2)
