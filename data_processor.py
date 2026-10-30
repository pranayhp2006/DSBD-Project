"""
Data Preprocessing Module (CO1 Syllabus Alignment)
Covers:
1. Missing value detection & imputation (Mean, Median, Mode, Constant, Drop)
2. Outlier detection & handling (IQR fence capping/filtering, Z-score)
3. Categorical variable encoding (One-Hot Encoding, Label Encoding)
4. Feature scaling (StandardScaler, MinMaxScaler)
5. Dimensionality reduction via Principal Component Analysis (PCA)
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from typing import Dict, List, Tuple, Any, Optional


class PreprocessingPipeline:
    def __init__(self):
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler: Optional[Any] = None
        self.pca: Optional[PCA] = None
        self.numeric_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.feature_columns: List[str] = []
        self.encoding_method: str = "onehot"
        self.scaling_method: str = "standard"

    @staticmethod
    def inspect_dataset(df: pd.DataFrame) -> Dict[str, Any]:
        """Returns structural analysis: shape, data types, null counts, duplicate count."""
        summary = {
            "total_rows": int(len(df)),
            "total_columns": int(df.shape[1]),
            "duplicates": int(df.duplicated().sum()),
            "columns": []
        }
        for col in df.columns:
            null_count = int(df[col].isnull().sum() + (df[col] == "").sum())
            null_pct = round((null_count / len(df)) * 100, 2)
            summary["columns"].append({
                "name": col,
                "dtype": str(df[col].dtype),
                "null_count": null_count,
                "null_pct": null_pct,
                "unique_values": int(df[col].nunique())
            })
        return summary

    @staticmethod
    def detect_missing_values(df: pd.DataFrame) -> pd.DataFrame:
        """Returns dataframe detailing missing values per column."""
        stats = []
        for col in df.columns:
            # Check standard NaN and empty string representations
            is_empty_str = (df[col].astype(str).str.strip() == "") & df[col].notnull()
            null_count = int(df[col].isnull().sum() + is_empty_str.sum())
            if null_count > 0:
                stats.append({
                    "Feature": col,
                    "Data Type": str(df[col].dtype),
                    "Missing Values": null_count,
                    "Missing (%)": round((null_count / len(df)) * 100, 2)
                })
        if not stats:
            return pd.DataFrame(columns=["Feature", "Data Type", "Missing Values", "Missing (%)"])
        return pd.DataFrame(stats).sort_values(by="Missing Values", ascending=False)

    @staticmethod
    def impute_missing_values(
        df: pd.DataFrame, 
        num_strategy: str = "median", 
        cat_strategy: str = "mode"
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Imputes missing values using chosen statistical strategy.
        num_strategy: 'mean', 'median', 'mode', or 'drop'
        cat_strategy: 'mode', 'constant', or 'drop'
        """
        processed_df = df.copy()
        imputation_log = {}

        # Replace empty strings with NaN first
        processed_df.replace(r'^\s*$', np.nan, regex=True, inplace=True)

        for col in processed_df.columns:
            if processed_df[col].isnull().sum() > 0:
                if pd.api.types.is_numeric_dtype(processed_df[col]):
                    if num_strategy == "mean":
                        val = processed_df[col].mean()
                        processed_df[col] = processed_df[col].fillna(val)
                        imputation_log[col] = f"Imputed with Mean: {val:.2f}"
                    elif num_strategy == "median":
                        val = processed_df[col].median()
                        processed_df[col] = processed_df[col].fillna(val)
                        imputation_log[col] = f"Imputed with Median: {val:.2f}"
                    elif num_strategy == "mode":
                        val = processed_df[col].mode()[0]
                        processed_df[col] = processed_df[col].fillna(val)
                        imputation_log[col] = f"Imputed with Mode: {val}"
                    elif num_strategy == "drop":
                        processed_df = processed_df.dropna(subset=[col])
                        imputation_log[col] = "Dropped rows with missing values"
                else:
                    if cat_strategy == "mode":
                        val = processed_df[col].mode()[0] if not processed_df[col].mode().empty else "Unknown"
                        processed_df[col] = processed_df[col].fillna(val)
                        imputation_log[col] = f"Imputed with Mode: '{val}'"
                    elif cat_strategy == "constant":
                        processed_df[col] = processed_df[col].fillna("Unknown")
                        imputation_log[col] = "Imputed with 'Unknown'"
                    elif cat_strategy == "drop":
                        processed_df = processed_df.dropna(subset=[col])
                        imputation_log[col] = "Dropped rows with missing values"

        return processed_df, imputation_log

    @staticmethod
    def detect_outliers(
        df: pd.DataFrame, 
        method: str = "iqr", 
        iqr_multiplier: float = 1.5, 
        z_threshold: float = 3.0,
        exclude_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Detects outliers across numeric features using either IQR Fence or Z-Score method.
        """
        exclude = set(exclude_cols or ["Salary_Package_LPA", "id", "Student_ID"])
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
        results = []

        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) < 5:
                continue

            if method == "iqr":
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - (iqr_multiplier * iqr)
                upper_bound = q3 + (iqr_multiplier * iqr)
                outliers = series[(series < lower_bound) | (series > upper_bound)]
                results.append({
                    "Feature": col,
                    "Method": "IQR",
                    "Lower Bound": round(lower_bound, 2),
                    "Upper Bound": round(upper_bound, 2),
                    "Outlier Count": len(outliers),
                    "Outlier (%)": round((len(outliers) / len(series)) * 100, 2),
                    "Min Found": round(series.min(), 2),
                    "Max Found": round(series.max(), 2)
                })
            else:  # Z-score
                mean = series.mean()
                std = series.std()
                if std > 0:
                    z_scores = np.abs((series - mean) / std)
                    outliers = series[z_scores > z_threshold]
                    results.append({
                        "Feature": col,
                        "Method": "Z-Score (>3.0)",
                        "Lower Bound": round(mean - z_threshold * std, 2),
                        "Upper Bound": round(mean + z_threshold * std, 2),
                        "Outlier Count": len(outliers),
                        "Outlier (%)": round((len(outliers) / len(series)) * 100, 2),
                        "Min Found": round(series.min(), 2),
                        "Max Found": round(series.max(), 2)
                    })

        return pd.DataFrame(results)

    @staticmethod
    def handle_outliers(
        df: pd.DataFrame, 
        method: str = "iqr", 
        action: str = "cap", 
        iqr_multiplier: float = 1.5,
        z_threshold: float = 3.0,
        exclude_cols: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Handles outliers:
        action: 'cap' (Winsorize to fences), 'drop' (Remove offending rows), or 'keep'
        """
        if action == "keep":
            return df.copy(), {"status": "Outliers kept as-is"}

        processed_df = df.copy()
        exclude = set(exclude_cols or ["Salary_Package_LPA", "id", "Student_ID"])
        numeric_cols = [c for c in processed_df.select_dtypes(include=[np.number]).columns if c not in exclude]
        action_log = {}

        if action == "cap":
            for col in numeric_cols:
                series = processed_df[col]
                if method == "iqr":
                    q1 = series.quantile(0.25)
                    q3 = series.quantile(0.75)
                    iqr = q3 - q1
                    low = q1 - (iqr_multiplier * iqr)
                    high = q3 + (iqr_multiplier * iqr)
                else:
                    mean = series.mean()
                    std = series.std()
                    low = mean - z_threshold * std
                    high = mean + z_threshold * std

                capped = np.clip(series, low, high)
                changes = (series != capped).sum()
                if changes > 0:
                    processed_df[col] = capped
                    action_log[col] = f"Capped {changes} outlier values to [{low:.2f}, {high:.2f}]"
        elif action == "drop":
            rows_to_drop = set()
            for col in numeric_cols:
                series = processed_df[col]
                if method == "iqr":
                    q1 = series.quantile(0.25)
                    q3 = series.quantile(0.75)
                    iqr = q3 - q1
                    low = q1 - (iqr_multiplier * iqr)
                    high = q3 + (iqr_multiplier * iqr)
                else:
                    mean = series.mean()
                    std = series.std()
                    low = mean - z_threshold * std
                    high = mean + z_threshold * std

                outlier_indices = processed_df[(processed_df[col] < low) | (processed_df[col] > high)].index
                rows_to_drop.update(outlier_indices)
            
            processed_df = processed_df.drop(index=list(rows_to_drop))
            action_log["dropped_rows"] = f"Removed {len(rows_to_drop)} rows containing outliers"

        return processed_df, action_log

    def encode_features(
        self, 
        df: pd.DataFrame, 
        method: str = "onehot", 
        categorical_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Encodes categorical variables using One-Hot or Label Encoding.
        """
        self.encoding_method = method
        encoded_df = df.copy()

        if categorical_cols is None:
            categorical_cols = [c for c in encoded_df.select_dtypes(include=["object", "category", "string", str]).columns 
                                if c not in ["Student_ID", "Name", "Placement_Status"]]

        self.categorical_cols = categorical_cols

        if method == "label":
            for col in categorical_cols:
                le = LabelEncoder()
                encoded_df[col] = le.fit_transform(encoded_df[col].astype(str))
                self.label_encoders[col] = le
        else:  # One-Hot Encoding
            encoded_df = pd.get_dummies(encoded_df, columns=categorical_cols, drop_first=True)

        return encoded_df

    def scale_features(
        self, 
        X: pd.DataFrame, 
        method: str = "standard"
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Scales features using StandardScaler (z = (x - u) / s) or MinMaxScaler (0 to 1).
        """
        self.scaling_method = method
        if method == "minmax":
            self.scaler = MinMaxScaler()
        else:
            self.scaler = StandardScaler()

        scaled_array = self.scaler.fit_transform(X)
        scaled_df = pd.DataFrame(scaled_array, columns=X.columns, index=X.index)
        return scaled_array, scaled_df

    def compute_pca(
        self, 
        scaled_data: np.ndarray, 
        feature_names: List[str], 
        n_components: int = 2
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Computes PCA for dimensionality reduction and returns PC coordinates + variance metrics.
        """
        max_comp = min(scaled_data.shape[1], scaled_data.shape[0], n_components)
        self.pca = PCA(n_components=max_comp)
        transformed = self.pca.fit_transform(scaled_data)

        pc_columns = [f"PC{i+1}" for i in range(max_comp)]
        pca_df = pd.DataFrame(transformed, columns=pc_columns)

        variance_ratio = self.pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(variance_ratio)

        # Feature loadings matrix (eigenvectors)
        loadings = pd.DataFrame(
            self.pca.components_.T,
            columns=pc_columns,
            index=feature_names
        )

        metrics = {
            "n_components": max_comp,
            "explained_variance_ratio": [round(float(v) * 100, 2) for v in variance_ratio],
            "total_explained_variance": round(float(cumulative_variance[-1]) * 100, 2),
            "loadings": loadings
        }

        return pca_df, metrics
