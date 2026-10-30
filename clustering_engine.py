"""
Student Segmentation & Clustering Engine (CO3 Syllabus Alignment)
Covers:
1. K-Means Clustering for unsupervised student segmentation.
2. Elbow Method (Inertia/WCSS) and Silhouette Score analysis for optimal K selection.
3. Automated tier assignment (Top Performers, Moderate, Needs Improvement).
4. Placement Cell targeted intervention recommendations.
5. 2D/3D PCA cluster projection for interactive visualization.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


class StudentClusteringEngine:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.kmeans: Optional[KMeans] = None
        self.n_clusters: int = 3
        self.cluster_labels: Optional[np.ndarray] = None
        self.cluster_profiles: List[Dict[str, Any]] = []
        self.elbow_data: Dict[str, Any] = {}
        self.tier_names: Dict[int, str] = {}
        self.recommendations: Dict[str, List[str]] = {}

    def compute_elbow_curve(
        self, 
        X_scaled: np.ndarray, 
        max_k: int = 8
    ) -> Dict[str, Any]:
        """
        Computes inertia (WCSS) and silhouette scores across k in [2, max_k]
        to identify the optimal number of clusters using the Elbow Method.
        """
        k_values = list(range(2, max_k + 1))
        inertias = []
        silhouettes = []

        for k in k_values:
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            km.fit(X_scaled)
            inertias.append(float(km.inertia_))
            sil = float(silhouette_score(X_scaled, km.labels_))
            silhouettes.append(round(sil, 3))

        self.elbow_data = {
            "k_values": k_values,
            "inertias": inertias,
            "silhouettes": silhouettes
        }
        return self.elbow_data

    def fit_and_profile(
        self, 
        X_scaled: np.ndarray, 
        original_df: pd.DataFrame, 
        feature_columns: List[str],
        n_clusters: int = 3
    ) -> Dict[str, Any]:
        """
        Executes K-Means, ranks clusters by overall skill composite,
        assigns semantic tier labels, and creates targeted training blueprints.
        """
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        self.cluster_labels = self.kmeans.fit_predict(X_scaled)

        df = original_df.copy()
        df["Cluster"] = self.cluster_labels

        # Rank clusters based on composite skill score (CGPA + Coding + Aptitude)
        cluster_metrics = []
        for c in range(n_clusters):
            c_data = df[df["Cluster"] == c]
            cgpa_mean = c_data["CGPA"].mean() if "CGPA" in c_data.columns else 0.0
            coding_mean = c_data["Coding_Score"].mean() if "Coding_Score" in c_data.columns else 0.0
            apt_mean = c_data["Aptitude_Score"].mean() if "Aptitude_Score" in c_data.columns else 0.0
            placed_pct = 0.0
            if "Placement_Status" in c_data.columns:
                placed_pct = (c_data["Placement_Status"] == "Placed").mean() * 100

            # Composite skill index
            composite_score = (cgpa_mean / 10.0) * 0.35 + (coding_mean / 100.0) * 0.40 + (apt_mean / 100.0) * 0.25

            cluster_metrics.append({
                "cluster_id": c,
                "composite_score": composite_score,
                "count": len(c_data),
                "cgpa_mean": round(float(cgpa_mean), 2),
                "coding_mean": round(float(coding_mean), 1),
                "apt_mean": round(float(apt_mean), 1),
                "placed_pct": round(float(placed_pct), 1),
                "internships_mean": round(float(c_data["Internships"].mean()), 1) if "Internships" in c_data.columns else 0.0,
                "projects_mean": round(float(c_data["Projects_Count"].mean()), 1) if "Projects_Count" in c_data.columns else 0.0
            })

        # Sort from highest performing to lowest performing
        sorted_clusters = sorted(cluster_metrics, key=lambda x: x["composite_score"], reverse=True)

        tier_mapping = {}
        profiles = []
        cluster_tier_names = {}

        for rank, cm in enumerate(sorted_clusters):
            cid = cm["cluster_id"]
            if rank == 0:
                tier_label = "Tier 1: Top Performers (High Readiness)"
                badge_color = "#10B981"  # Emerald
                focus = "High-tier Dream/Super-Dream packages (15-45 LPA)"
                recs = [
                    "Advanced System Design (LLD & HLD) workshops",
                    "Competitive Programming & FAANG-level mock coding rounds",
                    "Hackathon mentorship & high-impact open source portfolio reviews",
                    "Executive communication & leadership interview grooming"
                ]
            elif rank == len(sorted_clusters) - 1 and len(sorted_clusters) > 1:
                tier_label = "Tier 3: Needs Technical Foundation"
                badge_color = "#EF4444"  # Red
                focus = "Core employability & mass recruitment eligibility (3.5 - 7 LPA)"
                recs = [
                    "Intensive Data Structures & Algorithms bootcamps (Arrays, Trees, Graphs)",
                    "Mandatory weekly quantitative aptitude & logical reasoning practice",
                    "Assistance in academic backlog clearance & CGPA recovery strategies",
                    "Pair-programming mini projects in Full Stack or Python"
                ]
            else:
                tier_label = f"Tier 2: Moderate Competence (Group {rank})"
                badge_color = "#F59E0B"  # Amber
                focus = "Transitioning to Tier 1 / securing Product & Service offers (6-14 LPA)"
                recs = [
                    "Medium-level LeetCode problem solving drills & timing optimization",
                    "Mock technical interviews with alumni and industry mentors",
                    "Resume tailoring with quantified project impact bullets",
                    "Group discussion and HR behavioral communication training"
                ]

            cluster_tier_names[cid] = tier_label
            profiles.append({
                "cluster_id": cid,
                "tier_label": tier_label,
                "badge_color": badge_color,
                "focus_area": focus,
                "metrics": cm,
                "recommendations": recs
            })

        self.tier_names = cluster_tier_names
        self.cluster_profiles = profiles
        df["Performance_Tier"] = df["Cluster"].map(cluster_tier_names)

        # 2D PCA for cluster visualization
        pca2 = PCA(n_components=2, random_state=self.random_state)
        coords2d = pca2.fit_transform(X_scaled)
        df["PCA_1"] = coords2d[:, 0]
        df["PCA_2"] = coords2d[:, 1]

        # 3D PCA coordinates if available
        if X_scaled.shape[1] >= 3:
            pca3 = PCA(n_components=3, random_state=self.random_state)
            coords3d = pca3.fit_transform(X_scaled)
            df["PCA_3"] = coords3d[:, 2]

        return {
            "clustered_df": df,
            "profiles": profiles,
            "tier_mapping": cluster_tier_names,
            "pca_variance_2d": [round(float(v) * 100, 2) for v in pca2.explained_variance_ratio_]
        }

    def predict_student_tier(self, scaled_vector: np.ndarray) -> Dict[str, Any]:
        """Assigns an incoming student to a cluster and provides tailored recommendations."""
        if self.kmeans is None:
            raise ValueError("K-Means model is not trained yet.")

        vec = scaled_vector.reshape(1, -1)
        cid = int(self.kmeans.predict(vec)[0])
        tier_label = self.tier_names.get(cid, f"Cluster {cid}")

        # Find matching profile
        profile = next((p for p in self.cluster_profiles if p["cluster_id"] == cid), None)

        return {
            "cluster_id": cid,
            "tier_label": tier_label,
            "profile": profile
        }
