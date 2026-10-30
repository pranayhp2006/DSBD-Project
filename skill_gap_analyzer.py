"""
Skill Gap Analysis and Recommendation Engine
Calculates student skill benchmarks against placed cohort averages,
generates radar chart dimensions, and yields prioritized action items.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np


class SkillGapAnalyzer:
    # Standard benchmarks for target placement readiness
    BENCHMARKS = {
        "CGPA": 8.0,
        "Coding_Score": 75.0,
        "Aptitude_Score": 72.0,
        "Soft_Skills_Score": 75.0,
        "Internships": 1.5,
        "Projects_Count": 3.0,
        "Certifications_Count": 2.0,
        "Hackathons_Attended": 1.5,
        "Backlogs": 0.0
    }

    # Recruitment eligibility policy rules
    COMPANY_TIERS = [
        {
            "tier": "Super Dream (20+ LPA)",
            "min_cgpa": 8.5,
            "min_coding": 80.0,
            "min_aptitude": 75.0,
            "max_backlogs": 0,
            "min_internships": 1,
            "companies": ["Google", "Microsoft", "Amazon", "Adobe", "Uber", "Oracle"]
        },
        {
            "tier": "Dream (10 - 20 LPA)",
            "min_cgpa": 7.5,
            "min_coding": 70.0,
            "min_aptitude": 68.0,
            "max_backlogs": 0,
            "min_internships": 0,
            "companies": ["Samsung R&D", "Cisco", "Qualcomm", "Deloitte USI", "Target", "Walmart Global"]
        },
        {
            "tier": "Core & IT Services (4.5 - 10 LPA)",
            "min_cgpa": 6.5,
            "min_coding": 55.0,
            "min_aptitude": 60.0,
            "max_backlogs": 1,
            "min_internships": 0,
            "companies": ["TCS Digital", "Infosys DSE", "Accenture", "Cognizant", "L&T Technology", "Capgemini"]
        },
        {
            "tier": "Mass Recruitment (3.5 - 4.5 LPA)",
            "min_cgpa": 6.0,
            "min_coding": 45.0,
            "min_aptitude": 50.0,
            "max_backlogs": 2,
            "min_internships": 0,
            "companies": ["TCS Ninja", "Wipro Turbo", "HCL Tech", "Tech Mahindra", "Hexaware"]
        }
    ]

    @classmethod
    def evaluate_eligibility(cls, student_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Evaluates student metrics against recruitment tiers."""
        cgpa = float(student_data.get("CGPA", 0.0))
        coding = float(student_data.get("Coding_Score", 0.0))
        aptitude = float(student_data.get("Aptitude_Score", 0.0))
        backlogs = int(student_data.get("Backlogs", 0))
        internships = int(student_data.get("Internships", 0))

        status_list = []
        for tier in cls.COMPANY_TIERS:
            reasons = []
            if cgpa < tier["min_cgpa"]:
                reasons.append(f"CGPA ({cgpa}) < Required {tier['min_cgpa']}")
            if coding < tier["min_coding"]:
                reasons.append(f"Coding Score ({coding}) < Required {tier['min_coding']}")
            if aptitude < tier["min_aptitude"]:
                reasons.append(f"Aptitude Score ({aptitude}) < Required {tier['min_aptitude']}")
            if backlogs > tier["max_backlogs"]:
                reasons.append(f"Active Backlogs ({backlogs}) > Allowed {tier['max_backlogs']}")
            if internships < tier["min_internships"]:
                reasons.append(f"Internships ({internships}) < Required {tier['min_internships']}")

            eligible = len(reasons) == 0
            status_list.append({
                "tier_name": tier["tier"],
                "eligible": eligible,
                "reasons": reasons,
                "companies": tier["companies"],
                "criteria": f"CGPA ≥ {tier['min_cgpa']} | Coding ≥ {tier['min_coding']} | Aptitude ≥ {tier['min_aptitude']} | Backlogs ≤ {tier['max_backlogs']}"
            })
        return status_list

    @classmethod
    def compute_gap_analysis(
        cls, 
        student_data: Dict[str, Any], 
        df_cohort: pd.DataFrame = None
    ) -> Dict[str, Any]:
        """
        Calculates gap between student and cohort benchmark,
        returns normalized radar coordinates and recommendations.
        """
        # If placed cohort data is available, derive dynamic benchmarks
        if df_cohort is not None and not df_cohort.empty and "Placement_Status" in df_cohort.columns:
            placed = df_cohort[df_cohort["Placement_Status"] == "Placed"]
            benchmarks = {
                "CGPA": round(float(placed["CGPA"].mean()), 2),
                "Coding_Score": round(float(placed["Coding_Score"].mean()), 1),
                "Aptitude_Score": round(float(placed["Aptitude_Score"].mean()), 1),
                "Soft_Skills_Score": round(float(placed["Soft_Skills_Score"].mean()), 1),
                "Internships": round(float(placed["Internships"].mean()), 1),
                "Projects_Count": round(float(placed["Projects_Count"].mean()), 1),
                "Certifications_Count": round(float(placed["Certifications_Count"].mean()), 1),
                "Hackathons_Attended": round(float(placed["Hackathons_Attended"].mean()), 1) if "Hackathons_Attended" in placed.columns else 1.5,
                "Backlogs": 0.0
            }
        else:
            benchmarks = cls.BENCHMARKS.copy()

        gaps = []
        radar_categories = ["CGPA", "Coding Skill", "Aptitude", "Soft Skills", "Internships", "Projects", "Certifications"]
        
        # Student raw values
        stu_cgpa = float(student_data.get("CGPA", 0.0))
        stu_coding = float(student_data.get("Coding_Score", 0.0))
        stu_aptitude = float(student_data.get("Aptitude_Score", 0.0))
        stu_soft = float(student_data.get("Soft_Skills_Score", 0.0))
        stu_intern = float(student_data.get("Internships", 0.0))
        stu_proj = float(student_data.get("Projects_Count", 0.0))
        stu_cert = float(student_data.get("Certifications_Count", 0.0))
        stu_backlogs = int(student_data.get("Backlogs", 0))

        # Normalized to 0-100 for radar plot
        student_scores = [
            min(100.0, (stu_cgpa / 10.0) * 100),
            stu_coding,
            stu_aptitude,
            stu_soft,
            min(100.0, (stu_intern / 3.0) * 100),
            min(100.0, (stu_proj / 5.0) * 100),
            min(100.0, (stu_cert / 4.0) * 100)
        ]

        benchmark_scores = [
            min(100.0, (benchmarks["CGPA"] / 10.0) * 100),
            benchmarks["Coding_Score"],
            benchmarks["Aptitude_Score"],
            benchmarks["Soft_Skills_Score"],
            min(100.0, (benchmarks["Internships"] / 3.0) * 100),
            min(100.0, (benchmarks["Projects_Count"] / 5.0) * 100),
            min(100.0, (benchmarks["Certifications_Count"] / 4.0) * 100)
        ]

        # Generate prioritized recommendations
        recommendations = []
        if stu_backlogs > 0:
            recommendations.append({
                "priority": "HIGH",
                "color": "#EF4444",
                "area": "Academic Backlogs",
                "advice": f"You have {stu_backlogs} active backlog(s). Most Tier-1/Dream recruiters strictly require zero backlogs. Clear existing backlogs in supplementary exams immediately."
            })

        if stu_coding < benchmarks["Coding_Score"]:
            gap = benchmarks["Coding_Score"] - stu_coding
            recommendations.append({
                "priority": "HIGH" if gap > 15 else "MEDIUM",
                "color": "#EF4444" if gap > 15 else "#F59E0B",
                "area": "Coding & Problem Solving",
                "advice": f"Coding score is {stu_coding:.1f} vs benchmark {benchmarks['Coding_Score']:.1f} (-{gap:.1f}). Solve 2 medium DSA problems daily on LeetCode/HackerRank focusing on Arrays, Trees, Dynamic Programming."
            })

        if stu_intern < benchmarks["Internships"]:
            recommendations.append({
                "priority": "HIGH",
                "color": "#EF4444",
                "area": "Industry Internships",
                "advice": f"You have {int(stu_intern)} internships vs average {benchmarks['Internships']:.1f}. Target a 6-to-8 week summer internship or contribute to open-source GitHub projects to establish hands-on industry exposure."
            })

        if stu_aptitude < benchmarks["Aptitude_Score"]:
            gap = benchmarks["Aptitude_Score"] - stu_aptitude
            recommendations.append({
                "priority": "MEDIUM",
                "color": "#F59E0B",
                "area": "Quantitative Aptitude",
                "advice": f"Aptitude score is {stu_aptitude:.1f} vs benchmark {benchmarks['Aptitude_Score']:.1f} (-{gap:.1f}). Practice speed-math shortcuts, percentage, probability, and syllogisms on IndiaBIX or Prepleet."
            })

        if stu_cgpa < benchmarks["CGPA"]:
            gap = benchmarks["CGPA"] - stu_cgpa
            recommendations.append({
                "priority": "MEDIUM",
                "color": "#F59E0B",
                "area": "Academic CGPA",
                "advice": f"CGPA is {stu_cgpa:.2f} vs benchmark {benchmarks['CGPA']:.2f}. Aim for 8.5+ in upcoming semester exams to unlock Super-Dream recruitment shortlist cutoffs."
            })

        if stu_proj < benchmarks["Projects_Count"]:
            recommendations.append({
                "priority": "LOW",
                "color": "#3B82F6",
                "area": "Technical Projects",
                "advice": f"Build at least {int(benchmarks['Projects_Count'] - stu_proj)} full-stack or ML capstone projects with clean README docs and deployed live links."
            })

        if not recommendations:
            recommendations.append({
                "priority": "LOW",
                "color": "#10B981",
                "area": "Outstanding Profile",
                "advice": "Your skill profile exceeds the placement benchmarks across all core pillars! Focus on advanced system design and executive mock interviews."
            })

        return {
            "benchmarks": benchmarks,
            "radar_categories": radar_categories,
            "student_radar_scores": [round(s, 1) for s in student_scores],
            "benchmark_radar_scores": [round(b, 1) for b in benchmark_scores],
            "recommendations": recommendations
        }
