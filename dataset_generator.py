"""
Synthetic Placement Dataset Generator
Generates realistic historical student placement data covering:
- Academic performance (CGPA, Backlogs)
- Technical competence (Coding scores, Certifications, Projects, Hackathons)
- Industry readiness (Internships, Soft Skills, Aptitude)
- Classification target: Placement_Status ('Placed' / 'Not Placed')
- Regression target: Salary_Package_LPA (Continuous in Lakhs Per Annum)
- Realistic missing values and outliers for Data Preprocessing (CO1) demos.
"""

import random
import csv
import os
import math

FIRST_NAMES = [
    "Aarav", "Aditi", "Akhil", "Ananya", "Arjun", "Bhavna", "Chetan", "Deepa",
    "Divya", "Gaurav", "Harsh", "Isha", "Kavya", "Manish", "Neha", "Nikhil",
    "Pooja", "Pranay", "Priya", "Rahul", "Rhea", "Rohan", "Sakshi", "Sameer",
    "Sanjay", "Shreya", "Siddharth", "Sneha", "Tanvi", "Varun", "Vikas", "Yash"
]

LAST_NAMES = [
    "Sharma", "Verma", "Patil", "Reddy", "Gupta", "Mehta", "Iyer", "Nair",
    "Singh", "Choudhury", "Joshi", "Kulkarni", "Deshmukh", "Bhat", "Rao",
    "Malhotra", "Kapoor", "Mishra", "Saxena", "Agarwal", "Bose", "Ghosh"
]

BRANCHES = [
    ("Computer Science", 0.35),
    ("Information Technology", 0.25),
    ("Electronics & Comm", 0.20),
    ("Electrical & Electronics", 0.10),
    ("Mechanical", 0.10)
]

def sample_branch():
    r = random.random()
    cumulative = 0.0
    for branch, prob in BRANCHES:
        cumulative += prob
        if r <= cumulative:
            return branch
    return "Computer Science"

def generate_placement_dataset(num_records: int = 600, output_path: str = "students_placement_dataset.csv"):
    random.seed(42)
    records = []

    for i in range(1, num_records + 1):
        stu_id = f"STU{202400 + i:05d}"
        fname = random.choice(FIRST_NAMES)
        lname = random.choice(LAST_NAMES)
        full_name = f"{fname} {lname}"
        gender = random.choice(["Male", "Female", "Male", "Female", "Other"])
        branch = sample_branch()

        # Latent student aptitude / diligence factor (0.0 to 1.0)
        latent_ability = random.betavariate(2.8, 2.2)

        # Academic features
        cgpa = round(min(9.9, max(5.0, 5.0 + latent_ability * 4.8 + random.gauss(0, 0.35))), 2)
        backlogs = 0
        if latent_ability < 0.35:
            backlogs = random.choices([0, 1, 2, 3], weights=[0.4, 0.3, 0.2, 0.1])[0]
        elif latent_ability < 0.6:
            backlogs = random.choices([0, 1], weights=[0.85, 0.15])[0]

        # Technical & Skill features
        coding_score = round(min(99.0, max(25.0, 25.0 + latent_ability * 68.0 + random.gauss(0, 5.0))), 1)
        aptitude_score = round(min(99.0, max(30.0, 30.0 + latent_ability * 65.0 + random.gauss(0, 6.0))), 1)
        soft_skills = round(min(98.0, max(35.0, 40.0 + random.random() * 55.0)), 1)
        
        internships = int(min(4, max(0, round(latent_ability * 3.5 + random.gauss(0, 0.6)))))
        projects = int(min(6, max(1, round(latent_ability * 5.0 + random.gauss(0, 0.8)))))
        certifications = int(min(5, max(0, round(latent_ability * 4.0 + random.gauss(0, 0.7)))))
        hackathons = int(min(4, max(0, round(latent_ability * 3.0 + random.gauss(0, 0.6)))))
        workshops = int(min(6, max(0, random.randint(0, 4) + (1 if latent_ability > 0.5 else 0))))

        # Placement Probability calculation (logistic function)
        branch_bias = 0.5 if branch in ["Computer Science", "Information Technology"] else (0.1 if branch == "Electronics & Comm" else -0.3)
        z = (
            (cgpa - 7.0) * 0.95 +
            (coding_score - 60.0) * 0.05 +
            (aptitude_score - 60.0) * 0.04 +
            internships * 0.75 +
            projects * 0.35 +
            certifications * 0.25 +
            hackathons * 0.3 -
            backlogs * 1.5 +
            branch_bias +
            random.gauss(0, 0.4)
        )
        placement_prob = 1.0 / (1.0 + math.exp(-z))
        placed = placement_prob >= 0.5

        # Continuous target: Salary Package in LPA
        if placed:
            base_package = 4.0
            package_boost = (
                (cgpa - 6.5) * 1.8 +
                (coding_score - 50.0) * 0.28 +
                (aptitude_score - 50.0) * 0.12 +
                internships * 2.2 +
                hackathons * 1.5 +
                (2.0 if branch in ["Computer Science", "Information Technology"] else 0.5) +
                random.gauss(0, 1.2)
            )
            salary_lpa = round(max(3.6, min(44.0, base_package + package_boost)), 2)
            placement_status = "Placed"
        else:
            salary_lpa = 0.0
            placement_status = "Not Placed"

        record = {
            "Student_ID": stu_id,
            "Name": full_name,
            "Gender": gender,
            "Branch": branch,
            "CGPA": cgpa,
            "Coding_Score": coding_score,
            "Aptitude_Score": aptitude_score,
            "Soft_Skills_Score": soft_skills,
            "Internships": internships,
            "Projects_Count": projects,
            "Certifications_Count": certifications,
            "Backlogs": backlogs,
            "Workshops_Attended": workshops,
            "Hackathons_Attended": hackathons,
            "Placement_Status": placement_status,
            "Salary_Package_LPA": salary_lpa
        }
        records.append(record)

    # Introduce realistic missing values (in ~3-5% of records for Aptitude, Soft Skills, Certifications)
    for idx in random.sample(range(num_records), int(num_records * 0.04)):
        records[idx]["Aptitude_Score"] = ""
    for idx in random.sample(range(num_records), int(num_records * 0.03)):
        records[idx]["Soft_Skills_Score"] = ""
    for idx in random.sample(range(num_records), int(num_records * 0.02)):
        records[idx]["Certifications_Count"] = ""

    # Introduce a couple of outliers to demonstrate outlier detection in CO1
    outlier_idx_1 = random.randint(0, num_records - 1)
    records[outlier_idx_1]["Coding_Score"] = 150.0  # Clear high outlier
    outlier_idx_2 = random.randint(0, num_records - 1)
    records[outlier_idx_2]["Projects_Count"] = 25   # Clear count outlier

    # Write CSV
    fieldnames = list(records[0].keys())
    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"Generated {num_records} placement records at '{output_path}'.")
    return output_path

if __name__ == "__main__":
    generate_placement_dataset(650, "students_placement_dataset.csv")
