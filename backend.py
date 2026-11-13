
import pandas as pd

# ==========================================
# BIASShield AI - BACKEND ENGINE
# ==========================================

REQUIRED_COLUMNS = [
    "Candidate_ID",
    "Gender",
    "Marks",
    "Skills",
    "Experience",
    "Selected"
]


# ==========================================
# 1. DATASET VALIDATION
# ==========================================

def validate_dataset(data):

    if not all(col in data.columns for col in REQUIRED_COLUMNS):
        return False, "Missing required columns."

    data = data[REQUIRED_COLUMNS].copy()

    if data.empty:
        return False, "Dataset is empty."

    if data.isnull().any().any():
        return False, "Dataset contains empty values."

    data["Candidate_ID"] = data["Candidate_ID"].astype(str).str.strip()
    data["Gender"] = data["Gender"].astype(str).str.strip()

    if data["Candidate_ID"].duplicated().any():
        return False, "Candidate IDs must be unique."

    if (data["Candidate_ID"] == "").any():
        return False, "Candidate IDs cannot be blank."

    if (data["Gender"] == "").any():
        return False, "Gender values cannot be blank."

    for col in ["Marks", "Skills", "Experience", "Selected"]:
        try:
            data[col] = pd.to_numeric(data[col])
        except (ValueError, TypeError):
            return False, f"{col} must contain numeric values."

    if not data["Marks"].between(0, 100).all():
        return False, "Marks must be between 0 and 100."

    if not data["Skills"].between(0, 10).all():
        return False, "Skills must be between 0 and 10."

    if (data["Experience"] < 0).any():
        return False, "Experience cannot be negative."

    if (data["Selected"] % 1 != 0).any() or not data["Selected"].isin([0, 1]).all():
        return False, "Selected must contain only 0 or 1."

    if (data["Experience"] % 1 != 0).any():
        return False, "Experience must contain whole years."

    if (data["Marks"] % 1 != 0).any() or (data["Skills"] % 1 != 0).any():
        return False, "Marks and Skills must be whole numbers."

    data["Marks"] = data["Marks"].astype(int)
    data["Skills"] = data["Skills"].astype(int)
    data["Experience"] = data["Experience"].astype(int)
    data["Selected"] = data["Selected"].astype(int)

    return True, data


# ==========================================
# 2. FAIRNESS ANALYSIS
# ==========================================

def analyze_fairness(data, attribute="Gender"):

    if data.empty:
        return pd.DataFrame(), None, "Empty Dataset"

    if attribute not in data.columns:
        return pd.DataFrame(), None, "Attribute Not Found"

    result = data.groupby(attribute).agg(
        Total_Candidates=("Selected", "count"),
        Selected_Candidates=("Selected", "sum")
    ).reset_index()

    result["Rejected_Candidates"] = (
        result["Total_Candidates"] -
        result["Selected_Candidates"]
    )

    result["Selection_Rate"] = (
        result["Selected_Candidates"] /
        result["Total_Candidates"] * 100
    ).round(2)

    if len(result) < 2:
        return result, None, "Insufficient Groups"

    rates = result["Selection_Rate"]
    highest = rates.max()
    lowest = rates.min()

    if highest == 0:
        return result, None, "No Selections"

    ratio = lowest / highest

    if ratio >= 0.80:
        status = "Fairness Threshold Met"
    else:
        status = "Potential Bias Detected"

    return result, ratio, status


# ==========================================
# 3. CANDIDATE SCORE CALCULATION
# ==========================================

def calculate_candidate_score(candidate):

    marks_score = candidate["Marks"] / 100 * 50

    skills_score = candidate["Skills"] / 10 * 30

    experience_score = min(
        candidate["Experience"] / 5, 1
    ) * 20

    total_score = (
        marks_score +
        skills_score +
        experience_score
    )

    return round(total_score, 2)


# ==========================================
# 4. EXPLAINABLE AI
# ==========================================

def explain_candidate(candidate):

    marks_score = candidate["Marks"] / 100 * 50

    skills_score = candidate["Skills"] / 10 * 30

    experience_score = min(
        candidate["Experience"] / 5, 1
    ) * 20

    total_score = (
        marks_score +
        skills_score +
        experience_score
    )

    contributions = pd.DataFrame({
        "Factor": ["Marks", "Skills", "Experience"],
        "Contribution": [
            round(marks_score, 2),
            round(skills_score, 2),
            round(experience_score, 2)
        ]
    })

    return round(total_score, 2), contributions


# ==========================================
# 5. CANDIDATE COMPARISON
# ==========================================

def compare_candidates(candidate_a, candidate_b):

    score_a = calculate_candidate_score(candidate_a)
    score_b = calculate_candidate_score(candidate_b)

    return {
        "Candidate_A_Score": score_a,
        "Candidate_B_Score": score_b,
        "Score_Difference": round(abs(score_a - score_b), 2),
        "Same_Selection_Decision": (
            candidate_a["Selected"] == candidate_b["Selected"]
        )
    }


# ==========================================
# 6. FAIRNESS REPORT DATA
# ==========================================

def generate_fairness_report(data, attribute="Gender"):

    result, ratio, status = analyze_fairness(data, attribute)

    if result.empty:
        return result

    report = result.copy()

    report["Fairness_Status"] = status

    report["Disparate_Impact_Ratio"] = (
        round(ratio, 3) if ratio is not None else None
    )

    return report
