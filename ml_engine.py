
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


FEATURES = ["Marks", "Skills", "Experience"]
TARGET = "Selected"


def train_model(data):

    required = FEATURES + [TARGET, "Gender"]

    missing = [col for col in required if col not in data.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    clean_data = data[required].dropna().copy()

    X = clean_data[FEATURES]
    y = clean_data[TARGET]

    if y.nunique() < 2:
        raise ValueError("The dataset must contain both selected and rejected candidates.")

    if y.value_counts().min() < 2:
        raise ValueError("Each selection class must contain at least two records.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    metrics = {
        "Accuracy": round(accuracy_score(y_test, predictions), 4),
        "Precision": round(
            precision_score(y_test, predictions, zero_division=0), 4
        ),
        "Recall": round(
            recall_score(y_test, predictions, zero_division=0), 4
        ),
        "F1 Score": round(
            f1_score(y_test, predictions, zero_division=0), 4
        ),
        "Confusion Matrix": confusion_matrix(
            y_test, predictions
        ).tolist()
    }

    return model, metrics


def predict_candidate(model, marks, skills, experience):

    candidate = pd.DataFrame([{
        "Marks": marks,
        "Skills": skills,
        "Experience": experience
    }])

    prediction = int(model.predict(candidate)[0])

    probability = model.predict_proba(candidate)[0]

    return {
        "Prediction": prediction,
        "Selection Probability": round(float(probability[1]) * 100, 2)
    }


def analyze_prediction_fairness(data, model):

    clean_data = data.dropna(
        subset=FEATURES + ["Gender"]
    ).copy()

    clean_data["Predicted"] = model.predict(
        clean_data[FEATURES]
    )

    fairness = (
        clean_data.groupby("Gender")["Predicted"]
        .agg(Total="count", Selected="sum")
        .reset_index()
    )

    fairness["Selection_Rate"] = (
        fairness["Selected"] / fairness["Total"] * 100
    ).round(2)

    rates = fairness["Selection_Rate"].tolist()

    if len(rates) < 2:
        ratio = None
        status = "Insufficient Groups"

    elif max(rates) == 0:
        ratio = None
        status = "No Predicted Selections"

    else:
        ratio = round(min(rates) / max(rates), 3)

        if ratio >= 0.80:
            status = "Fairness Threshold Met"
        else:
            status = "Potential Bias Detected"

    return fairness, ratio, status
