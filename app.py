
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import backend
import ml_engine
from datetime import datetime

# =====================================================
# BIASShield AI
# Intelligent AI Bias Detection and Fairness Testing
# =====================================================

st.set_page_config(
    page_title="BiasShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# PROFESSIONAL UI DESIGN
# =====================================================


# =====================================================
# PROFESSIONAL UI DESIGN - WHITE FONT THEME
# =====================================================

st.markdown("""
<style>

/* Main Background */
.stApp {
    background: linear-gradient(135deg, #0b1220, #17243a);
    color: #ffffff;
}

/* Main content text */
.stApp p,
.stApp label,
.stApp span,
.stApp li,
.stApp div {
    color: #ffffff;
}

/* Sidebar Background */
[data-testid="stSidebar"] {
    background-color: #101b2d;
}

/* Sidebar text */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] li {
    color: #ffffff !important;
}

/* Sidebar Navigation */
[data-testid="stSidebar"] .stRadio label {
    color: #ffffff !important;
    font-weight: 600;
}

/* Sidebar Radio Buttons */
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
    color: #ffffff !important;
}

/* Headings */
h1, h2, h3, h4, h5 {
    color: #35d0ff !important;
}

/* Metric Cards */
[data-testid="stMetric"] {
    background: #1c2b40;
    border: 1px solid #344b66;
    padding: 20px;
    border-radius: 13px;
}

[data-testid="stMetricLabel"] {
    color: #ffffff !important;
}

[data-testid="stMetricValue"] {
    color: #35d0ff !important;
}

/* Input Labels */
.stTextInput label,
.stNumberInput label,
.stSelectbox label,
.stSlider label {
    color: #ffffff !important;
}

/* Buttons */
.stButton button {
    background-color: #087ea4;
    color: #ffffff !important;
    border-radius: 9px;
    border: none;
    font-weight: bold;
}

.stButton button:hover {
    background-color: #06a6cf;
    color: #ffffff !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    color: #ffffff;
}

/* Alerts */
[data-testid="stAlert"] p {
    color: #ffffff !important;
}

/* Sidebar Divider */
[data-testid="stSidebar"] hr {
    border-color: #344b66;
}

/* Footer */
footer {
    visibility: hidden;
}
/* Improve alert box visibility */
[data-testid="stAlert"] {
    background-color: #1c2b40 !important;
    border: 1px solid #35d0ff !important;
    border-radius: 12px;
}

[data-testid="stAlert"] p {
    color: #ffffff !important;
}

/* Improve table appearance */
[data-testid="stDataFrame"] {
    background-color: #1c2b40 !important;
}

</style>

""", unsafe_allow_html=True)



# =====================================================
# SAMPLE DATASET
# =====================================================

def sample_dataset():

    return pd.DataFrame({
        "Candidate_ID": [
            "C001", "C002", "C003", "C004",
            "C005", "C006", "C007", "C008",
            "C009", "C010", "C011", "C012"
        ],
        "Gender": [
            "Male", "Female", "Male", "Female",
            "Male", "Female", "Male", "Female",
            "Male", "Female", "Male", "Female"
        ],
        "Marks": [
            85, 85, 90, 90, 78, 78,
            88, 88, 82, 82, 92, 92
        ],
        "Skills": [
            8, 8, 9, 9, 7, 7,
            8, 8, 8, 8, 9, 9
        ],
        "Experience": [
            2, 2, 3, 3, 1, 1,
            2, 2, 1, 1, 3, 3
        ],
        "Selected": [
            1, 0, 1, 0, 1, 0,
            1, 0, 1, 0, 1, 0
        ]
    })


# =====================================================
# DATA VALIDATION
# =====================================================

REQUIRED_COLUMNS = [
    "Candidate_ID", "Gender", "Marks",
    "Skills", "Experience", "Selected"
]


def validate_dataset(data):
    return backend.validate_dataset(data)

# =====================================================
# FAIRNESS ANALYSIS
# =====================================================

def analyze_fairness(data, attribute):
    return backend.analyze_fairness(data, attribute)


# =====================================================
# EXPLAINABLE AI - TRANSPARENT SCORE ANALYSIS
# =====================================================

def explain_candidate(candidate):
    return backend.explain_candidate(candidate)

# =====================================================
# INITIALIZE SESSION DATA
# =====================================================

if "candidate_data" not in st.session_state:
    st.session_state.candidate_data = sample_dataset()


# =====================================================
# SIDEBAR NAVIGATION
# =====================================================

st.sidebar.markdown("# 🛡️ BiasShield AI")
st.sidebar.caption("AI Fairness Testing Platform")
st.sidebar.divider()

page = st.sidebar.radio(
    "MAIN MENU",
    [
        "Dashboard",
        "Dataset Management",
        "Candidate Comparison",
        "Bias Detection",
        "Explainable AI",
        "Fairness Report"
    ]
)

st.sidebar.divider()

st.sidebar.success("System Status: Active")

st.sidebar.caption(
    "Intelligent Bias Detection & Fairness Testing"
)


# =====================================================
# CURRENT DATA
# =====================================================

df = st.session_state.candidate_data.copy()

# =====================================================
# MACHINE LEARNING MODEL INITIALIZATION
# =====================================================

dataset_signature = int(
    pd.util.hash_pandas_object(df, index=True).sum()
)

if st.session_state.get("ml_dataset_signature") != dataset_signature:

    try:
        ml_model, ml_metrics = ml_engine.train_model(df)

        st.session_state.ml_model = ml_model
        st.session_state.ml_metrics = ml_metrics
        st.session_state.ml_dataset_signature = dataset_signature
        st.session_state.ml_error = None

    except Exception as e:
        st.session_state.ml_error = str(e)
        st.session_state.ml_model = None
        st.session_state.ml_metrics = None



# =====================================================
# DASHBOARD
# =====================================================


# =====================================================
# DASHBOARD - PROFESSIONAL VERSION
# =====================================================

if page == "Dashboard":

    st.title("🛡️ BiasShield AI")
    st.subheader("Intelligent AI Bias Detection Dashboard")

    st.write(
        "Monitor recruitment decisions, compare candidate "
        "selection rates and identify potential fairness issues."
    )

    st.divider()

    # -------------------------------
    # KPI CALCULATIONS
    # -------------------------------

    total = len(df)
    selected = int(df["Selected"].sum()) if total else 0
    rejected = total - selected

    selection_rate = (
        selected / total * 100 if total > 0 else 0
    )

    groups, ratio, status = analyze_fairness(df, "Gender")

    # -------------------------------
    # KPI CARDS
    # -------------------------------

    st.subheader("📊 Recruitment Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Candidates",
        total
    )

    c2.metric(
        "Selected Candidates",
        selected
    )

    c3.metric(
        "Rejected Candidates",
        rejected
    )

    c4.metric(
        "Overall Selection Rate",
        f"{selection_rate:.1f}%"
    )
    

    st.divider()

    # -------------------------------
    # FAIRNESS STATUS
    # -------------------------------

    st.subheader("🛡️ Fairness Overview")

    if status == "Fairness Threshold Met":

        st.success(
            "80% selection-rate threshold met. "
            "The analyzed groups show no disparity below this screening threshold."
        )

    elif status == "Potential Bias Detected":

        st.warning(
            "Potential selection-rate disparity detected. "
            "Further investigation is recommended."
        )

    elif status == "No Selections":

        st.warning(
            "No candidates have been selected. "
            "The selection-rate ratio cannot be meaningfully evaluated."
        )

    else:

        st.info(
            "Insufficient groups for fairness comparison."
        )

    f1, f2 = st.columns(2)

    with f1:

        if ratio is not None:
            st.metric(
                "Disparate Impact Ratio",
                f"{ratio:.3f}"
            )
        else:
            st.metric(
                "Disparate Impact Ratio",
                "N/A"
            )

    with f2:

        st.metric(
            "Fairness Screening Threshold",
            "80%"
        )

    st.caption(
        "The 80% rule is a screening guideline, not proof "
        "of discrimination or a guarantee of fairness."
    )

    st.divider()

    # -------------------------------
    # CHARTS SECTION
    # -------------------------------

    st.subheader("📈 Recruitment Analytics")

    left, right = st.columns(2)

    # Selection Distribution Chart

    with left:

        st.markdown("#### Candidate Selection Distribution")

        pie_data = pd.DataFrame({
            "Status": ["Selected", "Rejected"],
            "Count": [selected, rejected]
        })

        fig1 = px.pie(
            pie_data,
            names="Status",
            values="Count",
            hole=0.55,
            color="Status",
            color_discrete_map={
                "Selected": "#00c9a7",
                "Rejected": "#ff6b6b"
            }
        )

        fig1.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            height=350,
            showlegend=True,
            margin=dict(t=20, b=20, l=20, r=20)
        )

        st.plotly_chart(
            fig1,
            use_container_width=True
        )

    # Group Selection Rate Chart

    with right:

        st.markdown("#### Group-wise Selection Rate")

        fig2 = px.bar(
            groups,
            x="Gender",
            y="Selection_Rate",
            color="Gender",
            text="Selection_Rate",
            color_discrete_sequence=[
                "#35d0ff", "#00c9a7", "#ffb347"
            ]
        )

        fig2.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside"
        )

        fig2.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            yaxis_title="Selection Rate (%)",
            xaxis_title="Group",
            yaxis=dict(range=[0, 115]),
            height=350,
            showlegend=False
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

    # -------------------------------
    # GROUP DISTRIBUTION
    # -------------------------------

    st.divider()

    st.subheader("👥 Candidate Group Distribution")

    group_counts = (
        df.groupby("Gender")
        .size()
        .reset_index(name="Candidate_Count")
    )

    fig3 = px.bar(
        group_counts,
        x="Gender",
        y="Candidate_Count",
        color="Gender",
        text="Candidate_Count",
        color_discrete_sequence=[
            "#35d0ff", "#00c9a7", "#ffb347"
        ]
    )

    fig3.update_traces(
        textposition="outside"
    )

    fig3.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        xaxis_title="Group",
        yaxis_title="Number of Candidates",
        height=350,
        showlegend=False
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )

    # -------------------------------
    # GROUP ANALYSIS TABLE
    # -------------------------------

    st.subheader("📋 Group-wise Fairness Statistics")

    st.dataframe(
        groups,
        use_container_width=True,
        hide_index=True
    )

    # -------------------------------
    # RECENT CANDIDATE RECORDS
    # -------------------------------

    st.divider()

    st.subheader("📝 Recent Candidate Records")

    st.dataframe(
        df.head(10),
        use_container_width=True,
        hide_index=True
    )

    # -------------------------------
    # PROJECT INFORMATION
    # -------------------------------

    st.info(
        "BiasShield AI provides statistical fairness screening. "
        "Differences in selection rates require further investigation "
        "and do not independently establish discrimination."
    )

# =====================================================
# DATASET MANAGEMENT
# =====================================================



# =====================================================
# DATASET MANAGEMENT - PROFESSIONAL VERSION
# =====================================================

elif page == "Dataset Management":

    st.title("📂 Dataset Management")

    st.write(
        "Upload, validate, edit, manage and download "
        "candidate recruitment datasets."
    )

    st.divider()

    # -------------------------------
    # CURRENT DATASET OVERVIEW
    # -------------------------------

    current_data = st.session_state.candidate_data.copy()

    total = len(current_data)
    selected = int(current_data["Selected"].sum())
    rejected = total - selected

    st.subheader("📊 Dataset Overview")

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Records", total)
    c2.metric("Selected", selected)
    c3.metric("Rejected", rejected)

    st.divider()

    # -------------------------------
    # CSV UPLOAD
    # -------------------------------

    st.subheader("📤 Upload Candidate Dataset")

    st.write(
        "Upload a CSV file containing the required candidate columns."
    )

    with st.form("csv_upload_form"):

        uploaded_file = st.file_uploader(
            "Choose CSV File",
            type=["csv"]
        )

        upload_button = st.form_submit_button(
            "Upload and Validate Dataset",
            use_container_width=True
        )

    if upload_button:

        if uploaded_file is None:

            st.warning("Please select a CSV file first.")

        else:

            try:

                uploaded_data = pd.read_csv(uploaded_file)

                valid, result = validate_dataset(uploaded_data)

                if valid:

                    st.session_state.candidate_data = result

                    st.success(
                        "Dataset uploaded and validated successfully!"
                    )

                    st.rerun()

                else:

                    st.error(
                        f"Dataset validation failed: {result}"
                    )

            except Exception as e:

                st.error(
                    f"Unable to read CSV file: {e}"
                )

    st.divider()

    # -------------------------------
    # EDIT DATASET
    # -------------------------------

    st.subheader("✏️ Edit Candidate Information")

    st.info(
        "You can edit existing records or add and delete rows. "
        "Click Save Changes to validate and store your edits."
    )

    edited_data = st.data_editor(
        st.session_state.candidate_data,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        key="candidate_data_editor"
    )

    if st.button(
        "💾 Save Changes",
        use_container_width=True
    ):

        valid, result = validate_dataset(edited_data)

        if valid:

            st.session_state.candidate_data = result

            st.success(
                "Candidate dataset saved successfully!"
            )

            st.rerun()

        else:

            st.error(
                f"Cannot save dataset: {result}"
            )

    st.divider()

    # -------------------------------
    # DATASET PREVIEW
    # -------------------------------

    st.subheader("🔍 Dataset Preview")

    st.dataframe(
        st.session_state.candidate_data,
        use_container_width=True,
        hide_index=True
    )

    # -------------------------------
    # DOWNLOAD AND RESET
    # -------------------------------

    st.divider()

    st.subheader("⚙️ Dataset Actions")

    col1, col2 = st.columns(2)

    with col1:

        st.download_button(
            "⬇️ Download Candidate CSV",
            data=st.session_state.candidate_data.to_csv(
                index=False
            ),
            file_name="BiasShield_Candidate_Dataset.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col2:

        if st.button(
            "🔄 Reset Sample Dataset",
            use_container_width=True
        ):

            st.session_state.candidate_data = sample_dataset()

            st.success(
                "Sample dataset restored successfully!"
            )

            st.rerun()

    # -------------------------------
    # DATA FORMAT INFORMATION
    # -------------------------------

    st.divider()

    st.subheader("📌 Required CSV Format")

    st.write(
        "Your uploaded CSV must contain these six columns:"
    )

    required_format = pd.DataFrame({
        "Column": [
            "Candidate_ID",
            "Gender",
            "Marks",
            "Skills",
            "Experience",
            "Selected"
        ],
        "Description": [
            "Unique candidate identification",
            "Candidate group attribute",
            "Academic marks (0–100)",
            "Skills rating (0–10)",
            "Experience in whole years",
            "Selection decision (1 or 0)"
        ]
    })

    st.dataframe(
        required_format,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Validation checks required columns, missing values, "
        "duplicate IDs, numeric values and permitted ranges."
    )



# =====================================================
# BIAS DETECTION
# =====================================================


elif page == "Bias Detection":

    st.title("🛡️ Bias Detection Engine")
    st.write(
        "Analyze recruitment decisions and identify potential "
        "selection-rate differences between demographic groups."
    )
    st.divider()

    if df.empty:
        st.warning("No candidate records available. Upload a dataset first.")

    elif "Gender" not in df.columns:
        st.error("Gender column is missing from the dataset.")

    else:
        st.subheader("1. Select Group Attribute")

        attribute = st.selectbox(
            "Choose attribute for fairness analysis",
            ["Gender"],
            key="bias_attribute"
        )

        # Calculate group-wise statistics
        result = df.groupby(attribute).agg(
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

        st.divider()
        st.subheader("2. Group-wise Selection Analysis")

        st.dataframe(
            result,
            use_container_width=True,
            hide_index=True
        )

        # Calculate Disparate Impact Ratio
        rates = result["Selection_Rate"]

        if len(result) >= 2 and rates.max() > 0:

            ratio = rates.min() / rates.max()

            selection_gap = rates.max() - rates.min()

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Disparate Impact Ratio",
                f"{ratio:.3f}"
            )

            c2.metric(
                "Selection Rate Gap",
                f"{selection_gap:.2f}%"
            )

            c3.metric(
                "Groups Compared",
                len(result)
            )

            st.divider()
            st.subheader("3. Bias Detection Result")

            if ratio < 0.80:
                st.error(
                    "⚠️ Potential Selection Disparity Detected"
                )

                st.write(
                    "The Disparate Impact Ratio is below 0.80. "
                    "Further investigation is recommended."
                )

            else:
                st.success(
                    "✅ 80% Fairness Screening Threshold Met"
                )

                st.write(
                    "The group selection-rate ratio is at least 0.80."
                )

        else:
            ratio = None

            st.warning(
                "Insufficient data to calculate the "
                "Disparate Impact Ratio."
            )

        st.divider()
        st.subheader("4. Selection Rate Visualization")

        fig = px.bar(
            result,
            x=attribute,
            y="Selection_Rate",
            color=attribute,
            text="Selection_Rate",
            title="Selection Rate by Gender",
            labels={
                "Selection_Rate": "Selection Rate (%)",
                attribute: "Demographic Group"
            }
        )

        fig.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside"
        )

        fig.update_layout(
            yaxis_title="Selection Rate (%)",
            xaxis_title="Gender",
            yaxis_range=[
                0,
                max(100, result["Selection_Rate"].max() + 15)
            ],
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.divider()
        st.subheader("5. Automated Analysis")

        if ratio is not None:
            lowest_group = result.loc[
                result["Selection_Rate"].idxmin(),
                attribute
            ]

            highest_group = result.loc[
                result["Selection_Rate"].idxmax(),
                attribute
            ]

            st.write(
                f"• Highest selection-rate group: **{highest_group}**"
            )

            st.write(
                f"• Lowest selection-rate group: **{lowest_group}**"
            )

            st.write(
                f"• Disparate Impact Ratio: **{ratio:.3f}**"
            )

            st.write(
                f"• Selection-rate difference: **{selection_gap:.2f}%**"
            )

        st.info(
            "Note: The 80% rule is a screening guideline, not proof "
            "of discrimination. Results must be interpreted using "
            "sample size, recruitment criteria and relevant context."
        )

        # =====================================================
        # 6. MACHINE LEARNING-BASED FAIRNESS ANALYSIS
        # =====================================================

        st.divider()
        st.subheader("6. Machine Learning-Based Fairness Analysis")

        st.write(
            "Evaluate the selection decisions predicted by the "
            "trained Logistic Regression model across demographic groups."
        )

        if st.session_state.get("ml_model") is None:

            st.warning(
                "Machine learning model is not available. "
                "Please check the dataset and model training."
            )

        else:

            try:

                ml_fairness, ml_ratio, ml_status = (
                    ml_engine.analyze_prediction_fairness(
                        df,
                        st.session_state.ml_model
                    )
                )

                st.markdown("#### Predicted Group-wise Selection Rates")

                st.dataframe(
                    ml_fairness,
                    use_container_width=True,
                    hide_index=True
                )

                mc1, mc2, mc3 = st.columns(3)

                mc1.metric(
                    "ML Disparate Impact Ratio",
                    f"{ml_ratio:.3f}"
                    if ml_ratio is not None else "N/A"
                )

                mc2.metric(
                    "ML Fairness Status",
                    ml_status
                )

                mc3.metric(
                    "Groups Analyzed",
                    len(ml_fairness)
                )

                if ml_status == "Potential Bias Detected":

                    st.warning(
                        "The model's predicted selection rates "
                        "show potential group disparity."
                    )

                elif ml_status == "Fairness Threshold Met":

                    st.success(
                        "The model's predicted selection-rate ratio "
                        "meets the 80% screening threshold."
                    )

                else:

                    st.info(
                        "The model's predicted selection rates "
                        "cannot be fully compared."
                    )

                st.markdown("#### Predicted Selection Rate Visualization")

                ml_fig = px.bar(
                    ml_fairness,
                    x="Gender",
                    y="Selection_Rate",
                    color="Gender",
                    text="Selection_Rate",
                    title="ML-Predicted Selection Rate by Gender"
                )

                ml_fig.update_traces(
                    texttemplate="%{text:.2f}%",
                    textposition="outside"
                )

                ml_fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="white",
                    yaxis_title="Predicted Selection Rate (%)",
                    xaxis_title="Gender"
                )

                st.plotly_chart(
                    ml_fig,
                    use_container_width=True
                )
                

                st.info(
                    "Important: The model learns patterns from historical "
                    "selection outcomes. Meeting the 80% threshold does "
                    "not prove that the model is completely unbiased."
                )

            except Exception as e:

                st.error(
                    f"Unable to perform ML fairness analysis: {e}"
                )



 # =====================================================
# CANDIDATE COMPARISON - PROFESSIONAL VERSION
# =====================================================

elif page == "Candidate Comparison":

    st.title("👥 Candidate Comparison")

    st.write(
        "Compare two candidates based on academic performance, "
        "skills, experience and actual AI selection decisions."
    )

    st.divider()

    ids = df["Candidate_ID"].astype(str).tolist()

    if len(ids) < 2:

        st.warning("At least two candidates are required.")

    else:

        st.subheader("🔎 Select Candidates")

        col1, col2 = st.columns(2)

        with col1:
            a_id = st.selectbox(
                "Select Candidate A",
                ids,
                index=0,
                key="compare_candidate_a"
            )

        with col2:
            b_id = st.selectbox(
                "Select Candidate B",
                ids,
                index=1,
                key="compare_candidate_b"
            )

        if a_id == b_id:

            st.warning(
                "Please select two different candidates."
            )

        else:

            a = df[
                df["Candidate_ID"].astype(str) == a_id
            ].iloc[0]

            b = df[
                df["Candidate_ID"].astype(str) == b_id
            ].iloc[0]

            score_a, contribution_a = explain_candidate(a)
            score_b, contribution_b = explain_candidate(b)
                        # ==========================================
            # MACHINE LEARNING CANDIDATE PREDICTION
            # ==========================================

            st.divider()
            st.subheader("🤖 ML Candidate Predictions")

            if st.session_state.get("ml_model") is not None:

                model = st.session_state.ml_model

                prediction_a = ml_engine.predict_candidate(
                    model,
                    a["Marks"],
                    a["Skills"],
                    a["Experience"]
                )

                prediction_b = ml_engine.predict_candidate(
                    model,
                    b["Marks"],
                    b["Skills"],
                    b["Experience"]
                )

                pa, pb = st.columns(2)

                with pa:

                    st.markdown("### Candidate A")

                    st.metric(
                        "Predicted Selection Probability",
                        f"{prediction_a['Selection Probability']:.2f}%"
                    )

                    if prediction_a["Prediction"] == 1:
                        st.success("ML Prediction: Selected")
                    else:
                        st.warning("ML Prediction: Not Selected")

                with pb:

                    st.markdown("### Candidate B")

                    st.metric(
                        "Predicted Selection Probability",
                        f"{prediction_b['Selection Probability']:.2f}%"
                    )

                    if prediction_b["Prediction"] == 1:
                        st.success("ML Prediction: Selected")
                    else:
                        st.warning("ML Prediction: Not Selected")

            else:

                st.warning("ML model is not available.")

            st.divider()

            # Candidate profiles

            st.subheader("📋 Candidate Profiles")

            ca, cb = st.columns(2)

            with ca:

                st.markdown("### Candidate A")

                st.metric("Candidate ID", a_id)
                st.metric("Gender", str(a["Gender"]))
                st.metric("Marks", a["Marks"])
                st.metric("Skills", a["Skills"])
                st.metric("Experience", f"{a['Experience']} years")

                if a["Selected"] == 1:
                    st.success("AI Decision: Selected")
                else:
                    st.error("AI Decision: Rejected")

            with cb:

                st.markdown("### Candidate B")

                st.metric("Candidate ID", b_id)
                st.metric("Gender", str(b["Gender"]))
                st.metric("Marks", b["Marks"])
                st.metric("Skills", b["Skills"])
                st.metric("Experience", f"{b['Experience']} years")

                if b["Selected"] == 1:
                    st.success("AI Decision: Selected")
                else:
                    st.error("AI Decision: Rejected")

            st.divider()

            # Comparison table

            st.subheader("📊 Detailed Comparison")

            comparison = pd.DataFrame({
                "Criteria": [
                    "Marks",
                    "Skills",
                    "Experience",
                    "Illustrative Weighted Score"
                ],
                "Candidate A": [
                    a["Marks"],
                    a["Skills"],
                    a["Experience"],
                    round(score_a, 2)
                ],
                "Candidate B": [
                    b["Marks"],
                    b["Skills"],
                    b["Experience"],
                    round(score_b, 2)
                ]
            })

            comparison["Difference (A - B)"] = (
                comparison["Candidate A"] -
                comparison["Candidate B"]
            ).round(2)

            st.dataframe(
                comparison,
                use_container_width=True,
                hide_index=True
            )

            st.divider()

            # Weighted score comparison

            st.subheader("📈 Qualification Score Comparison")

            score_data = pd.DataFrame({
                "Candidate": [a_id, b_id],
                "Weighted Score": [
                    round(score_a, 2),
                    round(score_b, 2)
                ]
            })

            fig = px.bar(
                score_data,
                x="Candidate",
                y="Weighted Score",
                color="Candidate",
                text="Weighted Score",
                title="Illustrative Candidate Scores",
                color_discrete_sequence=[
                    "#35d0ff",
                    "#00c9a7"
                ]
            )

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
                yaxis_title="Score out of 100",
                yaxis=dict(range=[0, 110]),
                height=350
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.divider()

            # Decision analysis

            st.subheader("🛡️ Selection Decision Analysis")

            if a["Selected"] != b["Selected"]:

                st.warning(
                    "The two candidates received different AI "
                    "selection decisions. Investigate their "
                    "qualifications and the decision criteria."
                )

            else:

                st.info(
                    "Both candidates received the same AI "
                    "selection decision."
                )

            score_difference = abs(score_a - score_b)

            st.metric(
                "Difference in Illustrative Scores",
                f"{score_difference:.2f} points"
            )

            if score_difference <= 5:

                st.info(
                    "The illustrative scores are relatively close. "
                    "If decisions differ, investigate the decision "
                    "process carefully."
                )

            else:

                st.info(
                    "The illustrative scores differ. Review the "
                    "individual qualification factors before "
                    "interpreting the selection decisions."
                )

            st.caption(
                "The weighted score is a transparent rule-based "
                "illustration, not a trained AI model. A difference "
                "in decisions does not independently prove bias."
            )

    


# =====================================================
# EXPLAINABLE AI
# =====================================================

elif page == "Explainable AI":

    st.title("🧠 Explainable AI (XAI)")

    st.write(
        "Understand how candidate qualifications contribute "
        "to an illustrative selection score."
    )

    ids = df["Candidate_ID"].tolist()

    if not ids:

        st.warning("No candidate records available.")

    else:

        chosen = st.selectbox(
            "Select Candidate",
            ids
        )

        candidate = df[
            df["Candidate_ID"] == chosen
        ].iloc[0]

        score, contributions = explain_candidate(candidate)

        st.subheader("Candidate Analysis")

        c1, c2, c3 = st.columns(3)

        c1.metric("Marks", candidate["Marks"])
        c2.metric("Skills", candidate["Skills"])
        c3.metric("Experience", candidate["Experience"])

        st.divider()

        st.metric(
            "Illustrative Weighted Score",
            f"{score:.2f}/100"
        )

        fig = px.bar(
            contributions,
            x="Factor",
            y="Contribution",
            color="Factor",
            text="Contribution",
            title="Contribution of Each Qualification"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader("Explanation")

        for _, row in contributions.iterrows():

            st.write(
                f"**{row['Factor']}:** "
                f"{row['Contribution']} points contributed."
            )

        # =====================================================
        # MACHINE LEARNING MODEL EXPLANATION
        # =====================================================

        st.divider()

        st.subheader("🤖 Machine Learning Model Explanation")

        if st.session_state.get("ml_model") is not None:

            model = st.session_state.ml_model

            features = ml_engine.FEATURES

            candidate_input = pd.DataFrame([{
                "Marks": candidate["Marks"],
                "Skills": candidate["Skills"],
                "Experience": candidate["Experience"]
            }])[features]

            scaler = model.named_steps["scaler"]
            classifier = model.named_steps["classifier"]

            scaled_values = scaler.transform(candidate_input)[0]

            coefficients = classifier.coef_[0]

            ml_contributions = scaled_values * coefficients

            explanation_data = pd.DataFrame({
                "Feature": features,
                "Standardized Value": scaled_values.round(3),
                "Model Coefficient": coefficients.round(3),
                "Log-Odds Contribution": ml_contributions.round(3),
                "Influence": [
                    "Increases selection score" if value > 0
                    else "Decreases selection score" if value < 0
                    else "No influence"
                    for value in ml_contributions
                ]
            })

            prediction = int(model.predict(candidate_input)[0])

            probability = model.predict_proba(candidate_input)[0][1] * 100

            st.metric(
                "ML Predicted Selection Probability",
                f"{probability:.2f}%"
            )

            if prediction == 1:
                st.success("ML Prediction: Selected")
            else:
                st.warning("ML Prediction: Not Selected")

            st.markdown("#### Feature-wise Model Explanation")

            st.dataframe(
                explanation_data,
                use_container_width=True,
                hide_index=True
            )

            explanation_fig = px.bar(
                explanation_data,
                x="Feature",
                y="Log-Odds Contribution",
                color="Influence",
                title="How Each Feature Influences the ML Prediction"
            )

            explanation_fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="white"
            )

            st.plotly_chart(
                explanation_fig,
                use_container_width=True
            )

            st.info(
                "These values explain each feature's contribution "
                "to the Logistic Regression model's log-odds for "
                "the selected class. They are not percentages of "
                "the final probability. Gender is excluded from "
                "the model's prediction features."
            )

        else:

            st.warning("ML model is not available.")


        st.info(
            "This is a transparent rule-based scoring example, "
            "not a trained AI prediction. Gender is not included "
            "in this score. It does not explain an external "
            "recruitment model's actual decision."
        )


# =====================================================
# FAIRNESS REPORT
# =====================================================

elif page == "Fairness Report":

    st.title("📑 Fairness Analysis Report")

    attribute = st.selectbox(
        "Choose Group Attribute",
        [
            col for col in df.columns
            if col not in [
                "Candidate_ID", "Marks",
                "Skills", "Experience", "Selected"
            ]
        ]
    )

    result, ratio, status = analyze_fairness(
        df, attribute
    )

    selected = int(df["Selected"].sum())

    st.subheader("Report Summary")

    st.write(
        f"**Generated:** {datetime.now().strftime('%d-%m-%Y %H:%M')}"
    )

    st.write(f"**Total Candidates:** {len(df)}")

    st.write(f"**Selected Candidates:** {selected}")

    st.write(f"**Rejected Candidates:** {len(df) - selected}")

    st.write(f"**Fairness Status:** {status}")

    st.write(
        "**Disparate Impact Ratio:**",
        f"{ratio:.3f}" if ratio is not None else "N/A"
    )

    st.divider()

    st.subheader("Group-wise Results")

    st.dataframe(
        result,
        use_container_width=True,
        hide_index=True
    )

    report = result.copy()

    report["Fairness_Status"] = status

    report["Disparate_Impact_Ratio"] = (
        round(ratio, 3) if ratio is not None else None
    )

    st.download_button(
        "📥 Download Fairness Report CSV",
        data=report.to_csv(index=False),
        file_name="BiasShield_Fairness_Report.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.warning(
        "This report is an initial statistical fairness assessment. "
        "It is not a legal finding or a complete audit of an AI model."
    )

    # =====================================================
    # MACHINE LEARNING FAIRNESS REPORT
    # =====================================================

    st.divider()

    st.subheader("🤖 Machine Learning Fairness Report")

    if st.session_state.get("ml_model") is not None:

        ml_fairness, ml_ratio, ml_status = (
            ml_engine.analyze_prediction_fairness(
                df,
                st.session_state.ml_model
            )
        )

        st.write(
            "Fairness assessment based on the trained "
            "Logistic Regression model's predicted selections."
        )

        r1, r2 = st.columns(2)

        r1.metric(
            "ML Disparate Impact Ratio",
            f"{ml_ratio:.3f}" if ml_ratio is not None else "N/A"
        )

        r2.metric(
            "ML Fairness Status",
            ml_status
        )

        st.markdown("#### ML Group-wise Results")

        st.dataframe(
            ml_fairness,
            use_container_width=True,
            hide_index=True
        )

        ml_report = ml_fairness.copy()

        ml_report["ML_Fairness_Status"] = ml_status

        ml_report["ML_Disparate_Impact_Ratio"] = (
            ml_ratio if ml_ratio is not None else None
        )

        st.download_button(
            "📥 Download ML Fairness Report",
            data=ml_report.to_csv(index=False),
            file_name="BiasShield_ML_Fairness_Report.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.info(
            "The 80% threshold is a screening measure, not proof "
            "that the model is completely unbiased."
        )

    else:

        st.warning("ML model is not available.")



# =====================================================
# FOOTER
# =====================================================

st.divider()

st.markdown(
    "<center>🛡️ BiasShield AI | "
    "Intelligent AI Bias Detection and Fairness Testing</center>",
    unsafe_allow_html=True
)
