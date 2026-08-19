import hashlib
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(
    page_title="VitaminCare",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
html,body,[class*="css"]{font-family:Inter,system-ui,sans-serif}
.stApp{background:#f4f7fb}
.block-container{max-width:1120px;padding:0 1.2rem 6rem}
.top{background:linear-gradient(135deg,#dff7ed,#e8efff);padding:30px 24px;border-radius:0 0 30px 30px;margin-bottom:18px}
.logo{font-size:30px;font-weight:800}.muted{color:#667085}
.card{background:#fff;border:1px solid #e7ebf2;border-radius:20px;padding:20px;margin:10px 0;box-shadow:0 6px 20px rgba(30,50,80,.05)}
.kpi{font-size:28px;font-weight:800}.pill{display:inline-block;padding:6px 12px;border-radius:99px;background:#eef4ff;font-weight:700}
.result{background:linear-gradient(135deg,#ffffff,#f2f8ff);border:1px solid #dfe8f5;border-radius:24px;padding:26px}
</style>
""", unsafe_allow_html=True)


# -----------------------------
# Dataset and ML model
# -----------------------------
DATASET_NAME = "vitamin_deficiency_disease_dataset_20260123.csv"


@st.cache_data
def load():
    path = Path(DATASET_NAME)
    if not path.exists():
        # Helpful fallback if the uploaded CSV has a slightly different name.
        csv_files = list(Path(".").glob("*.csv"))
        if len(csv_files) == 1:
            path = csv_files[0]
        else:
            raise FileNotFoundError(
                f"Could not find {DATASET_NAME}. Upload the CSV dataset to the "
                "same GitHub repository as app.py."
            )
    data = pd.read_csv(path)
    if "disease_diagnosis" not in data.columns:
        raise ValueError(
            "The dataset must contain a 'disease_diagnosis' column."
        )
    return data


@st.cache_resource
def train_model(d):
    X = d.drop(columns=["disease_diagnosis"]).copy()
    y = d["disease_diagnosis"].copy()

    nums = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    cats = [c for c in X.columns if c not in nums]

    pre = ColumnTransformer(
        [
            ("num", "passthrough", nums),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                cats,
            ),
        ]
    )

    pipe = Pipeline(
        [
            ("pre", pre),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    pipe.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, pipe.predict(X_test))

    return pipe, accuracy, X.columns.tolist(), nums


try:
    d = load()
    model, acc, features, nums = train_model(d)
    cats = [c for c in features if c not in nums]
except Exception as exc:
    st.error("VitaminCare could not start.")
    st.exception(exc)
    st.stop()


# -----------------------------
# Session state
# -----------------------------
defaults = {
    "logged": False,
    "uid": None,
    "email": "",
    "name": "",
    "screen": "Home",
    "form": {},
    "history": [],
    "accounts": {},
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


def go(screen):
    st.session_state.screen = screen


def hash_password(password):
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def local_auth(mode, email, password, name=""):
    """
    Simple session-based authentication for the Streamlit prototype.
    This replaces the undefined Firebase functions from the original app.
    Accounts exist for the current browser session only.
    """
    email = email.strip().lower()

    if not email or "@" not in email:
        return None, "Please enter a valid email address."

    if len(password) < 6:
        return None, "Password must contain at least 6 characters."

    if mode == "Create account":
        if email in st.session_state.accounts:
            return None, "An account with this email already exists. Please log in."

        uid = f"user_{len(st.session_state.accounts) + 1}"
        st.session_state.accounts[email] = {
            "uid": uid,
            "name": name.strip() or email.split("@")[0],
            "password": hash_password(password),
        }
        return st.session_state.accounts[email], None

    account = st.session_state.accounts.get(email)
    if account is None:
        return None, "No account found in this session. Choose 'Create account' first."

    if account["password"] != hash_password(password):
        return None, "Incorrect password."

    return account, None


def save_profile(uid, name, email):
    """Keep profile information in the current Streamlit session."""
    st.session_state.profile = {
        "uid": uid,
        "name": name,
        "email": email,
    }


def save_assessment(uid, form, result):
    """Keep assessment history in the current Streamlit session."""
    # The prediction is already stored in st.session_state.history.
    st.session_state.last_saved_uid = uid
    st.session_state.last_saved_assessment = {
        "input": dict(form),
        "result": result.copy(),
    }


def inp(c):
    label = (
        c.replace("_", " ")
        .replace("percent", "%")
        .replace("ng ml", "ng/mL")
        .replace("pg ml", "pg/mL")
        .title()
    )

    if c in cats:
        options = d[c].dropna().astype(str).unique().tolist()
        if not options:
            return
        current = st.session_state.form.get(c, options[0])
        index = options.index(current) if current in options else 0
        st.session_state.form[c] = st.selectbox(
            label, options, index=index
        )
    else:
        values = pd.to_numeric(d[c], errors="coerce").dropna()

        if values.empty:
            st.session_state.form[c] = 0.0
            st.number_input(label, value=0.0)
            return

        median = float(values.median())
        low = float(values.min())
        high = float(values.max())

        current = st.session_state.form.get(c, median)
        current = float(current)

        # number_input requires value to be within min/max.
        current = max(low, min(high, current))

        st.session_state.form[c] = st.number_input(
            label,
            min_value=low,
            max_value=high,
            value=current,
        )


# -----------------------------
# Login / signup
# -----------------------------
if not st.session_state.logged:
    st.markdown(
        '<div style="height:8vh"></div>'
        '<div class="top" style="text-align:center">'
        '<div style="font-size:70px">🧬</div>'
        '<div class="logo">VitaminCare</div>'
        '<p class="muted">AI nutrition screening prototype</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    mode = st.radio(
        "Account",
        ["Log in", "Create account"],
        horizontal=True,
    )
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    name = st.text_input("Full name") if mode == "Create account" else ""

    if st.button(
        "Continue",
        type="primary",
        use_container_width=True,
    ):
        account, error = local_auth(mode, email, password, name)

        if error:
            st.error(error)
        else:
            st.session_state.logged = True
            st.session_state.uid = account["uid"]
            st.session_state.email = email.strip().lower()
            st.session_state.name = account["name"]
            save_profile(
                st.session_state.uid,
                st.session_state.name,
                st.session_state.email,
            )
            go("Home")
            st.rerun()

    st.info(
        "This deployed prototype uses session-based accounts. "
        "No Firebase configuration is required."
    )
    st.stop()


# -----------------------------
# Header
# -----------------------------
st.markdown(
    f'<div class="top">'
    f'<div class="logo">🧬 VitaminCare</div>'
    f'<p class="muted">Good to see you, {st.session_state.name}!</p>'
    f'</div>',
    unsafe_allow_html=True,
)

screen = st.session_state.screen


# -----------------------------
# Home
# -----------------------------
if screen == "Home":
    st.subheader("Health overview")

    a, b, c = st.columns(3)

    with a:
        st.markdown(
            '<div class="card"><div class="muted">Model</div>'
            '<div class="kpi">Random Forest</div></div>',
            unsafe_allow_html=True,
        )

    with b:
        st.markdown(
            f'<div class="card"><div class="muted">Test accuracy</div>'
            f'<div class="kpi">{acc:.1%}</div></div>',
            unsafe_allow_html=True,
        )

    with c:
        st.markdown(
            f'<div class="card"><div class="muted">Dataset</div>'
            f'<div class="kpi">{len(d):,}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="card"><h2>Start a screening</h2>'
        '<p class="muted">Assess nutrition, symptoms, lifestyle and '
        'optional laboratory values.</p></div>',
        unsafe_allow_html=True,
    )

    if st.button(
        "🔍 Start Assessment",
        type="primary",
        use_container_width=True,
    ):
        go("Assessment")
        st.rerun()

    st.subheader("Quick access")

    a, b, c = st.columns(3)

    with a:
        if st.button("📊 Reports", use_container_width=True):
            go("Reports")
            st.rerun()

    with b:
        if st.button("📚 Learn", use_container_width=True):
            go("Learn")
            st.rerun()

    with c:
        if st.button("👤 Profile", use_container_width=True):
            go("Profile")
            st.rerun()


# -----------------------------
# Assessment
# -----------------------------
elif screen == "Assessment":
    st.title("Health Assessment")
    st.caption(
        "Enter information available to you. Lab values should come "
        "from an actual report."
    )

    tabs = st.tabs(
        ["👤 Profile", "🥗 Nutrition", "🩺 Symptoms & Labs"]
    )

    with tabs[0]:
        for c in [
            "age",
            "gender",
            "bmi",
            "smoking_status",
            "alcohol_consumption",
            "exercise_level",
            "diet_type",
            "sun_exposure",
            "income_level",
            "latitude_region",
        ]:
            if c in features:
                inp(c)

    with tabs[1]:
        for c in [
            "vitamin_a_percent_rda",
            "vitamin_c_percent_rda",
            "vitamin_d_percent_rda",
            "vitamin_e_percent_rda",
            "vitamin_b12_percent_rda",
            "folate_percent_rda",
            "calcium_percent_rda",
            "iron_percent_rda",
        ]:
            if c in features:
                inp(c)

    with tabs[2]:
        for c in [
            "has_night_blindness",
            "has_fatigue",
            "has_bleeding_gums",
            "has_bone_pain",
            "has_muscle_weakness",
            "has_numbness_tingling",
            "has_memory_problems",
            "has_pale_skin",
        ]:
            if c in features:
                st.session_state.form[c] = st.checkbox(
                    c.replace("has_", "")
                    .replace("_", " ")
                    .title(),
                    value=bool(st.session_state.form.get(c, False)),
                )

        for c in [
            "hemoglobin_g_dl",
            "serum_vitamin_d_ng_ml",
            "serum_vitamin_b12_pg_ml",
            "serum_folate_ng_ml",
        ]:
            if c in features:
                inp(c)

    if st.button(
        "✨ Analyze My Health",
        type="primary",
        use_container_width=True,
    ):
        row = {}

        for c in features:
            if c in st.session_state.form:
                row[c] = st.session_state.form[c]
            elif c in nums:
                value = pd.to_numeric(
                    d[c], errors="coerce"
                ).median()
                row[c] = float(value)
            else:
                mode_values = d[c].dropna().astype(str).mode()
                row[c] = (
                    mode_values.iloc[0]
                    if not mode_values.empty
                    else ""
                )

        x = pd.DataFrame([row])
        probabilities = model.predict_proba(x)[0]

        res = pd.DataFrame(
            {
                "Condition": model.classes_,
                "Score": probabilities,
            }
        ).sort_values("Score", ascending=False)

        st.session_state.result = res
        st.session_state.last_input = x
        st.session_state.history.append(res.copy())

        save_assessment(
            st.session_state.uid,
            st.session_state.form,
            res,
        )

        st.session_state.save_message = (
            "Assessment saved for this session."
        )

        go("Results")
        st.rerun()


# -----------------------------
# Results
# -----------------------------
elif screen == "Results":
    st.title("Your screening result")

    if "result" not in st.session_state:
        st.info("Complete an assessment first.")
    else:
        r = st.session_state.result
        top = r.iloc[0]

        st.markdown(
            f'<div class="result">'
            f'<span class="pill">AI SCREENING</span>'
            f'<h1 style="margin-bottom:4px">{top["Condition"]}</h1>'
            f'<div class="kpi">{top["Score"]:.1%}</div>'
            f'<p class="muted">Highest model score</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.subheader("Risk profile")
        st.bar_chart(r.set_index("Condition")["Score"])

        st.dataframe(
            r.assign(
                Score=r.Score.map(lambda x: f"{x:.1%}")
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("What may be contributing")

        x = st.session_state.last_input
        reasons = []

        for c in [
            "vitamin_a_percent_rda",
            "vitamin_c_percent_rda",
            "vitamin_d_percent_rda",
            "vitamin_b12_percent_rda",
            "folate_percent_rda",
            "iron_percent_rda",
            "calcium_percent_rda",
        ]:
            if c in x.columns and float(x[c].iloc[0]) < 70:
                reasons.append(
                    c.replace("_", " ")
                    .replace("percent", "%")
                    .title()
                    + " below 70% RDA"
                )

        for c in [
            "has_fatigue",
            "has_bone_pain",
            "has_muscle_weakness",
            "has_night_blindness",
            "has_bleeding_gums",
            "has_pale_skin",
        ]:
            if c in x.columns and bool(x[c].iloc[0]):
                reasons.append(
                    c.replace("has_", "")
                    .replace("_", " ")
                    .title()
                    + " reported"
                )

        for reason in reasons[:7]:
            st.write("•", reason)

        if not reasons:
            st.write(
                "No major low-intake or selected symptom flags "
                "were identified."
            )

        st.warning(
            "Model scores are not medical diagnoses or clinical "
            "probabilities. Consult a qualified healthcare professional "
            "for diagnosis."
        )

        if st.session_state.get("save_message"):
            st.success(st.session_state.save_message)

        csv_data = r.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Result CSV",
            csv_data,
            "vitamincare_result.csv",
            "text/csv",
        )

        if st.button(
            "🔄 New Assessment",
            use_container_width=True,
        ):
            go("Assessment")
            st.rerun()


# -----------------------------
# Reports
# -----------------------------
elif screen == "Reports":
    st.title("📊 Reports & Progress")

    if not st.session_state.history:
        st.info(
            "No reports yet. Complete an assessment to create one."
        )
    else:
        latest = st.session_state.history[-1]

        st.subheader("Latest result")
        st.dataframe(
            latest.assign(
                Score=latest.Score.map(lambda x: f"{x:.1%}")
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Assessment trend")

        trend = []
        for i, history_item in enumerate(
            st.session_state.history, 1
        ):
            top_result = history_item.iloc[0]
            trend.append(
                {
                    "Assessment": i,
                    "Top score": float(top_result.Score) * 100,
                }
            )

        st.line_chart(
            pd.DataFrame(trend).set_index("Assessment")
        )

        st.caption(
            "Trend shows the highest model score recorded at "
            "each assessment in this session."
        )


# -----------------------------
# Profile
# -----------------------------
elif screen == "Profile":
    st.title("👤 My Profile")

    st.markdown(
        f'<div class="card"><h2>{st.session_state.name}</h2>'
        f'<p class="muted">{st.session_state.email}</p>'
        f'<p class="muted">VitaminCare member</p></div>',
        unsafe_allow_html=True,
    )

    st.subheader("App information")
    st.write("Dataset:", f"{len(d):,} records / {d.shape[1]} columns")
    st.write("Model: Random Forest")
    st.write("Test accuracy:", f"{acc:.1%}")

    st.info(
        "This project is an academic AI screening prototype. "
        "It does not measure vitamins and does not replace medical diagnosis."
    )


# -----------------------------
# Learn
# -----------------------------
elif screen == "Learn":
    st.title("📚 Learn")

    info = {
        "Vitamin A": (
            "Supports vision and immune function. Severe deficiency "
            "can be associated with night blindness."
        ),
        "Vitamin C": (
            "Supports collagen formation and tissue health. Severe "
            "deficiency can cause scurvy."
        ),
        "Vitamin D": (
            "Supports calcium absorption and bone health. Low vitamin D "
            "is associated with rickets/osteomalacia."
        ),
        "Vitamin B12": (
            "Supports blood-cell formation and nervous-system function. "
            "Deficiency can contribute to anemia."
        ),
        "Folate (B9)": (
            "Supports DNA synthesis and red-blood-cell formation."
        ),
        "Iron": (
            "Essential for hemoglobin production and oxygen transport."
        ),
    }

    for key, value in info.items():
        with st.expander(key):
            st.write(value)


# -----------------------------
# Bottom navigation
# -----------------------------
st.markdown("<br><hr>", unsafe_allow_html=True)

cols = st.columns(5)
nav = [
    ("🏠", "Home"),
    ("🔍", "Assessment"),
    ("📊", "Reports"),
    ("📚", "Learn"),
    ("👤", "Profile"),
]

for col, (icon, label) in zip(cols, nav):
    with col:
        if st.button(
            f"{icon}\n{label}",
            use_container_width=True,
        ):
            go(label)
            st.rerun()
