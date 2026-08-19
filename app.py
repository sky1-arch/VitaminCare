
import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore
import requests
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="VitaminCare", page_icon="🧬", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
html,body,[class*="css"]{font-family:Inter,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}

/* =========================================================
   VITAMINCARE GLOBAL APP THEME
   Keep the same light-blue/white/red design throughout
   every screen after the Home page.
   ========================================================= */
.stApp{
    background:#f4f7fb !important;
    color:#13233f !important;
}
.block-container{
    max-width:1120px !important;
    padding:0 1.2rem 5rem !important;
    color:#13233f !important;
}

/* Shared header */
.top{
    background:linear-gradient(135deg,#dff7ed 0%,#e8efff 100%) !important;
    padding:30px 24px;
    border-radius:0 0 30px 30px;
    margin-bottom:22px;
}
.logo{font-size:30px;font-weight:800;color:#13233f !important}
.muted{color:#5f718b !important}

/* White cards used across Home, Results, Reports and Profile */
.card,
[data-testid="stVerticalBlockBorderWrapper"]{
    background:#ffffff !important;
    border:1px solid #e5eaf1 !important;
    border-radius:20px !important;
    box-shadow:0 7px 24px rgba(30,50,80,.06) !important;
}
.card{
    padding:22px !important;
    margin:10px 0;
}
.card h1,.card h2,.card h3,.card p,
[data-testid="stVerticalBlockBorderWrapper"] h1,
[data-testid="stVerticalBlockBorderWrapper"] h2,
[data-testid="stVerticalBlockBorderWrapper"] h3,
[data-testid="stVerticalBlockBorderWrapper"] p{
    color:#13233f !important;
}

.kpi{
    font-size:30px;
    font-weight:800;
    color:#13233f !important;
}
.pill{
    display:inline-block;
    padding:6px 12px;
    border-radius:99px;
    background:#eef4ff;
    color:#24456f !important;
    font-weight:700;
}
.result{
    background:linear-gradient(135deg,#ffffff 0%,#f2f8ff 100%) !important;
    border:1px solid #dfe8f5 !important;
    border-radius:24px;
    padding:26px;
}

/* All normal text/headings */
.stApp p,
.stApp label,
.stApp h1,.stApp h2,.stApp h3,
.stApp h4,.stApp h5,.stApp h6,
.stApp [data-testid="stMarkdownContainer"]{
    color:#13233f !important;
}
.stApp .stCaption,
.stApp small{
    color:#5f718b !important;
}

/* Primary red/coral actions — same as Home Start Assessment */
.stApp div[data-testid="stButton"] > button[kind="primary"]{
    background:#ff4d4f !important;
    color:#13233f !important;
    border:1px solid #ff4d4f !important;
    border-radius:10px !important;
    min-height:50px !important;
    font-weight:600 !important;
    box-shadow:none !important;
}
.stApp div[data-testid="stButton"] > button[kind="primary"]:hover{
    background:#f13f42 !important;
    border-color:#f13f42 !important;
}

/* Secondary buttons — white like the Home quick-access buttons */
.stApp div[data-testid="stButton"] > button:not([kind="primary"]){
    background:#ffffff !important;
    color:#13233f !important;
    border:1px solid #dfe5ed !important;
    border-radius:10px !important;
    min-height:50px !important;
    font-weight:600 !important;
    box-shadow:none !important;
}
.stApp div[data-testid="stButton"] > button:not([kind="primary"]):hover{
    background:#f8fafc !important;
    border-color:#cbd5e1 !important;
}

/* Tabs */
.stApp button[data-baseweb="tab"]{
    color:#50627a !important;
}
.stApp button[data-baseweb="tab"][aria-selected="true"]{
    color:#13233f !important;
}

/* Inputs/selects */
.stApp input,
.stApp textarea,
.stApp [data-baseweb="select"] > div{
    color:#13233f !important;
}
.stApp input::placeholder,
.stApp textarea::placeholder{
    color:#7b8799 !important;
}

/* Tables */
.stApp [data-testid="stDataFrame"]{
    border:1px solid #e1e7ef !important;
    border-radius:14px !important;
    overflow:hidden !important;
}

/* Expanders */
.stApp [data-testid="stExpander"]{
    background:#ffffff !important;
    border:1px solid #e1e7ef !important;
    border-radius:14px !important;
}
.stApp [data-testid="stExpander"] summary{
    color:#13233f !important;
}

/* Alerts/messages */
.stApp [data-testid="stAlert"]{
    border-radius:14px !important;
}

/* Bottom navigation */
.stApp hr{
    border-color:#dce3ec !important;
    margin-top:34px !important;
}
.stApp .stButton button{
    transition:all .15s ease;
}
.stApp .stButton button:hover{
    transform:translateY(-1px);
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load():
    return pd.read_csv("vitamin_deficiency_disease_dataset_20260123.csv")

@st.cache_resource
def train(d):
    X=d.drop(columns=["disease_diagnosis"]); y=d["disease_diagnosis"]
    nums=X.select_dtypes(include=["number","bool"]).columns.tolist()
    cats=[c for c in X.columns if c not in nums]
    pre=ColumnTransformer([("num","passthrough",nums),("cat",OneHotEncoder(handle_unknown="ignore"),cats)])
    pipe=Pipeline([("pre",pre),("model",RandomForestClassifier(n_estimators=300,random_state=42,class_weight="balanced"))])
    a,b,c,e=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    pipe.fit(a,c)
    return pipe,accuracy_score(e,pipe.predict(b)),X.columns.tolist(),nums

d=load(); model,acc,features,nums=train(d); cats=[c for c in features if c not in nums]
if "logged" not in st.session_state: st.session_state.logged=False
if "uid" not in st.session_state: st.session_state.uid=None
if "email" not in st.session_state: st.session_state.email=""
if "screen" not in st.session_state: st.session_state.screen="Home"
if "form" not in st.session_state: st.session_state.form={}
if "history" not in st.session_state: st.session_state.history=[]

def go(x): st.session_state.screen=x
def inp(c):
    label=c.replace("_"," ").replace("percent","%").replace("ng ml","ng/mL").replace("pg ml","pg/mL").title()
    if c in cats:
        o=d[c].dropna().astype(str).unique().tolist()
        v=st.session_state.form.get(c,o[0]); st.session_state.form[c]=st.selectbox(label,o,index=o.index(v) if v in o else 0)
    else:
        v=pd.to_numeric(d[c],errors="coerce").dropna(); med=float(v.median())
        lo=float(v.min()); hi=float(v.max())
        st.session_state.form[c]=st.number_input(label,min_value=lo,max_value=hi,value=float(st.session_state.form.get(c,med)))

# Login / signup
if not st.session_state.logged:
    st.markdown('<div style="height:8vh"></div><div class="top" style="text-align:center"><div style="font-size:70px">🧬</div><div class="logo">VitaminCare</div><p class="muted">Secure AI nutrition screening</p></div>',unsafe_allow_html=True)
    mode=st.radio("Account",["Log in","Create account"],horizontal=True)
    email=st.text_input("Email")
    password=st.text_input("Password",type="password")
    name=st.text_input("Full name") if mode=="Create account" else ""
    if st.button("Continue",type="primary",use_container_width=True):
        endpoint="accounts:signInWithPassword" if mode=="Log in" else "accounts:signUp"
        data,err=auth_request(endpoint,email,password)
        if err:
            st.error(err)
        else:
            st.session_state.logged=True
            st.session_state.uid=data["localId"]
            st.session_state.email=email
            st.session_state.name=name or email.split("@")[0]
            save_profile(st.session_state.uid,st.session_state.name,email)
            go("Home"); st.rerun()
    if db is None:
        st.warning("Firebase is not configured yet. Add the secrets described in SETUP.md.")
    st.caption("Accounts and assessment history are stored in Firebase when configured.")
    st.stop()

# Header
st.markdown(f'<div class="top"><div class="logo">🧬 VitaminCare</div><p class="muted">Good to see you, {st.session_state.name}!</p></div>',unsafe_allow_html=True)

screen=st.session_state.screen

if screen=="Home":
    st.subheader("Health overview")
    a,b,c=st.columns(3)
    with a: st.markdown('<div class="card"><div class="muted">Model</div><div class="kpi">Random Forest</div></div>',unsafe_allow_html=True)
    with b: st.markdown(f'<div class="card"><div class="muted">Test accuracy</div><div class="kpi">{acc:.1%}</div></div>',unsafe_allow_html=True)
    with c: st.markdown(f'<div class="card"><div class="muted">Dataset</div><div class="kpi">{len(d):,}</div></div>',unsafe_allow_html=True)
    st.markdown('<div class="card"><h2>Start a screening</h2><p class="muted">Assess nutrition, symptoms, lifestyle and optional laboratory values.</p></div>',unsafe_allow_html=True)
    if st.button("🔍 Start Assessment",type="primary",use_container_width=True): go("Assessment"); st.rerun()
    st.subheader("Quick access")
    a,b,c=st.columns(3)
    with a:
        if st.button("📊 Reports",use_container_width=True): go("Reports"); st.rerun()
    with b:
        if st.button("📚 Learn",use_container_width=True): go("Learn"); st.rerun()
    with c:
        if st.button("👤 Profile",use_container_width=True): go("Profile"); st.rerun()

elif screen=="Assessment":
    st.markdown('<div class="card"><h1 style="margin:0 0 6px;color:#13233f">Health Assessment</h1><p class="muted">Complete the screening using the information available to you.</p></div>',unsafe_allow_html=True)
    st.caption("Enter information available to you. Lab values should come from an actual report.")
    tabs=st.tabs(["👤 Profile","🥗 Nutrition","🩺 Symptoms & Labs"])
    with tabs[0]:
        for c in ["age","gender","bmi","smoking_status","alcohol_consumption","exercise_level","diet_type","sun_exposure","income_level","latitude_region"]:
            if c in features: inp(c)
    with tabs[1]:
        for c in ["vitamin_a_percent_rda","vitamin_c_percent_rda","vitamin_d_percent_rda","vitamin_e_percent_rda","vitamin_b12_percent_rda","folate_percent_rda","calcium_percent_rda","iron_percent_rda"]:
            if c in features: inp(c)
    with tabs[2]:
        for c in ["has_night_blindness","has_fatigue","has_bleeding_gums","has_bone_pain","has_muscle_weakness","has_numbness_tingling","has_memory_problems","has_pale_skin"]:
            if c in features:
                st.session_state.form[c]=st.checkbox(c.replace("has_","").replace("_"," ").title(),value=bool(st.session_state.form.get(c,False)))
        for c in ["hemoglobin_g_dl","serum_vitamin_d_ng_ml","serum_vitamin_b12_pg_ml","serum_folate_ng_ml"]:
            if c in features: inp(c)
    if st.button("✨ Analyze My Health",type="primary",use_container_width=True):
        row={}
        for c in features:
            row[c]=st.session_state.form.get(c, float(pd.to_numeric(d[c],errors="coerce").median()) if c in nums else d[c].dropna().astype(str).mode().iloc[0])
        x=pd.DataFrame([row]); p=model.predict_proba(x)[0]
        res=pd.DataFrame({"Condition":model.classes_,"Score":p}).sort_values("Score",ascending=False)
        st.session_state.result=res; st.session_state.last_input=x; st.session_state.history.append(res.copy())
        try:
            save_assessment(st.session_state.uid, st.session_state.form, res)
            st.session_state.save_message="Assessment saved to Firebase."
        except Exception:
            st.session_state.save_message="Prediction completed, but Firebase save failed."
        go("Results"); st.rerun()

elif screen=="Results":
    st.markdown('<div class="card"><h1 style="margin:0 0 6px;color:#13233f">Your screening result</h1><p class="muted">Review your AI screening result and contributing factors.</p></div>',unsafe_allow_html=True)
    if "result" not in st.session_state:
        st.info("Complete an assessment first.")
    else:
        r=st.session_state.result; top=r.iloc[0]
        st.markdown(f'<div class="result"><span class="pill">AI SCREENING</span><h1 style="margin-bottom:4px">{top["Condition"]}</h1><div class="kpi">{top["Score"]:.1%}</div><p class="muted">Highest model score</p></div>',unsafe_allow_html=True)
        st.subheader("Risk profile")
        st.bar_chart(r.set_index("Condition")["Score"])
        st.dataframe(r.assign(Score=r.Score.map(lambda x:f"{x:.1%}")),use_container_width=True,hide_index=True)
        st.subheader("What may be contributing")
        x=st.session_state.last_input; reasons=[]
        for c in ["vitamin_a_percent_rda","vitamin_c_percent_rda","vitamin_d_percent_rda","vitamin_b12_percent_rda","folate_percent_rda","iron_percent_rda","calcium_percent_rda"]:
            if c in x and float(x[c].iloc[0])<70: reasons.append(c.replace("_"," ").replace("percent","%").title()+" below 70% RDA")
        for c in ["has_fatigue","has_bone_pain","has_muscle_weakness","has_night_blindness","has_bleeding_gums","has_pale_skin"]:
            if c in x and bool(x[c].iloc[0]): reasons.append(c.replace("has_","").replace("_"," ").title()+" reported")
        for q in reasons[:7]: st.write("•",q)
        if not reasons: st.write("No major low-intake or selected symptom flags were identified.")
        st.warning("Model scores are not medical diagnoses or clinical probabilities. Consult a qualified healthcare professional for diagnosis.")
        if st.button("📥 Download Result CSV"):
            st.download_button("Download",r.to_csv(index=False),"vitamincare_result.csv","text/csv")
        if st.button("🔄 New Assessment",use_container_width=True): go("Assessment"); st.rerun()

elif screen=="Reports":
    st.markdown('<div class="card"><h1 style="margin:0 0 6px;color:#13233f">📊 Reports & Progress</h1><p class="muted">Track your previous screening results and progress.</p></div>',unsafe_allow_html=True)
    if not st.session_state.history: st.info("No reports yet. Complete an assessment to create one.")
    else:
        latest=st.session_state.history[-1]
        st.subheader("Latest result")
        st.dataframe(latest.assign(Score=latest.Score.map(lambda x:f"{x:.1%}")),use_container_width=True,hide_index=True)
        # Use a simple trend from saved top scores
        trend=[]
        for i,h in enumerate(st.session_state.history,1):
            t=h.iloc[0]
            trend.append({"Assessment":i,"Top score":float(t.Score)*100})
        st.subheader("Assessment trend")
        st.line_chart(pd.DataFrame(trend).set_index("Assessment"))
        st.caption("Trend shows the highest model score recorded at each assessment in this session.")

elif screen=="Profile":
    st.markdown('<div class="card"><h1 style="margin:0 0 6px;color:#13233f">👤 My Profile</h1><p class="muted">Your VitaminCare account and app information.</p></div>',unsafe_allow_html=True)
    st.markdown(f'<div class="card"><h2>{st.session_state.name}</h2><p class="muted">VitaminCare member</p></div>',unsafe_allow_html=True)
    st.subheader("App information")
    st.write("Dataset:",f"{len(d):,} records / {d.shape[1]} columns")
    st.write("Model: Random Forest")
    st.write("Test accuracy:",f"{acc:.1%}")
    st.info("This project is an academic AI screening prototype. It does not measure vitamins and does not replace medical diagnosis.")

elif screen=="Learn":
    st.markdown('<div class="card"><h1 style="margin:0 0 6px;color:#13233f">📚 Learn</h1><p class="muted">Learn about vitamins, symptoms and nutrition.</p></div>',unsafe_allow_html=True)
    info={
        "Vitamin A":"Supports vision and immune function. Severe deficiency can be associated with night blindness.",
        "Vitamin C":"Supports collagen formation and tissue health. Severe deficiency can cause scurvy.",
        "Vitamin D":"Supports calcium absorption and bone health. Low vitamin D is associated with rickets/osteomalacia.",
        "Vitamin B12":"Supports blood-cell formation and nervous-system function. Deficiency can contribute to anemia.",
        "Folate (B9)":"Supports DNA synthesis and red-blood-cell formation.",
        "Iron":"Essential for hemoglobin production and oxygen transport."
    }
    for k,v in info.items():
        with st.expander(k): st.write(v)

# Bottom navigation
st.markdown("<br><hr>",unsafe_allow_html=True)
cols=st.columns(5)
nav=[("🏠","Home"),("🔍","Assessment"),("📊","Reports"),("📚","Learn"),("👤","Profile")]
for col,(icon,label) in zip(cols,nav):
    with col:
        if st.button(f"{icon}\n{label}",use_container_width=True):
            go(label); st.rerun()
