import streamlit as st
import os, warnings, io, time
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="EnrollIQ",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Session State ──────────────────────────────────────────────────────────────
if 'splash_done'     not in st.session_state: st.session_state.splash_done = False
if 'saved_scenarios' not in st.session_state: st.session_state.saved_scenarios = []
if 'show_welcome'    not in st.session_state: st.session_state.show_welcome = False

# ── SPLASH SCREEN ─────────────────────────────────────────────────────────────
if not st.session_state.splash_done:
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=JetBrains+Mono:wght@300;400&display=swap');
    html, body, [class*="css"], .stApp { background: #0a0a0f !important; }
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }
    [data-testid="stToolbar"] { display: none; }
    .main .block-container { padding: 0 !important; max-width: 100% !important; }
    .splash-wrap {
        position: fixed; inset: 0; background: #0a0a0f;
        display: flex; flex-direction: column;
        align-items: center; justify-content: center; z-index: 9999;
    }
    .splash-logo {
        font-family: 'Playfair Display', serif;
        font-size: 4rem; font-weight: 900; color: #4682B4;
        letter-spacing: -0.03em;
        animation: fadeUp 0.8s ease forwards; opacity: 0;
    }
    .splash-sub {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem; color: #7a7890;
        text-transform: uppercase; letter-spacing: 0.25em; margin-top: 10px;
        animation: fadeUp 0.8s 0.2s ease forwards; opacity: 0;
    }
    .splash-tagline {
        font-family: 'Playfair Display', serif;
        font-size: 1.1rem; color: #f0ede8; margin-top: 28px;
        animation: fadeUp 0.8s 0.4s ease forwards; opacity: 0; font-style: italic;
    }
    .splash-bar-wrap {
        width: 280px; height: 2px; background: #2a2a3a;
        border-radius: 2px; margin-top: 40px;
        animation: fadeUp 0.6s 0.6s ease forwards; opacity: 0; overflow: hidden;
    }
    .splash-bar {
        height: 2px;
        background: linear-gradient(90deg, #e8c547, #4fd1a5);
        border-radius: 2px;
        animation: loadBar 2.5s 0.8s ease forwards; width: 0%;
    }
    .splash-status {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.62rem; color: #4a4a5a;
        text-transform: uppercase; letter-spacing: 0.15em; margin-top: 14px;
        animation: fadeUp 0.6s 0.8s ease forwards; opacity: 0;
    }
    .splash-dots {
        display: flex; gap: 8px; margin-top: 28px;
        animation: fadeUp 0.6s 1s ease forwards; opacity: 0;
    }
    .splash-dot { width: 6px; height: 6px; border-radius: 50%; background: #2a2a3a; }
    .splash-dot:nth-child(1) { animation: dotPulse 1.2s 1.2s infinite; background: #e8c547; }
    .splash-dot:nth-child(2) { animation: dotPulse 1.2s 1.4s infinite; }
    .splash-dot:nth-child(3) { animation: dotPulse 1.2s 1.6s infinite; }
    @keyframes fadeUp { from{opacity:0;transform:translateY(20px)} to{opacity:1;transform:translateY(0)} }
    @keyframes loadBar { 0%{width:0%} 20%{width:15%} 50%{width:60%} 80%{width:85%} 100%{width:100%} }
    @keyframes dotPulse { 0%,100%{background:#2a2a3a;transform:scale(1)} 50%{background:#e8c547;transform:scale(1.4)} }
    </style>
    <div class='splash-wrap'>
      <div class='splash-logo'>EnrollIQ</div>
      <div class='splash-sub'> RJ College of Arts Science & Commerce · Course Demand Intelligence</div>
      <div class='splash-tagline'>Opening your Dashboard...</div>
      <div class='splash-bar-wrap'><div class='splash-bar'></div></div>
      <div class='splash-status'>Initialising models &amp; data pipeline</div>
      <div class='splash-dots'>
        <div class='splash-dot'></div>
        <div class='splash-dot'></div>
        <div class='splash-dot'></div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    time.sleep(3.5)
    st.session_state.splash_done = True
    st.rerun()

# ── MAIN CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;0,900;1,400&family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');
:root {
  --bg:#0a0a0f; --surface:#111118; --card:#16161f; --border:#2a2a3a;
  --accent:#e8c547; --v2:#7c6af7; --v3:#4fd1a5; --danger:#f06a6a;
  --warm:#f0a05a; --text:#f0ede8; --muted:#7a7890; --subtle:#2e2e40;
}
*{box-sizing:border-box;}
html,body,[class*="css"],.stApp{background:var(--bg)!important;color:var(--text)!important;font-family:'Syne',sans-serif!important;}
#MainMenu,footer,header{visibility:hidden;}
.stDeployButton{display:none;}
[data-testid="stToolbar"]{display:none;}
::-webkit-scrollbar{width:4px;height:4px;}
::-webkit-scrollbar-track{background:var(--bg);}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:4px;}
.main .block-container{padding:0!important;max-width:100%!important;}

/* TOP BAR */
.top-bar{background:var(--surface);border-bottom:1px solid var(--border);padding:12px 40px;display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;z-index:999;}
.top-bar-logo{font-family:'Playfair Display',serif;font-size:1.35rem;font-weight:900;color:var(--accent);letter-spacing:-0.02em;}
.top-bar-sub{font-size:0.62rem;color:var(--muted);font-family:'JetBrains Mono',monospace;letter-spacing:0.1em;text-transform:uppercase;margin-top:1px;}
.top-bar-right{font-family:'JetBrains Mono',monospace;font-size:0.68rem;color:var(--muted);display:flex;align-items:center;gap:20px;}
.status-dot{width:8px;height:8px;border-radius:50%;background:#4fd1a5;box-shadow:0 0 6px #4fd1a5;display:inline-block;margin-right:6px;animation:pulse 2s infinite;}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.4}}

/* WELCOME */
.welcome-overlay{position:fixed;inset:0;background:rgba(0,0,0,0.88);z-index:9998;display:flex;align-items:center;justify-content:center;backdrop-filter:blur(6px);}
.welcome-box{background:#16161f;border:1px solid #2a2a3a;border-radius:20px;padding:40px 48px;max-width:560px;width:90%;text-align:center;animation:fadeUp 0.4s ease;}
.welcome-title{font-family:'Playfair Display',serif;font-size:2rem;font-weight:900;color:#e8c547;margin-bottom:8px;}
.welcome-sub{font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:#7a7890;text-transform:uppercase;letter-spacing:0.15em;margin-bottom:20px;}
.welcome-feat{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:16px 0 24px;text-align:left;}
.welcome-feat-item{background:#2e2e40;border-radius:10px;padding:12px 14px;font-size:0.78rem;color:#f0ede8;}
.welcome-feat-item span{font-family:'JetBrains Mono',monospace;font-size:0.6rem;color:#7a7890;display:block;margin-top:3px;}

/* KPI */
.kpi-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:16px;margin-bottom:32px;}
.kpi-card{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:22px 20px 18px;position:relative;overflow:hidden;transition:border-color 0.25s,transform 0.2s;}
.kpi-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:var(--accent);opacity:0.7;}
.kpi-card:hover{border-color:var(--accent);transform:translateY(-2px);}
.kpi-label{font-family:'JetBrains Mono',monospace;font-size:0.6rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.12em;margin-bottom:10px;}
.kpi-value{font-family:'Playfair Display',serif;font-size:2.1rem;font-weight:700;color:var(--accent);line-height:1;margin-bottom:6px;}
.kpi-sub{font-size:0.7rem;color:var(--muted);}
.kpi-trend-up{color:#4fd1a5;font-size:0.7rem;font-family:'JetBrains Mono',monospace;}

/* SECTION */
.sec-label{font-family:'JetBrains Mono',monospace;font-size:0.62rem;color:var(--accent);text-transform:uppercase;letter-spacing:0.18em;margin-bottom:6px;}
.sec-title{font-family:'Playfair Display',serif;font-size:2rem;font-weight:700;color:var(--text);margin:0 0 24px 0;line-height:1.15;}
.param-label{font-family:'JetBrains Mono',monospace;font-size:0.62rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.1em;margin-bottom:2px;}

/* RISK */
.risk-banner{border-radius:10px;padding:12px 20px;font-family:'JetBrains Mono',monospace;font-size:0.76rem;font-weight:500;text-align:center;margin-top:14px;letter-spacing:0.06em;}
.risk-low{background:rgba(79,209,165,0.12);color:#4fd1a5;border:1px solid rgba(79,209,165,0.3);}
.risk-moderate{background:rgba(232,197,71,0.12);color:#e8c547;border:1px solid rgba(232,197,71,0.3);}
.risk-high{background:rgba(240,106,106,0.12);color:#f06a6a;border:1px solid rgba(240,106,106,0.3);}
.risk-critical{background:rgba(240,106,106,0.2);color:#ff4444;border:1px solid rgba(255,68,68,0.4);}

/* RESOURCE */
.resource-row{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:20px;}
.resource-chip{background:var(--subtle);border:1px solid var(--border);border-radius:10px;padding:14px 10px;text-align:center;transition:border-color 0.2s;}
.resource-chip:hover{border-color:var(--accent);}
.resource-chip .r-val{font-family:'Playfair Display',serif;font-size:1.5rem;font-weight:700;color:var(--text);}
.resource-chip .r-lbl{font-family:'JetBrains Mono',monospace;font-size:0.58rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.08em;margin-top:3px;}

/* MODEL PILLS */
.model-compare{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:16px;}
.model-pill{background:var(--subtle);border:1px solid var(--border);border-radius:10px;padding:14px;text-align:center;transition:all 0.2s;}
.model-pill:hover{transform:translateY(-2px);}
.model-pill.best{border-color:var(--accent);background:rgba(232,197,71,0.06);}
.model-pill-name{font-family:'JetBrains Mono',monospace;font-size:0.58rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px;}
.model-pill-val{font-family:'Playfair Display',serif;font-size:1.5rem;font-weight:700;color:var(--text);}
.best-star{font-size:0.58rem;color:var(--accent);font-family:'JetBrains Mono',monospace;margin-top:4px;}

/* SCENARIO */
.scenario-card{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:18px 20px;margin-bottom:12px;position:relative;transition:border-color 0.2s,transform 0.15s;}
.scenario-card:hover{border-color:var(--accent);transform:translateY(-2px);}
.scenario-num{position:absolute;top:14px;right:16px;font-family:'Playfair Display',serif;font-size:2rem;font-weight:900;color:var(--border);}
.scenario-course{font-family:'Playfair Display',serif;font-size:1.1rem;font-weight:700;color:var(--text);}
.scenario-pred{font-family:'Playfair Display',serif;font-size:2.2rem;font-weight:900;color:var(--accent);line-height:1;}
.scenario-meta{font-family:'JetBrains Mono',monospace;font-size:0.6rem;color:var(--muted);margin-top:4px;}

/* LEADERBOARD */
.lb-row{display:flex;align-items:center;gap:16px;background:var(--card);border:1px solid var(--border);border-radius:12px;padding:14px 20px;margin-bottom:10px;transition:border-color 0.2s,transform 0.15s;}
.lb-row:hover{border-color:var(--accent);transform:translateX(4px);}
.lb-rank{font-family:'Playfair Display',serif;font-size:1.6rem;font-weight:900;color:var(--border);min-width:36px;}
.lb-rank.gold{color:#e8c547;} .lb-rank.silver{color:#a8b2c0;} .lb-rank.bronze{color:#cd7f32;}
.lb-emoji{font-size:1.5rem;}
.lb-name{font-family:'Syne',sans-serif;font-size:0.9rem;font-weight:600;color:var(--text);flex:1;}
.lb-score{font-family:'Playfair Display',serif;font-size:1.4rem;font-weight:700;color:var(--accent);}
.lb-bar-wrap{flex:2;background:var(--subtle);border-radius:4px;height:6px;}
.lb-bar{height:6px;border-radius:4px;background:linear-gradient(90deg,var(--accent),var(--v3));}

/* HEALTH */
.health-card{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:20px;text-align:center;transition:all 0.2s;}
.health-card:hover{border-color:var(--accent);transform:translateY(-2px);}
.health-score{font-family:'Playfair Display',serif;font-size:2.4rem;font-weight:900;line-height:1;}
.health-name{font-family:'JetBrains Mono',monospace;font-size:0.6rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.1em;margin-top:6px;}
.health-grade{font-family:'Syne',sans-serif;font-size:0.72rem;font-weight:700;margin-top:4px;}

/* SEM GRID */
.sem-grid{display:grid;grid-template-columns:repeat(8,1fr);gap:8px;margin:8px 0 16px;}
.sem-cell{border-radius:10px;padding:10px 6px;text-align:center;border:1px solid var(--border);transition:transform 0.15s;}
.sem-cell:hover{transform:scale(1.06);}
.sem-cell-num{font-family:'JetBrains Mono',monospace;font-size:0.55rem;color:var(--muted);text-transform:uppercase;}
.sem-cell-val{font-family:'Playfair Display',serif;font-size:1.2rem;font-weight:700;margin-top:4px;}

/* ALERT */
.alert-box{border-radius:14px;padding:16px 22px;margin-bottom:12px;border-left:4px solid;animation:slideIn 0.3s ease;}
@keyframes slideIn{from{opacity:0;transform:translateX(-10px)}to{opacity:1;transform:translateX(0)}}
.alert-critical{background:rgba(255,68,68,0.08);border-color:#ff4444;}
.alert-high{background:rgba(240,106,106,0.08);border-color:#f06a6a;}
.alert-ok{background:rgba(79,209,165,0.08);border-color:#4fd1a5;}
.alert-title{font-family:'Syne',sans-serif;font-size:0.85rem;font-weight:700;}
.alert-body{font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:var(--muted);margin-top:4px;}

/* BUDGET */
.budget-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:16px 0;}
.budget-item{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:18px;position:relative;overflow:hidden;transition:all 0.2s;}
.budget-item:hover{border-color:var(--accent);transform:translateY(-2px);}
.budget-item::after{content:'';position:absolute;bottom:0;left:0;right:0;height:2px;background:linear-gradient(90deg,var(--v2),var(--accent));}
.budget-lbl{font-family:'JetBrains Mono',monospace;font-size:0.6rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.1em;}
.budget-val{font-family:'Playfair Display',serif;font-size:1.6rem;font-weight:700;color:var(--accent);margin-top:6px;}

/* H2H */
.h2h-side{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:24px;text-align:center;transition:all 0.2s;}
.h2h-winner{border-color:var(--accent);background:rgba(232,197,71,0.04);}
.h2h-num{font-family:'Playfair Display',serif;font-size:3.5rem;font-weight:900;line-height:1;}

/* TABS */
[data-testid="stTabs"] [role="tablist"]{background:var(--surface)!important;border-bottom:1px solid var(--border)!important;gap:0!important;padding:0!important;}
[data-testid="stTabs"] button[role="tab"]{font-family:'JetBrains Mono',monospace!important;font-size:0.66rem!important;text-transform:uppercase!important;letter-spacing:0.1em!important;color:var(--muted)!important;padding:14px 18px!important;border-radius:0!important;border-bottom:2px solid transparent!important;}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"]{color:var(--accent)!important;border-bottom:2px solid var(--accent)!important;background:transparent!important;}

/* INPUTS */
[data-testid="stSelectbox"]>div>div{background:var(--card)!important;border:1px solid var(--border)!important;border-radius:8px!important;color:var(--text)!important;}
[data-testid="stSlider"]>div>div>div{background:var(--accent)!important;}
[data-testid="stSlider"]>div>div>div>div{background:var(--accent)!important;box-shadow:0 0 8px rgba(232,197,71,0.5)!important;}
[data-testid="stDataFrame"]{border:1px solid var(--border)!important;border-radius:10px!important;overflow:hidden!important;}
[data-testid="stDownloadButton"] button{background:transparent!important;border:1px solid var(--accent)!important;color:var(--accent)!important;font-family:'JetBrains Mono',monospace!important;font-size:0.7rem!important;border-radius:8px!important;padding:8px 20px!important;}
[data-testid="stDownloadButton"] button:hover{background:rgba(232,197,71,0.1)!important;}
[data-baseweb="tag"]{background:rgba(232,197,71,0.15)!important;color:var(--accent)!important;border-radius:4px!important;}
[data-baseweb="multi-select"]{background:var(--card)!important;border-color:var(--border)!important;}
.stAlert{border-radius:10px!important;}
.fancy-divider{border:none;border-top:1px solid var(--border);margin:28px 0;}
.stButton>button{background:rgba(232,197,71,0.1)!important;border:1px solid var(--accent)!important;color:var(--accent)!important;font-family:'JetBrains Mono',monospace!important;font-size:0.72rem!important;border-radius:8px!important;padding:8px 20px!important;letter-spacing:0.06em!important;transition:all 0.2s!important;}
.stButton>button:hover{background:rgba(232,197,71,0.2)!important;}
@keyframes fadeUp{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:translateY(0)}}
.fade-in{animation:fadeUp 0.5s ease forwards;}
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
DEPARTMENTS = {
    'Technology':     ['AI & ML','Data Science','Cybersecurity','Cloud Computing','Software Engineering','DevOps'],
    'Business':       ['Business Analytics','Digital Marketing','Finance & Accounting','Entrepreneurship','Supply Chain Management'],
    'Arts & Music':   ['Music Production','Sound Engineering','Music Theory','Live Performance'],
    'Design & Media': ['UI/UX Design','Graphic Design','Film & Video Production','Photography','Animation'],
}
COURSES = [c for dept_list in DEPARTMENTS.values() for c in dept_list]
COURSE_DEPT = {c: dept for dept, courses in DEPARTMENTS.items() for c in courses}
DEPT_COLORS = {
    'Technology':     '#58a6ff',
    'Business':       '#4fd1a5',
    'Arts & Music':   '#f0a05a',
    'Design & Media': '#bc8cff',
}
COURSE_EMOJI = {
    'AI & ML':'🤖','Data Science':'📊','Cybersecurity':'🔐','Cloud Computing':'☁️',
    'Software Engineering':'💻','DevOps':'🚀',
    'Business Analytics':'📈','Digital Marketing':'📣','Finance & Accounting':'💰',
    'Entrepreneurship':'🏢','Supply Chain Management':'🔗',
    'Music Production':'🎵','Sound Engineering':'🎚️','Music Theory':'🎼','Live Performance':'🎤',
    'UI/UX Design':'🎨','Graphic Design':'🖌️','Film & Video Production':'🎬',
    'Photography':'📷','Animation':'✨',
}
COURSE_DEMAND_BIAS = {
    'AI & ML':9,'Data Science':9,'Cybersecurity':8,'Cloud Computing':8,
    'Software Engineering':7,'DevOps':7,
    'Business Analytics':7,'Digital Marketing':5,'Finance & Accounting':7,
    'Entrepreneurship':6,'Supply Chain Management':6,
    'Music Production':5,'Sound Engineering':5,'Music Theory':4,'Live Performance':4,
    'UI/UX Design':7,'Graphic Design':6,'Film & Video Production':5,
    'Photography':4,'Animation':6,
}
DEPT_CAPACITY = {
    'Technology':     (20, 180),
    'Business':       (25, 160),
    'Arts & Music':   (10, 60),
    'Design & Media': (15, 80),
}
FEATURES = ['semester','faculty_rating','previous_enrollment','course_difficulty','is_elective','has_lab','industry_demand','year','season_encoded','faculty_demand_interaction','elective_diff_interaction']
SEASON_MAP = {'Fall':0,'Spring':2,'Summer':1}
SALARY_PER_FACULTY = 80000
ROOM_COST_PER_SECTION = 5000
MATERIAL_PER_STUDENT = 120
PL = dict(paper_bgcolor='#16161f',plot_bgcolor='#0a0a0f',font=dict(color='#7a7890',family='JetBrains Mono',size=11),xaxis=dict(gridcolor='#2a2a3a',zerolinecolor='#2a2a3a',tickfont=dict(size=10)),yaxis=dict(gridcolor='#2a2a3a',zerolinecolor='#2a2a3a',tickfont=dict(size=10)),margin=dict(l=40,r=20,t=44,b=40),title_font=dict(family='Syne',size=13,color='#f0ede8'))
C = ['#e8c547','#7c6af7','#4fd1a5','#f06a6a','#f0a05a','#5ac8f5','#bc8cff','#39d353','#ff6b6b','#4ecdc4']

# ── ML ─────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def generate_data():
    np.random.seed(42)
    n=2000; names=np.random.choice(COURSES,n)
    ind=np.array([min(10,max(1,int(np.random.normal(COURSE_DEMAND_BIAS[c],1.5)))) for c in names])
    fr=np.round(np.random.uniform(3.0,5.0,n),2); pe=np.random.randint(15,150,n)
    cd=np.random.randint(1,6,n); ie=np.random.choice([0,1],n); hl=np.random.choice([0,1],n)
    fdi=fr*ind; edi=ie*(6-cd)
    enr=(pe*0.5+fr*12+ind*3.5+ie*8-hl*5-cd*8+fdi*0.8+edi*2.5+np.random.normal(0,8,n)).astype(int).clip(10,200)
    return pd.DataFrame({'course_name':names,'semester':np.random.randint(1,9,n),'faculty_rating':fr,'previous_enrollment':pe,'course_difficulty':cd,'is_elective':ie,'has_lab':hl,'industry_demand':ind,'faculty_demand_interaction':np.round(fdi,2),'elective_diff_interaction':edi,'year':np.random.choice([2022,2023,2024],n),'season':np.random.choice(['Fall','Spring','Summer'],n),'expected_enrollment':enr})

def acc_tol(y_true,y_pred,tol=0.15):
    y_true,y_pred=np.array(y_true),np.array(y_pred)
    return round((np.abs(y_true-y_pred)<=tol*y_true).mean()*100,2)

@st.cache_resource(show_spinner=False)
def train_all(df):
    rows,mdict,edict=[],{},{}
    for subj in COURSES:
        sub=df[df['course_name']==subj].copy().reset_index(drop=True)
        if len(sub)<20: continue
        le=LabelEncoder(); sub['season_encoded']=le.fit_transform(sub['season'])
        X,y=sub[FEATURES],sub['expected_enrollment']
        Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42)
        gs=GridSearchCV(RandomForestRegressor(random_state=42),{'n_estimators':[100,200],'max_depth':[8,12,None]},cv=3,scoring='r2',n_jobs=-1)
        gs.fit(Xtr,ytr)
        mdls={'Linear Regression':LinearRegression(),'Decision Tree':DecisionTreeRegressor(max_depth=8,random_state=42),'Random Forest':gs.best_estimator_}
        res={}
        for nm,m in mdls.items():
            m.fit(Xtr,ytr); p=m.predict(Xte); cv=cross_val_score(m,X,y,cv=5,scoring='r2').mean(); mse=mean_squared_error(yte,p)
            res[nm]={'Accuracy (±15%)':acc_tol(yte,p),'R² Score':round(r2_score(yte,p),4),'CV R²':round(cv,4),'MAE':round(mean_absolute_error(yte,p),2),'RMSE':round(np.sqrt(mse),2),'MAPE (%)':round(mean_absolute_percentage_error(yte,p)*100,2)}
        rdf=pd.DataFrame(res).T; best=rdf['R² Score'].idxmax(); br=rdf.loc[best]
        mdict[subj]={n:m for n,m in mdls.items()}; edict[subj]=le
        rows.append({'Subject':subj,'Records':len(sub),'Avg Enrollment':round(sub['expected_enrollment'].mean(),1),'Best Model':best,'Accuracy (±15%)':f"{br['Accuracy (±15%)']}%",'R² Score':br['R² Score'],'CV R²':br['CV R²'],'MAE':br['MAE'],'RMSE':br['RMSE'],'MAPE (%)':br['MAPE (%)'],'R² ✓':'✅' if br['R² Score']>=0.90 else '❌','MAE ✓':'✅' if br['MAE']<15 else '❌','all_results':res})
    return pd.DataFrame(rows),mdict,edict

def predict(subj,sem,fr,pe,cd,ie,hl,ind,yr,season,mdict,edict):
    fi=fr*ind; edi=int(ie)*(6-cd)
    row=pd.DataFrame([{'semester':sem,'faculty_rating':fr,'previous_enrollment':pe,'course_difficulty':cd,'is_elective':int(ie),'has_lab':int(hl),'industry_demand':ind,'year':yr,'season_encoded':SEASON_MAP.get(season,0),'faculty_demand_interaction':round(fi,2),'elective_diff_interaction':edi}])
    safe = subj.replace(' ','_').replace('&','and').replace('/','_')
    return {nm:int(round(m.predict(row)[0])) for nm,m in mdict[subj].items()}

def risk_info(n):
    if n>120: return ("CRITICAL","risk-critical","🔴")
    if n>80:  return ("HIGH","risk-high","🟠")
    if n>50:  return ("MODERATE","risk-moderate","🟡")
    return        ("LOW","risk-low","🟢")

def health_score(r2,acc,mae,avg_enr):
    score=int(min(100,r2*100)*0.35+min(100,acc)*0.30+max(0,100-mae*3)*0.20+min(100,avg_enr/2)*0.15)
    if score>=85: return score,"#4fd1a5","Excellent"
    if score>=70: return score,"#e0bb33","Good"
    if score>=55: return score,"#f0a05a","Fair"
    return score,"#f06a6a","Needs Work"

def budget_calc(predicted,sections,faculty):
    fc=faculty*SALARY_PER_FACULTY; rc=sections*ROOM_COST_PER_SECTION; mc=predicted*MATERIAL_PER_STUDENT
    return fc,rc,mc,fc+rc+mc

# ── Load ───────────────────────────────────────────────────────────────────────
full_df=generate_data()
with st.spinner(""):
    summary_df,mdict,edict=train_all(full_df)
    # Make sure Department column exists even if loaded from old SQLite cache
    if 'Department' not in summary_df.columns:
        summary_df['Department'] = summary_df['Subject'].apply(lambda x: COURSE_DEPT.get(x,'Unknown'))

# ── WELCOME MODAL ─────────────────────────────────────────────────────────────
if st.session_state.show_welcome:
    st.markdown("""
    <div class='welcome-overlay'>
      <div class='welcome-box'>
        <div style='font-size:2.5rem;margin-bottom:8px;'>🎓</div>
        <div class='welcome-title'>Welcome to EnrollIQ</div>
        <div class='welcome-sub'>Course Demand Intelligence · RJ College of Arts Science & Commerce</div>
        <div style='font-size:0.82rem;color:#7a7890;line-height:1.6;'>
          Predict student enrollment for any course instantly.<br>
          Built by Rishab Singh · 10441
        </div>
        <div class='welcome-feat'>
          <div class='welcome-feat-item'>⚡ Live Prediction<span>Instant results as you move sliders</span></div>
          <div class='welcome-feat-item'>📌 Save &amp; Compare<span>Compare multiple scenarios</span></div>
          <div class='welcome-feat-item'>🏆 Leaderboard<span>See top performing courses</span></div>
          <div class='welcome-feat-item'>🚨 Smart Alerts<span>Auto flag risky courses</span></div>
          <div class='welcome-feat-item'>⚔️ Head to Head<span>Compare any two courses</span></div>
          <div class='welcome-feat-item'>📋 Resource Plan<span>Sections, faculty &amp; budget</span></div>
        </div>
        <div style='margin-top:20px;font-family:JetBrains Mono,monospace;font-size:0.65rem;color:#7a7890;'>
          ↓ Click the button below to open the dashboard
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Button sits below overlay — always visible and clickable
    st.markdown("<br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀  Open Dashboard — EnrollIQ", use_container_width=True):
            st.session_state.show_welcome = False
            st.rerun()
    st.stop()
    # DO NOT use st.stop() — let the rest of the app render behind the overlay

# ── TOP BAR ───────────────────────────────────────────────────────────────────
avg_r2=summary_df['R² Score'].mean(); mae_pass=(summary_df['MAE']<15).sum()
st.markdown(f"""
<div class='top-bar'>
  <div>
    <div class='top-bar-logo'>EnrollIQ</div>
    <div class='top-bar-sub'>Course Demand Intelligence · RJ College of Arts Scinece & Commerce</div>
  </div>
  <div class='top-bar-right'>
    <span><span class='status-dot'></span>Live</span>
    <span>Avg R² {avg_r2:.3f}</span>
    <span>MAE ✓ {mae_pass}/10</span>
    <span>Rishab Singh · 10441</span>
    <span>{datetime.now().strftime('%d %b %Y')}</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8=st.tabs([
    "⚡  Predict","📌  Compare","🏆  Leaderboard",
    "⚔️  Head to Head","🚨  Alerts","🏠  Overview","🤖  Models","📋  Planning"
])

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("<div style='padding:28px 0 0 0;'>",unsafe_allow_html=True)
    st.markdown("<div class='sec-label'>Live Intelligence</div>",unsafe_allow_html=True)
    st.markdown("<div class='sec-title'>Enrollment Predictor</div>",unsafe_allow_html=True)
    r1c1,r1c2=st.columns([3,1])
    with r1c1: subject=st.selectbox("Course",COURSES,format_func=lambda x:f"{COURSE_EMOJI[x]}  {x}")
    with r1c2: season=st.selectbox("Season",['Fall','Spring','Summer'])
    left_col,right_col=st.columns([1.1,1],gap="medium")
    with left_col:
        st.markdown("---")
        c1,c2=st.columns(2)
        with c1:
            st.markdown("<div class='param-label'>Faculty Rating</div>",unsafe_allow_html=True)
            faculty_rating=st.slider("fr",3.0,5.0,4.2,0.1,label_visibility="collapsed")
            st.markdown("<div class='param-label'>Prev. Enrollment</div>",unsafe_allow_html=True)
            prev_enroll=st.slider("pe",10,150,65,label_visibility="collapsed")
            st.markdown("<div class='param-label'>Industry Demand</div>",unsafe_allow_html=True)
            ind_demand=st.slider("id",1,10,COURSE_DEMAND_BIAS.get(subject,7),label_visibility="collapsed")
            st.markdown("<div class='param-label'>Year</div>",unsafe_allow_html=True)
            year=st.selectbox("yr",[2024,2025,2026],index=2,label_visibility="collapsed")
        with c2:
            st.markdown("<div class='param-label'>Semester</div>",unsafe_allow_html=True)
            semester=st.slider("sem",1,8,3,label_visibility="collapsed")
            st.markdown("<div class='param-label'>Course Difficulty</div>",unsafe_allow_html=True)
            difficulty=st.slider("diff",1,5,3,label_visibility="collapsed")
            st.markdown("<br>",unsafe_allow_html=True)
            is_elective=st.checkbox("📚 Elective Course",value=True)
            has_lab=st.checkbox("🔬 Has Lab Component",value=False)
    with right_col:
        preds=predict(subject,semester,faculty_rating,prev_enroll,difficulty,is_elective,has_lab,ind_demand,year,season,mdict,edict)
        best_name=summary_df.loc[summary_df['Subject']==subject,'Best Model'].values[0]
        best_pred=preds[best_name]
        risk_txt,risk_cls,risk_emoji=risk_info(best_pred)
        sections=max(1,round(best_pred/40)); faculty=max(1,round(sections*1.2))
        fc,rc,mc,total=budget_calc(best_pred,sections,faculty)
        # this is the predicted enrollement css
        st.markdown(f"""
        <div class='fade-in' style='background:linear-gradient(10deg,#16161f,#1a1a28);border:1px solid #2a2a3a;border-radius:20px;padding:32px 24px;text-align:center;position:relative;overflow:hidden;'>
          <div style='position:absolute;inset:0;background:radial-gradient(ellipse at 60% 30%,rgba(232,197,71,0.08) 0%,transparent 65%);'></div>
          <div style='font-family:JetBrains Mono,monospace;font-size:1rem;color:#7a7890;text-transform:uppercase;letter-spacing:0.2em;margin-bottom:8px;'>Predicted Enrollment</div>
          <div style='font-family:Playfair Display,serif;font-size:6rem;font-weight:900;color:#5EAAD9;line-height:1;text-shadow:0 0 80px rgba(232,197,71,0.4);position:relative;z-index:1;'>{best_pred}</div>
          <div style='font-family:JetBrains Mono,monospace;font-size:0.80rem;color:#7a7890;letter-spacing:0.15em;text-transform:uppercase;margin-top:8px;'>students expected</div>
          <div style='display:inline-block;margin-top:12px;font-family:JetBrains Mono,monospace;font-size:0.70rem;color:#4fd1a5;background:rgba(79,209,165,0.1);border:1px solid rgba(79,209,165,0.3);padding:4px 14px;border-radius:20px;'>via {best_name}</div>
        </div>""",unsafe_allow_html=True)
        st.markdown(f"<div class='risk-banner {risk_cls}'>{risk_emoji} &nbsp; {risk_txt} DEMAND — {subject}</div>",unsafe_allow_html=True)
        st.markdown(f"""
        <div class='resource-row'>
          <div class='resource-chip'><div class='r-val'>{sections}</div><div class='r-lbl'>Sections</div></div>
          <div class='resource-chip'><div class='r-val'>{faculty}</div><div class='r-lbl'>Faculty</div></div>
          <div class='resource-chip'><div class='r-val'>{sections}</div><div class='r-lbl'>Rooms</div></div>
          <div class='resource-chip'><div class='r-val'>₹{total//1000}K</div><div class='r-lbl'>Budget</div></div>
        </div>""",unsafe_allow_html=True)
        lr_p=preds.get('Linear Regression',0); dt_p=preds.get('Decision Tree',0); rf_p=preds.get('Random Forest',0)
        st.markdown(f"""
        <div style='margin-top:16px;'>
          <div style='font-family:JetBrains Mono,monospace;font-size:1.30rem;color:#7a7890;text-transform:uppercase;letter-spacing:0.12em;margin-bottom:10px;'>All Models</div>
          <div class='model-compare'>
            <div class='model-pill {"best" if best_name=="Linear Regression" else ""}'><div class='model-pill-name'>Linear</div><div class='model-pill-val'>{lr_p}</div>{'<div class="best-star">★ BEST</div>' if best_name=="Linear Regression" else ''}</div>
            <div class='model-pill {"best" if best_name=="Decision Tree" else ""}'><div class='model-pill-name'>Dec. Tree</div><div class='model-pill-val'>{dt_p}</div>{'<div class="best-star">★ BEST</div>' if best_name=="Decision Tree" else ''}</div>
            <div class='model-pill {"best" if best_name=="Random Forest" else ""}'><div class='model-pill-name'>Rand. Forest</div><div class='model-pill-val'>{rf_p}</div>{'<div class="best-star">★ BEST</div>' if best_name=="Random Forest" else ''}</div>
          </div>
        </div>""",unsafe_allow_html=True)
    st.markdown("<hr class='fancy-divider'>",unsafe_allow_html=True)
    sv1,_=st.columns([1,3])
    with sv1:
        if st.button("📌  Save this Scenario",use_container_width=True):
            st.session_state.saved_scenarios.append({'id':len(st.session_state.saved_scenarios)+1,'course':subject,'emoji':COURSE_EMOJI[subject],'predicted':best_pred,'model':best_name,'faculty_r':faculty_rating,'difficulty':difficulty,'demand':ind_demand,'semester':semester,'season':season,'sections':sections,'faculty':faculty,'budget':total,'risk':risk_txt,'risk_cls':risk_cls})
            st.success(f"✅ Scenario #{len(st.session_state.saved_scenarios)} saved! Go to 📌 Compare tab.")
    st.markdown("<div class='sec-label'>Batch Mode</div>",unsafe_allow_html=True)
    b1,b2=st.columns([1,2])
    with b1:
        st.markdown("Upload a CSV to predict multiple courses at once.")
        tmpl=pd.DataFrame([{'course_name':'AI & ML','semester':3,'faculty_rating':4.2,'previous_enrollment':70,'course_difficulty':3,'is_elective':1,'has_lab':0,'industry_demand':9,'year':2026,'season':'Fall'},{'course_name':'Data Science','semester':4,'faculty_rating':4.0,'previous_enrollment':60,'course_difficulty':4,'is_elective':1,'has_lab':1,'industry_demand':9,'year':2026,'season':'Spring'}])
        buf=io.StringIO(); tmpl.to_csv(buf,index=False)
        st.download_button("↓ Template CSV",buf.getvalue(),file_name='template.csv',mime='text/csv')
    with b2:
        up=st.file_uploader("Drop CSV",type=['csv'],label_visibility="collapsed")
        if up:
            bdf=pd.read_csv(up)
            req=['course_name','semester','faculty_rating','previous_enrollment','course_difficulty','is_elective','has_lab','industry_demand','year','season']
            miss=[c for c in req if c not in bdf.columns]
            if miss: st.error(f"Missing columns: {miss}")
            else:
                out=[]
                for _,r in bdf.iterrows():
                    s=r['course_name']
                    if s not in COURSES: out.append({**r,'Predicted':'—','Model':'—','Sections':'—','Risk':'—'}); continue
                    p=predict(s,int(r['semester']),float(r['faculty_rating']),int(r['previous_enrollment']),int(r['course_difficulty']),bool(r['is_elective']),bool(r['has_lab']),int(r['industry_demand']),int(r['year']),r['season'],mdict,edict)
                    bm=summary_df.loc[summary_df['Subject']==s,'Best Model'].values[0]; pr=p[bm]
                    out.append({**r,'Predicted Enrollment':pr,'Best Model':bm,'Sections':max(1,round(pr/40)),'Risk':risk_info(pr)[0]})
                rdf=pd.DataFrame(out); st.success(f"✅ {len(rdf)} predictions ready")
                st.dataframe(rdf,use_container_width=True)
                st.download_button("↓ Download Results",rdf.to_csv(index=False),file_name='predictions.csv',mime='text/csv')

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — COMPARE
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<div style='padding:28px 0 0 0;'>",unsafe_allow_html=True)
    st.markdown("<div class='sec-label' font-size:2rem>Scenario Comparison</div>",unsafe_allow_html=True)
    st.markdown("<div class='sec-title'>Saved Scenarios</div>",unsafe_allow_html=True)
    if not st.session_state.saved_scenarios:
        st.markdown("""<div style='text-align:center;padding:60px 20px;border:1px dashed #2a2a3a;border-radius:16px;'>
          <div style='font-size:3rem;margin-bottom:12px;'>📌</div>
          <div style='font-family:Playfair Display,serif;font-size:1.3rem;color:#FFFFFF;'>No scenarios saved yet</div>
          <div style='font-family:JetBrains Mono,monospace;font-size:1rem;color:##FFFFFF;margin-top:8px;'>Go to ⚡ Predict · set parameters · click Save</div>
        </div>""",unsafe_allow_html=True)
    else:
        cl1,cl2,_=st.columns([1,1,3])
        with cl1:
            if st.button("🗑  Clear All"): st.session_state.saved_scenarios=[]; st.rerun()
        with cl2:
            sdf_exp=pd.DataFrame(st.session_state.saved_scenarios).drop(columns=['emoji','risk_cls'],errors='ignore')
            st.download_button("↓ Export CSV",sdf_exp.to_csv(index=False),file_name='scenarios.csv',mime='text/csv')
        scenarios=st.session_state.saved_scenarios
        cols=st.columns(min(3,len(scenarios)))
        for i,sc in enumerate(scenarios):
            with cols[i%3]:
                st.markdown(f"""<div class='scenario-card fade-in'>
                  <div class='scenario-num'>#{sc['id']}</div>
                  <div style='font-size:1.8rem;margin-bottom:6px;'>{sc['emoji']}</div>
                  <div class='scenario-course'>{sc['course']}</div>
                  <div class='scenario-pred'>{sc['predicted']}</div>
                  <div class='scenario-meta'>students · {sc['model'][:12]}</div>
                  <div style='margin-top:10px;'><span style='font-family:JetBrains Mono,monospace;font-size:0.6rem;padding:3px 10px;border-radius:20px;background:rgba(232,197,71,0.1);color:#e8c547;border:1px solid rgba(232,197,71,0.3);'>{sc['risk']}</span></div>
                  <div style='margin-top:12px;display:grid;grid-template-columns:1fr 1fr;gap:6px;'>
                    <div style='background:#2e2e40;border-radius:6px;padding:6px;text-align:center;'><div style='font-family:JetBrains Mono,monospace;font-size:0.52rem;color:#7a7890;'>FACULTY</div><div style='font-family:Playfair Display,serif;font-size:0.95rem;font-weight:700;'>{sc['faculty_r']}</div></div>
                    <div style='background:#2e2e40;border-radius:6px;padding:6px;text-align:center;'><div style='font-family:JetBrains Mono,monospace;font-size:0.52rem;color:#7a7890;'>DEMAND</div><div style='font-family:Playfair Display,serif;font-size:0.95rem;font-weight:700;'>{sc['demand']}</div></div>
                    <div style='background:#2e2e40;border-radius:6px;padding:6px;text-align:center;'><div style='font-family:JetBrains Mono,monospace;font-size:0.52rem;color:#7a7890;'>SECTIONS</div><div style='font-family:Playfair Display,serif;font-size:0.95rem;font-weight:700;'>{sc['sections']}</div></div>
                    <div style='background:#2e2e40;border-radius:6px;padding:6px;text-align:center;'><div style='font-family:JetBrains Mono,monospace;font-size:0.52rem;color:#7a7890;'>BUDGET</div><div style='font-family:Playfair Display,serif;font-size:0.95rem;font-weight:700;'>₹{sc['budget']//1000}K</div></div>
                  </div>
                </div>""",unsafe_allow_html=True)
        if len(scenarios)>1:
            st.markdown("<hr class='fancy-divider'>",unsafe_allow_html=True)
            st.markdown("<div class='sec-label'>Side by Side Comparison</div>",unsafe_allow_html=True)
            labels=[f"#{s['id']} {s['course'][:8]}" for s in scenarios]
            fig=make_subplots(rows=1,cols=3,subplot_titles=("Predicted Enrollment","Budget (₹)","Sections"))
            fig.add_trace(go.Bar(x=labels,y=[s['predicted'] for s in scenarios],marker_color=C[:len(labels)],text=[s['predicted'] for s in scenarios],textposition='outside',marker_line_width=0),row=1,col=1)
            fig.add_trace(go.Bar(x=labels,y=[s['budget'] for s in scenarios],marker_color=C[:len(labels)],text=[f"₹{s['budget']//1000}K" for s in scenarios],textposition='outside',marker_line_width=0),row=1,col=2)
            fig.add_trace(go.Bar(x=labels,y=[s['sections'] for s in scenarios],marker_color=C[:len(labels)],text=[s['sections'] for s in scenarios],textposition='outside',marker_line_width=0),row=1,col=3)
            fig.update_layout(**PL,height=360,showlegend=False); fig.update_xaxes(tickangle=20)
            st.plotly_chart(fig,use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — LEADERBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<div style='padding:28px 0 0 0;'>",unsafe_allow_html=True)
    st.markdown("<div class='sec-label'>Semester Rankings</div>",unsafe_allow_html=True)
    st.markdown("<div class='sec-title'>Course Leaderboard</div>",unsafe_allow_html=True)
    lbc1,lbc2,lbc3=st.columns(3)
    with lbc1: lb_season=st.selectbox("Season",['Fall','Spring','Summer'],key='lb_s')
    with lbc2: lb_fr=st.slider("Faculty Rating",3.0,5.0,4.2,0.1,key='lb_fr')
    with lbc3: lb_dept=st.selectbox("Department",['All']+list(DEPARTMENTS.keys()),key='lb_dept')
    lb_rows=[]
    courses_to_rank = COURSES if lb_dept=='All' else DEPARTMENTS[lb_dept]
    for subj in courses_to_rank:
        p=predict(subj,4,lb_fr,65,3,True,False,COURSE_DEMAND_BIAS[subj],2024,lb_season,mdict,edict)
        bm=summary_df.loc[summary_df['Subject']==subj,'Best Model'].values[0]; pr=p[bm]
        r2=summary_df.loc[summary_df['Subject']==subj,'R² Score'].values[0]
        acc=float(str(summary_df.loc[summary_df['Subject']==subj,'Accuracy (±15%)'].values[0]).strip('%'))
        mae=summary_df.loc[summary_df['Subject']==subj,'MAE'].values[0]
        avg=summary_df.loc[summary_df['Subject']==subj,'Avg Enrollment'].values[0]
        hs,hc,hg=health_score(r2,acc,mae,avg)
        lb_rows.append({'course':subj,'emoji':COURSE_EMOJI[subj],'pred':pr,'health':hs,'hcolor':hc,'hgrade':hg,'r2':r2})
    lb_rows=sorted(lb_rows,key=lambda x:x['pred'],reverse=True)
    max_pred=lb_rows[0]['pred']
    for i,r in enumerate(lb_rows):
        rc='gold' if i==0 else ('silver' if i==1 else ('bronze' if i==2 else ''))
        medal='🥇' if i==0 else ('🥈' if i==1 else ('🥉' if i==2 else str(i+1)))
        bar_pct=int((r['pred']/max_pred)*100)
        st.markdown(f"""<div class='lb-row fade-in'>
          <div class='lb-rank {rc}'>{medal}</div>
          <div class='lb-emoji'>{r['emoji']}</div>
          <div class='lb-name'>{r['course']}</div>
          <div class='lb-bar-wrap'><div class='lb-bar' style='width:{bar_pct}%;'></div></div>
          <div style='font-family:JetBrains Mono,monospace;font-size:0.62rem;color:#7a7890;min-width:60px;text-align:center;'>R²={r['r2']}</div>
          <div style='font-family:Playfair Display,serif;font-size:0.75rem;font-weight:700;color:{r["hcolor"]};min-width:80px;text-align:center;'>{r['hgrade']}</div>
          <div class='lb-score'>{r['pred']}</div>
        </div>""",unsafe_allow_html=True)
    st.markdown("<hr class='fancy-divider'>",unsafe_allow_html=True)
    st.markdown("<div class='sec-label'>Health Scores</div>",unsafe_allow_html=True)
    hcols=st.columns(5)
    for i,r in enumerate(lb_rows):
        with hcols[i%5]:
            st.markdown(f"""<div class='health-card fade-in'>
              <div style='font-size:1.6rem;'>{r['emoji']}</div>
              <div class='health-score' style='color:{r["hcolor"]};'>{r['health']}</div>
              <div class='health-name'>{r['course'][:12]}</div>
              <div class='health-grade' style='color:{r["hcolor"]};'>{r['hgrade']}</div>
            </div>""",unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — HEAD TO HEAD
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("<div style='padding:28px 0 0 0;'>",unsafe_allow_html=True)
    st.markdown("<div class='sec-label'>Direct Competition</div>",unsafe_allow_html=True)
    st.markdown("<div class='sec-title'>Head to Head</div>",unsafe_allow_html=True)
    hh1,hh2=st.columns(2)
    with hh1: course_a=st.selectbox("Course A",COURSES,index=0,format_func=lambda x:f"{COURSE_EMOJI[x]}  {x}",key='hha')
    with hh2: course_b=st.selectbox("Course B",COURSES,index=1,format_func=lambda x:f"{COURSE_EMOJI[x]}  {x}",key='hhb')
    p1,p2,p3,p4=st.columns(4)
    with p1: hh_fr=st.slider("Faculty Rating",3.0,5.0,4.2,0.1,key='hh_fr')
    with p2: hh_pe=st.slider("Prev Enrollment",10,150,65,key='hh_pe')
    with p3: hh_sem=st.slider("Semester",1,8,3,key='hh_sem')
    with p4: hh_seas=st.selectbox("Season",['Fall','Spring','Summer'],key='hh_seas')
    pA=predict(course_a,hh_sem,hh_fr,hh_pe,3,True,False,COURSE_DEMAND_BIAS[course_a],2024,hh_seas,mdict,edict)
    pB=predict(course_b,hh_sem,hh_fr,hh_pe,3,True,False,COURSE_DEMAND_BIAS[course_b],2024,hh_seas,mdict,edict)
    bmA=summary_df.loc[summary_df['Subject']==course_a,'Best Model'].values[0]
    bmB=summary_df.loc[summary_df['Subject']==course_b,'Best Model'].values[0]
    predA,predB=pA[bmA],pB[bmB]; winnerA=predA>=predB
    st.markdown(f"""
    <div style='display:grid;grid-template-columns:1fr auto 1fr;gap:0;align-items:stretch;margin:24px 0;'>
      <div class='h2h-side {"h2h-winner" if winnerA else ""}' style='border-radius:16px 0 0 16px;'>
        <div style='font-size:2.2rem;margin-bottom:10px;'>{COURSE_EMOJI[course_a]}</div>
        <div style='font-family:Playfair Display,serif;font-size:1rem;font-weight:700;color:#f0ede8;margin-bottom:14px;'>{course_a}</div>
        <div class='h2h-num' style='color:{"#e8c547" if winnerA else "#7a7890"};'>{predA}</div>
        <div style='font-family:JetBrains Mono,monospace;font-size:0.6rem;color:#7a7890;margin-top:6px;'>students predicted</div>
        {"<div style='margin-top:12px;font-family:JetBrains Mono,monospace;font-size:0.65rem;color:#e8c547;letter-spacing:0.1em;'>★ WINNER</div>" if winnerA else ""}
      </div>
      <div style='background:#111118;border-top:1px solid #2a2a3a;border-bottom:1px solid #2a2a3a;padding:0 24px;display:flex;align-items:center;'>
        <div style='font-family:Playfair Display,serif;font-size:1.6rem;font-weight:900;color:#2a2a3a;'>VS</div>
      </div>
      <div class='h2h-side {"h2h-winner" if not winnerA else ""}' style='border-radius:0 16px 16px 0;'>
        <div style='font-size:2.2rem;margin-bottom:10px;'>{COURSE_EMOJI[course_b]}</div>
        <div style='font-family:Playfair Display,serif;font-size:1rem;font-weight:700;color:#f0ede8;margin-bottom:14px;'>{course_b}</div>
        <div class='h2h-num' style='color:{"#e8c547" if not winnerA else "#7a7890"};'>{predB}</div>
        <div style='font-family:JetBrains Mono,monospace;font-size:0.6rem;color:#7a7890;margin-top:6px;'>students predicted</div>
        {"<div style='margin-top:12px;font-family:JetBrains Mono,monospace;font-size:0.65rem;color:#e8c547;letter-spacing:0.1em;'>★ WINNER</div>" if not winnerA else ""}
      </div>
    </div>""",unsafe_allow_html=True)
    # Simple grouped bar chart — easy to read and understand at a glance
    rowA=summary_df[summary_df['Subject']==course_a].iloc[0]
    rowB=summary_df[summary_df['Subject']==course_b].iloc[0]
    accA=float(str(rowA['Accuracy (±15%)']).strip('%'))
    accB=float(str(rowB['Accuracy (±15%)']).strip('%'))
    hsA=health_score(rowA['R² Score'],accA,rowA['MAE'],rowA['Avg Enrollment'])[0]
    hsB=health_score(rowB['R² Score'],accB,rowB['MAE'],rowB['Avg Enrollment'])[0]

    st.markdown("<div class='sec-label'>Metric by Metric Comparison</div>",unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.8rem;color:#7a7890;margin-bottom:16px;'>Each bar shows the value for that course. Longer = better (except MAE where shorter = better).</div>",unsafe_allow_html=True)

    metrics     = ['Predicted Enrollment','R² Score (×100)','Accuracy %','Health Score','MAE (lower=better)']
    vals_a      = [predA, round(rowA['R² Score']*100,1), accA, hsA, rowA['MAE']]
    vals_b      = [predB, round(rowB['R² Score']*100,1), accB, hsB, rowB['MAE']]

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        name=course_a, x=metrics, y=vals_a,
        marker_color='#e8c547', text=[str(v) for v in vals_a],
        textposition='outside', marker_line_width=0
    ))
    fig_bar.add_trace(go.Bar(
        name=course_b, x=metrics, y=vals_b,
        marker_color='#7c6af7', text=[str(v) for v in vals_b],
        textposition='outside', marker_line_width=0
    ))
    fig_bar.update_layout(**PL, barmode='group', height=400,
                          title='Head to Head — All Key Metrics',
                          legend=dict(font=dict(color='#f0ede8')))
    st.plotly_chart(fig_bar, use_container_width=True)

    # Also show a simple note if comparing across departments
    deptA = COURSE_DEPT.get(course_a,'')
    deptB = COURSE_DEPT.get(course_b,'')
    if deptA != deptB:
        st.markdown(f"""
        <div style='background:rgba(232,197,71,0.08);border:1px solid rgba(232,197,71,0.3);
             border-radius:10px;padding:12px 18px;margin-top:12px;
             font-family:JetBrains Mono,monospace;font-size:0.68rem;color:#e8c547;'>
          ⚠️ Cross-department comparison — {course_a} ({deptA}) vs {course_b} ({deptB})<br>
          <span style='color:#7a7890;'>Enrollment scales differ by department. Music courses naturally have lower enrollment than Tech — this is by design, not a performance gap.</span>
        </div>
        """,unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 5 — ALERTS
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("<div style='padding:28px 0 0 0;'>",unsafe_allow_html=True)
    st.markdown("<div class='sec-label'>Smart Alerts</div>",unsafe_allow_html=True)
    st.markdown("<div class='sec-title'>Enrollment Alert System</div>",unsafe_allow_html=True)
    al1,al2,al3=st.columns(3)
    with al1: alert_fr=st.slider("Faculty Rating",3.0,5.0,4.2,0.1,key='al_fr')
    with al2: alert_season=st.selectbox("Season",['Fall','Spring','Summer'],key='al_s')
    with al3: threshold=st.slider("Alert Threshold",40,180,100,key='al_t')
    st.markdown("<hr class='fancy-divider'>",unsafe_allow_html=True)
    alert_rows=[]
    for subj in COURSES:
        p=predict(subj,4,alert_fr,65,3,True,False,COURSE_DEMAND_BIAS[subj],2024,alert_season,mdict,edict)
        bm=summary_df.loc[summary_df['Subject']==subj,'Best Model'].values[0]
        alert_rows.append({'course':subj,'emoji':COURSE_EMOJI[subj],'pred':p[bm]})
    alert_rows=sorted(alert_rows,key=lambda x:x['pred'],reverse=True)
    critical=[r for r in alert_rows if r['pred']>threshold*1.3]
    high=[r for r in alert_rows if threshold<r['pred']<=threshold*1.3]
    ok=[r for r in alert_rows if r['pred']<=threshold]
    if critical:
        st.markdown(f"<div style='font-family:JetBrains Mono,monospace;font-size:0.62rem;color:#ff4444;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:10px;'>🚨 CRITICAL — {len(critical)} course(s)</div>",unsafe_allow_html=True)
        for r in critical:
            over=r['pred']-threshold; extra=max(1,round(over/40))
            st.markdown(f"""<div class='alert-box alert-critical fade-in'>
              <div class='alert-title'>{r['emoji']} {r['course']} — {r['pred']} students predicted</div>
              <div class='alert-body'>⚠️ {over} over threshold · Add {extra} extra section(s) · Est. extra cost: ₹{extra*(ROOM_COST_PER_SECTION+int(SALARY_PER_FACULTY*1.2)):,}</div>
            </div>""",unsafe_allow_html=True)
    if high:
        st.markdown(f"<div style='font-family:JetBrains Mono,monospace;font-size:0.62rem;color:#f06a6a;text-transform:uppercase;letter-spacing:0.1em;margin:16px 0 10px;'>⚠️ HIGH — {len(high)} course(s)</div>",unsafe_allow_html=True)
        for r in high:
            st.markdown(f"""<div class='alert-box alert-high fade-in'>
              <div class='alert-title'>{r['emoji']} {r['course']} — {r['pred']} students predicted</div>
              <div class='alert-body'>Monitor closely · {r['pred']-threshold} students above threshold</div>
            </div>""",unsafe_allow_html=True)
    if ok:
        st.markdown(f"<div style='font-family:JetBrains Mono,monospace;font-size:0.62rem;color:#4fd1a5;text-transform:uppercase;letter-spacing:0.1em;margin:16px 0 10px;'>✅ NORMAL — {len(ok)} course(s)</div>",unsafe_allow_html=True)
        for r in ok:
            st.markdown(f"""<div class='alert-box alert-ok fade-in'>
              <div class='alert-title'>{r['emoji']} {r['course']} — {r['pred']} students predicted</div>
              <div class='alert-body'>Within capacity · No action needed</div>
            </div>""",unsafe_allow_html=True)
    st.markdown("<hr class='fancy-divider'>",unsafe_allow_html=True)
    st.markdown("<div class='sec-label'>Semester Planning Board</div>",unsafe_allow_html=True)
    st.markdown("<div style='display:flex;gap:10px;margin-bottom:14px;'><span style='background:rgba(255,68,68,0.12);color:#ff4444;font-family:JetBrains Mono,monospace;font-size:0.58rem;padding:3px 10px;border-radius:4px;'>🔴 Critical</span><span style='background:rgba(240,106,106,0.12);color:#f06a6a;font-family:JetBrains Mono,monospace;font-size:0.58rem;padding:3px 10px;border-radius:4px;'>🟠 High</span><span style='background:rgba(232,197,71,0.12);color:#e8c547;font-family:JetBrains Mono,monospace;font-size:0.58rem;padding:3px 10px;border-radius:4px;'>🟡 Moderate</span><span style='background:rgba(79,209,165,0.12);color:#4fd1a5;font-family:JetBrains Mono,monospace;font-size:0.58rem;padding:3px 10px;border-radius:4px;'>🟢 Low</span></div>",unsafe_allow_html=True)
    for subj in COURSES:
        sem_preds=[]
        for sem in range(1,9):
            p=predict(subj,sem,alert_fr,65,3,True,False,COURSE_DEMAND_BIAS[subj],2024,alert_season,mdict,edict)
            bm=summary_df.loc[summary_df['Subject']==subj,'Best Model'].values[0]
            sem_preds.append(p[bm])
        max_sp=max(sem_preds)
        st.markdown(f"<div style='font-family:JetBrains Mono,monospace;font-size:0.6rem;color:#7a7890;text-transform:uppercase;letter-spacing:0.1em;margin:10px 0 6px;'>{COURSE_EMOJI[subj]} {subj}</div>",unsafe_allow_html=True)
        cells=""
        for si,sp in enumerate(sem_preds):
            intensity=sp/max_sp
            if sp>threshold*1.3:   bg,tc=f"rgba(255,68,68,{0.08+intensity*0.18})","#ff4444"
            elif sp>threshold:     bg,tc=f"rgba(240,106,106,{0.08+intensity*0.18})","#f06a6a"
            elif sp>threshold*0.7: bg,tc=f"rgba(232,197,71,{0.08+intensity*0.18})","#e8c547"
            else:                  bg,tc=f"rgba(79,209,165,{0.08+intensity*0.18})","#4fd1a5"
            cells+=f"<div class='sem-cell' style='background:{bg};border-color:{tc}33;'><div class='sem-cell-num'>S{si+1}</div><div class='sem-cell-val' style='color:{tc};'>{sp}</div></div>"
        st.markdown(f"<div class='sem-grid'>{cells}</div>",unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 6 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown("<div style='padding:28px 0 0 0;'>",unsafe_allow_html=True)
    st.markdown("<div class='sec-label'>Dashboard</div>",unsafe_allow_html=True)
    st.markdown("<div class='sec-title'>System Overview</div>",unsafe_allow_html=True)
    avg_r2=summary_df['R² Score'].mean(); r2_pass=(summary_df['R² Score']>=0.90).sum()
    mae_pass=(summary_df['MAE']<15).sum()
    avg_acc=summary_df['Accuracy (±15%)'].apply(lambda x:float(str(x).strip('%'))).mean()
    best_s=summary_df.loc[summary_df['R² Score'].idxmax()]
    st.markdown(f"""<div class='kpi-grid fade-in'>
      <div class='kpi-card'><div class='kpi-label'>Courses Trained</div><div class='kpi-value'>10</div><div class='kpi-sub'>all subjects · 1200 records</div></div>
      <div class='kpi-card'><div class='kpi-label'>Avg R² Score</div><div class='kpi-value'>{avg_r2:.3f}</div><div class='kpi-sub'>{r2_pass}/10 meet &gt;0.90 target</div><div class='kpi-trend-up'>↑ with tuning</div></div>
      <div class='kpi-card'><div class='kpi-label'>Avg Accuracy</div><div class='kpi-value'>{avg_acc:.1f}%</div><div class='kpi-sub'>within ±15% tolerance</div></div>
      <div class='kpi-card'><div class='kpi-label'>MAE Target</div><div class='kpi-value'>{mae_pass}/10</div><div class='kpi-sub'>subjects with MAE &lt; 15</div><div class='kpi-trend-up'>↑ all passing</div></div>
      <div class='kpi-card'><div class='kpi-label'>Best Subject</div><div class='kpi-value'>{COURSE_EMOJI[best_s["Subject"]]}</div><div class='kpi-sub'>{best_s["Subject"]} · R²={best_s["R² Score"]}</div></div>
    </div>""",unsafe_allow_html=True)
    acc_vals=summary_df['Accuracy (±15%)'].apply(lambda x:float(str(x).strip('%')))
    r2_vals=summary_df['R² Score']; subjs=summary_df['Subject']
    fig=make_subplots(rows=1,cols=2,subplot_titles=("R² Score per Subject","Accuracy ±15% per Subject"))
    fig.add_trace(go.Bar(x=subjs,y=r2_vals,marker_color=['#4fd1a5' if r>=0.90 else '#f06a6a' for r in r2_vals],text=[f'{v:.3f}' for v in r2_vals],textposition='outside',marker_line_width=0),row=1,col=1)
    fig.add_hline(y=0.90,line_dash='dash',line_color='#e8c547',annotation_text='0.90 target',row=1,col=1)
    fig.add_trace(go.Bar(x=subjs,y=acc_vals,marker_color=['#4fd1a5' if a>=80 else '#f06a6a' for a in acc_vals],text=[f'{v:.1f}%' for v in acc_vals],textposition='outside',marker_line_width=0),row=1,col=2)
    fig.add_hline(y=80,line_dash='dash',line_color='#e8c547',annotation_text='80% target',row=1,col=2)
    fig.update_layout(**PL,height=380,showlegend=False); fig.update_xaxes(tickangle=35)
    st.plotly_chart(fig,use_container_width=True)
    # Only use columns that definitely exist in summary_df
    dcols_all = ['Subject','Department','Best Model','Accuracy (±15%)','R² Score','CV R²','MAE','RMSE','MAPE (%)','R² ✓','MAE ✓']
    dcols = [c for c in dcols_all if c in summary_df.columns]
    st.dataframe(summary_df[dcols].style.applymap(lambda v:'color:#4fd1a5' if v=='✅' else ('color:#f06a6a' if v=='❌' else ''),subset=['R² ✓','MAE ✓']).background_gradient(subset=['R² Score'],cmap='YlOrBr'),use_container_width=True,height=380)

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 7 — MODELS
# ══════════════════════════════════════════════════════════════════════════════
with tab7:
    st.markdown("<div style='padding:28px 0 0 0;'>",unsafe_allow_html=True)
    st.markdown("<div class='sec-label'>Model Analysis</div>",unsafe_allow_html=True)
    st.markdown("<div class='sec-title'>Performance Deep-Dive</div>",unsafe_allow_html=True)
    sel_subj=st.selectbox("Select subject",COURSES,format_func=lambda x:f"{COURSE_EMOJI[x]}  {x}",key='m_s')
    row=summary_df[summary_df['Subject']==sel_subj].iloc[0]; ar=row['all_results']; bmn=row['Best Model']
    st.markdown(f"<span style='font-family:JetBrains Mono,monospace;font-size:0.68rem;color:#4fd1a5;background:rgba(79,209,165,0.1);border:1px solid rgba(79,209,165,0.3);padding:3px 12px;border-radius:20px;'>⭐ Best: {bmn} · R²={row['R² Score']}</span>",unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(ar).T.style.background_gradient(subset=['R² Score'],cmap='YlOrBr'),use_container_width=True)
    mn=list(ar.keys())
    r2l=[ar[m]['R² Score'] for m in mn]; acl=[ar[m]['Accuracy (±15%)'] for m in mn]
    mal=[ar[m]['MAE'] for m in mn]; rml=[ar[m]['RMSE'] for m in mn]
    c1,c2,c3=st.columns(3)
    with c1:
        fig_r2=go.Figure(go.Bar(x=mn,y=r2l,marker_color=['#4fd1a5' if m==bmn else '#7c6af7' for m in mn],text=[f"{v:.3f}" for v in r2l],textposition='outside',marker_line_width=0))
        fig_r2.add_hline(y=0.90,line_dash='dash',line_color='#e8c547')
        fig_r2.update_layout(**PL,title='R² Score',height=280,yaxis_range=[0,1.2],showlegend=False)
        st.plotly_chart(fig_r2,use_container_width=True)
    with c2:
        fig_ac=go.Figure(go.Bar(x=mn,y=acl,marker_color=['#4fd1a5' if m==bmn else '#7c6af7' for m in mn],text=[f"{v:.1f}%" for v in acl],textposition='outside',marker_line_width=0))
        fig_ac.add_hline(y=80,line_dash='dash',line_color='#e8c547')
        fig_ac.update_layout(**PL,title='Accuracy ±15%',height=280,yaxis_range=[0,115],showlegend=False)
        st.plotly_chart(fig_ac,use_container_width=True)
    with c3:
        fig_e=go.Figure()
        fig_e.add_trace(go.Bar(name='MAE',x=mn,y=mal,marker_color='#bc8cff',marker_line_width=0))
        fig_e.add_trace(go.Bar(name='RMSE',x=mn,y=rml,marker_color='#f06a6a',marker_line_width=0))
        fig_e.add_hline(y=15,line_dash='dash',line_color='#e8c547')
        fig_e.update_layout(**PL,title='MAE vs RMSE',height=280,barmode='group')
        st.plotly_chart(fig_e,use_container_width=True)
    rf=mdict[sel_subj]['Random Forest']; fi=pd.Series(rf.feature_importances_,index=FEATURES).sort_values()
    fig_fi=go.Figure(go.Bar(x=fi.values,y=fi.index,orientation='h',marker_color=['#e8c547' if f==fi.idxmax() else '#7c6af7' for f in fi.index],text=[f'{v:.3f}' for v in fi.values],textposition='outside',marker_line_width=0))
    fig_fi.update_layout(**PL,title=f'Feature Importance — {sel_subj}',height=360,xaxis_title='Importance')
    st.plotly_chart(fig_fi,use_container_width=True)
    hmd=[]
    for _,r in summary_df.iterrows():
        for m in ['Linear Regression','Decision Tree','Random Forest']:
            try: r2v=r['all_results'][m]['R² Score']
            except: r2v=0.0
            hmd.append({'Subject':r['Subject'],'Model':m,'R²':r2v})
    hm=pd.DataFrame(hmd).pivot(index='Model',columns='Subject',values='R²')
    fig_hm=go.Figure(go.Heatmap(z=hm.values,x=hm.columns.tolist(),y=hm.index.tolist(),colorscale='YlOrBr',text=hm.round(3).values,texttemplate='%{text}',textfont=dict(size=10),zmin=0,zmax=1))
    fig_hm.update_layout(**PL,height=260,title='R² — All Models × All Subjects')
    st.plotly_chart(fig_hm,use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 8 — PLANNING
# ══════════════════════════════════════════════════════════════════════════════
with tab8:
    st.markdown("<div style='padding:28px 0 0 0;'>",unsafe_allow_html=True)
    st.markdown("<div class='sec-label'>Academic Operations</div>",unsafe_allow_html=True)
    st.markdown("<div class='sec-title'>Resource Planning</div>",unsafe_allow_html=True)
    prows=[]
    for _,r in summary_df.iterrows():
        avg=r['Avg Enrollment']; sec=max(1,round(avg/40)); fac=max(1,round(sec*1.2))
        fc,rc,mc,tot=budget_calc(int(avg),sec,fac)
        if avg>120: rsk='🔴 Critical'
        elif avg>80: rsk='🟠 High'
        elif avg>50: rsk='🟡 Moderate'
        else: rsk='🟢 Low'
        prows.append({'Subject':r['Subject'],'Avg Enrollment':avg,'Sections':sec,'Faculty':fac,'Rooms':sec,'Est. Budget':f"₹{tot//1000}K",'Risk':rsk,'R²':r['R² Score']})
    plan_df=pd.DataFrame(prows)
    st.dataframe(plan_df,use_container_width=True,height=370)
    c1,c2=st.columns(2)
    rc_map={'🔴 Critical':'#f06a6a','🟠 High':'#f0a05a','🟡 Moderate':'#e8c547','🟢 Low':'#4fd1a5'}
    with c1:
        fig_e=go.Figure(go.Bar(x=plan_df['Subject'],y=plan_df['Avg Enrollment'],marker_color=[rc_map[r] for r in plan_df['Risk']],text=plan_df['Avg Enrollment'],textposition='outside',marker_line_width=0))
        fig_e.update_layout(**PL,title='Avg Enrollment (Risk Coloured)',height=320); fig_e.update_xaxes(tickangle=30)
        st.plotly_chart(fig_e,use_container_width=True)
    with c2:
        fig_a=go.Figure()
        fig_a.add_trace(go.Bar(name='Sections',x=plan_df['Subject'],y=plan_df['Sections'],marker_color='#e8c547',marker_line_width=0))
        fig_a.add_trace(go.Bar(name='Faculty',x=plan_df['Subject'],y=plan_df['Faculty'],marker_color='#7c6af7',marker_line_width=0))
        fig_a.update_layout(**PL,title='Sections & Faculty',barmode='group',height=320); fig_a.update_xaxes(tickangle=30)
        st.plotly_chart(fig_a,use_container_width=True)
    st.markdown("<hr class='fancy-divider'>",unsafe_allow_html=True)
    st.markdown("<div class='sec-label'>Budget Deep Dive</div>",unsafe_allow_html=True)
    bd_subj=st.selectbox("Course",COURSES,format_func=lambda x:f"{COURSE_EMOJI[x]}  {x}",key='bd_s')
    bd_enr=int(summary_df.loc[summary_df['Subject']==bd_subj,'Avg Enrollment'].values[0])
    bd_sec=max(1,round(bd_enr/40)); bd_fac=max(1,round(bd_sec*1.2))
    bd_fc,bd_rc,bd_mc,bd_tot=budget_calc(bd_enr,bd_sec,bd_fac)
    st.markdown(f"""<div class='budget-grid fade-in'>
      <div class='budget-item'><div class='budget-lbl'>Faculty Salaries</div><div class='budget-val'>₹{bd_fc:,}</div><div style='font-family:JetBrains Mono,monospace;font-size:0.58rem;color:#7a7890;margin-top:4px;'>{bd_fac} faculty × ₹{SALARY_PER_FACULTY:,}</div></div>
      <div class='budget-item'><div class='budget-lbl'>Classroom Costs</div><div class='budget-val'>₹{bd_rc:,}</div><div style='font-family:JetBrains Mono,monospace;font-size:0.58rem;color:#7a7890;margin-top:4px;'>{bd_sec} sections × ₹{ROOM_COST_PER_SECTION:,}</div></div>
      <div class='budget-item'><div class='budget-lbl'>Materials</div><div class='budget-val'>₹{bd_mc:,}</div><div style='font-family:JetBrains Mono,monospace;font-size:0.58rem;color:#7a7890;margin-top:4px;'>{bd_enr} students × ₹{MATERIAL_PER_STUDENT}</div></div>
    </div>
    <div style='background:linear-gradient(135deg,#16161f,#1a1a28);border:1px solid #e8c547;border-radius:14px;padding:20px 24px;margin-top:14px;display:flex;justify-content:space-between;align-items:center;'>
      <div style='font-family:JetBrains Mono,monospace;font-size:0.62rem;color:#7a7890;text-transform:uppercase;letter-spacing:0.1em;'>Total Budget — {bd_subj}</div>
      <div style='font-family:Playfair Display,serif;font-size:2rem;font-weight:900;color:#e8c547;'>₹{bd_tot:,}</div>
    </div>""",unsafe_allow_html=True)
    st.markdown("<hr class='fancy-divider'>",unsafe_allow_html=True)
    st.markdown("<div class='sec-label'>What-If Simulator</div>",unsafe_allow_html=True)
    wi_fr=st.slider("Hypothetical Faculty Rating",3.0,5.0,4.0,0.1,key='wi_fr')
    wi=[]
    for subj in COURSES:
        p=predict(subj,4,wi_fr,60,3,True,False,COURSE_DEMAND_BIAS[subj],2024,'Fall',mdict,edict)
        bm=summary_df.loc[summary_df['Subject']==subj,'Best Model'].values[0]; pr=p[bm]
        wi.append({'Subject':subj,'Predicted':pr,'Sections':max(1,round(pr/40))})
    wi_df=pd.DataFrame(wi)
    fig_wi=go.Figure(go.Bar(x=wi_df['Subject'],y=wi_df['Predicted'],marker_color='#e8c547',text=wi_df['Predicted'],textposition='outside',marker_line_width=0))
    fig_wi.update_layout(**PL,title=f'All Courses · Faculty Rating = {wi_fr}',height=300); fig_wi.update_xaxes(tickangle=30)
    st.plotly_chart(fig_wi,use_container_width=True)
    st.download_button("↓ Download Resource Plan CSV",plan_df.to_csv(index=False),file_name='resource_plan.csv',mime='text/csv')
