import streamlit as st
import numpy as np
import pandas as pd
import joblib
import datetime
import plotly.graph_objects as go

st.set_page_config(
    page_title="NS Delay Predictor",
    page_icon="🚆",
    layout="centered",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Background */
.stApp {
    background: #0d1117;
    color: #e6edf3;
}

/* Header */
.hero {
    text-align: center;
    padding: 2.5rem 0 1.5rem 0;
    border-bottom: 1px solid #21262d;
    margin-bottom: 2rem;
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.4rem;
    color: #e6edf3;
    letter-spacing: -0.5px;
    margin: 0;
}
.hero h1 span {
    color: #f0b429;
}
.hero p {
    color: #8b949e;
    font-size: 0.95rem;
    margin-top: 0.5rem;
    font-weight: 300;
}

/* Section title */
.section-title {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 0.7rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #f0b429;
    margin-bottom: 1rem;
    margin-top: 1.5rem;
}

/* Result card */
.result-card {
    border-radius: 12px;
    padding: 1.8rem 2rem;
    margin: 1.5rem 0;
    border: 1px solid;
    text-align: center;
}
.result-card.green  { background: #0d2318; border-color: #238636; }
.result-card.yellow { background: #2b1d0a; border-color: #9e6a03; }
.result-card.orange { background: #2d1b00; border-color: #d4730a; }
.result-card.red    { background: #2d0d0d; border-color: #b91c1c; }

.result-card .emoji { font-size: 2.8rem; margin-bottom: 0.4rem; }
.result-card .label {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1.5rem;
    margin-bottom: 0.3rem;
}
.result-card.green  .label { color: #3fb950; }
.result-card.yellow .label { color: #f0b429; }
.result-card.orange .label { color: #ff8c42; }
.result-card.red    .label { color: #f85149; }

.result-card .sublabel {
    color: #8b949e;
    font-size: 0.88rem;
    font-weight: 300;
}

/* Compensation box */
.comp-box {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    margin: 1rem 0;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.comp-box .comp-label { color: #8b949e; font-size: 0.88rem; }
.comp-box .comp-value {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1.4rem;
    color: #e6edf3;
}

/* Stats footer */
.stats-row {
    display: flex;
    gap: 1rem;
    margin-top: 2rem;
    padding-top: 1.5rem;
    border-top: 1px solid #21262d;
}
.stat-box {
    flex: 1;
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.stat-box .stat-val {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1.15rem;
    color: #f0b429;
}
.stat-box .stat-lbl {
    font-size: 0.72rem;
    color: #8b949e;
    margin-top: 0.2rem;
}

/* Divider */
.divider {
    border: none;
    border-top: 1px solid #21262d;
    margin: 1.5rem 0;
}

/* Hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

/* Form inputs dark */
.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stTimeInput > div > div > input {
    background-color: #161b22 !important;
    border-color: #30363d !important;
    color: #e6edf3 !important;
}
label { color: #c9d1d9 !important; font-size: 0.88rem !important; }

/* Button */
.stButton > button {
    background: #f0b429;
    color: #0d1117;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    border: none;
    border-radius: 8px;
    padding: 0.7rem 2rem;
    width: 100%;
    margin-top: 1rem;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

/* Columns gap */
[data-testid="column"] { padding: 0 0.4rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    m1     = joblib.load('brf_model_s1.pkl')
    m2     = joblib.load('brf_model_s2.pkl')
    m3     = joblib.load('brf_model_s3.pkl')
    config = joblib.load('brf_config.pkl')
    return m1, m2, m3, config

try:
    model_s1, model_s2, model_s3, config = load_models()
    FEATURES    = config['features']
    thr_s1      = config['thr_s1']
    thr_s2      = config['thr_s2']
    thr_s3      = config['thr_s3']
    AVG_TICKET  = config['avg_ticket']
    COMP_50     = config['comp_50']
    COMP_100    = config['comp_100']
    MODELS_OK   = True
except Exception as e:
    MODELS_OK = False
    st.error(f"Erreur chargement modèles : {e}")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
SERVICE_TYPES = [
    'Sprinter', 'Intercity', 'Stoptrein', 'Sneltrein',
    'Intercity direct', 'Eurostar', 'ICE International',
    'Nightjet', 'EuroCity', 'Eurocity Direct',
    'Stopbus ipv trein', 'Snelbus ipv trein', 'Metro ipv trein',
    'Taxibus ipv trein', 'Extra trein', 'Speciale Trein',
    'European Sleeper', 'Nachttrein', 'Bus', 'Int. Trein',
    'Belbus', 'Stoomtrein'
]

COMPANIES = [
    'NS', 'Arriva', 'Connexxion', 'Keolis', 'Eurostar',
    'DB', 'Thalys', 'NS International', 'Syntus'
]

CLASS_LABELS = {
    0: 'À l\'heure',
    1: 'Retard mineur',
    2: 'Retard modéré',
    3: 'Retard sévère',
}

def estimate_ticket_price(duration_min):
    if duration_min < 15:  return 5.0
    elif duration_min < 30:  return 8.0
    elif duration_min < 60:  return 14.0
    elif duration_min < 90:  return 22.0
    elif duration_min < 120: return 30.0
    else:                    return 40.0

# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def cascade_predict(X_df):
    proba1  = model_s1.predict_proba(X_df)[:, 1][0]
    pred_cls = 0
    proba_cls3 = 0.0
    proba_cls2 = 0.0
    proba_cls1_step = proba1

    if proba1 >= thr_s1:
        pred_cls = 1
        proba2   = model_s2.predict_proba(X_df)[:, 1][0]
        proba_cls2 = proba2
        if proba2 >= thr_s2:
            pred_cls = 2
            proba3   = model_s3.predict_proba(X_df)[:, 1][0]
            proba_cls3 = proba3
            if proba3 >= thr_s3:
                pred_cls = 3

    # Approximate class probabilities for display
    p0 = 1 - proba1
    p1 = proba1 * (1 - proba_cls2) if pred_cls >= 1 else proba1
    p2 = proba1 * proba_cls2 * (1 - proba_cls3) if pred_cls >= 2 else 0
    p3 = proba1 * proba_cls2 * proba_cls3 if pred_cls >= 3 else 0

    total = p0 + p1 + p2 + p3
    probas = [p0/total, p1/total, p2/total, p3/total]

    return pred_cls, probas


def build_features(dep_hour, day_of_week, month, service_type_enc,
                   company_enc, n_stops, duration_min,
                   platform_change, cancelled_stops,
                   completely_cancelled, partly_cancelled,
                   first_station_enc, last_station_enc,
                   hist_delay_route, hist_delay_hour, hist_delay_type):

    week       = ((month - 1) * 4 + day_of_week // 7 + 1)
    quarter    = (month - 1) // 3 + 1
    is_weekend = int(day_of_week >= 5)
    is_peak    = int((7 <= dep_hour <= 9) or (16 <= dep_hour <= 19))
    is_monday  = int(day_of_week == 0)
    is_friday  = int(day_of_week == 4)
    is_night   = int(dep_hour >= 22 or dep_hour <= 5)
    winter     = int(month in [11, 12, 1, 2])

    platform_change_rate = platform_change / max(n_stops, 1)
    n_platform_changes   = platform_change
    any_arr_cancelled    = int(cancelled_stops > 0)
    any_dep_cancelled    = int(cancelled_stops > 0)
    peak_x_weekday       = is_peak * (1 - is_weekend)
    long_route           = int(n_stops > 10)
    has_cancellation     = int(completely_cancelled or partly_cancelled)
    cancel_severity      = (completely_cancelled * 3 +
                            partly_cancelled * 2 +
                            any_arr_cancelled + any_dep_cancelled)

    row = {
        'month': month, 'day_of_week': day_of_week, 'dep_hour': dep_hour,
        'week': week, 'quarter': quarter,
        'is_weekend': is_weekend, 'is_peak_hour': is_peak,
        'is_monday': is_monday, 'is_friday': is_friday,
        'is_night': is_night, 'winter_month': winter,
        'n_stops': n_stops,
        'n_platform_changes': n_platform_changes,
        'platform_change_rate': platform_change_rate,
        'n_cancelled_stops': cancelled_stops,
        'any_arr_cancelled': any_arr_cancelled,
        'any_dep_cancelled': any_dep_cancelled,
        'Svc_completely_cancelled': int(completely_cancelled),
        'Svc_partly_cancelled': int(partly_cancelled),
        'peak_x_weekday': peak_x_weekday,
        'long_route': long_route,
        'has_cancellation': has_cancellation,
        'cancel_severity': cancel_severity,
        'Service_Type_enc': service_type_enc,
        'Service_Company_enc': company_enc,
        'first_station_enc': first_station_enc,
        'last_station_enc': last_station_enc,
        'service_duration_min': duration_min,
        'hist_delay_route': hist_delay_route,
        'hist_delay_hour': hist_delay_hour,
        'hist_delay_type': hist_delay_type,
    }
    return pd.DataFrame([row])[FEATURES]


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🚆 NS <span>Delay</span> Predictor</h1>
    <p>Balanced Random Forest · Cascade 3 étapes · Barèmes NS réels</p>
</div>
""", unsafe_allow_html=True)

if not MODELS_OK:
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# FORM
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">🔧 Caractéristiques du service</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    service_type = st.selectbox(
        "Type de train",
        SERVICE_TYPES,
        index=0,
        help="Type de service ferroviaire"
    )
    dep_time = st.time_input(
        "Heure de départ",
        value=datetime.time(8, 0),
    )
    dep_date = st.date_input(
        "Date du service",
        value=datetime.date.today(),
    )
    n_stops = st.number_input(
        "Nombre de stops",
        min_value=1, max_value=60,
        value=8,
    )

with col2:
    duration_min = st.number_input(
        "Durée du trajet (minutes)",
        min_value=1, max_value=480,
        value=45,
    )
    cancelled_stops = st.number_input(
        "Arrêts annulés",
        min_value=0, max_value=30,
        value=0,
    )
    platform_change = st.number_input(
        "Changements de quai",
        min_value=0, max_value=20,
        value=0,
    )

st.markdown('<div class="section-title">⚠️ Perturbations</div>', unsafe_allow_html=True)

col3, col4 = st.columns(2)
with col3:
    completely_cancelled = st.checkbox("Service complètement annulé", value=False)
with col4:
    partly_cancelled = st.checkbox("Service partiellement annulé", value=False)

# ─────────────────────────────────────────────────────────────────────────────
# ENCODE INPUTS
# ─────────────────────────────────────────────────────────────────────────────
# Encodages approximatifs basés sur le label encoding du dataset
SERVICE_TYPE_ENC = {t: i for i, t in enumerate(sorted(SERVICE_TYPES))}
COMPANY_ENC = {c: i for i, c in enumerate(sorted(COMPANIES))}

service_type_enc = SERVICE_TYPE_ENC.get(service_type, 0)
company_enc      = 0   # NS par défaut
dep_hour         = dep_time.hour
day_of_week      = dep_date.weekday()
month            = dep_date.month

# Valeurs historiques médianes (issues du dataset)
HIST_DELAY_BY_HOUR = {
    0: 3.2, 1: 2.8, 2: 2.5, 3: 2.4, 4: 2.6, 5: 2.9,
    6: 3.5, 7: 4.8, 8: 5.2, 9: 4.6, 10: 4.1, 11: 4.0,
    12: 4.3, 13: 4.5, 14: 4.7, 15: 5.0, 16: 5.8, 17: 6.2,
    18: 5.9, 19: 5.4, 20: 4.8, 21: 4.2, 22: 3.9, 23: 3.5
}
hist_delay_hour  = HIST_DELAY_BY_HOUR.get(dep_hour, 4.5)
hist_delay_route = 4.5   # médiane globale
hist_delay_type  = 4.5

ticket_price = estimate_ticket_price(duration_min)

# ─────────────────────────────────────────────────────────────────────────────
# PREDICT BUTTON
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<hr class='divider'>", unsafe_allow_html=True)

predict_clicked = st.button("🔍 Prédire le retard")

if predict_clicked:
    X_input = build_features(
        dep_hour=dep_hour,
        day_of_week=day_of_week,
        month=month,
        service_type_enc=service_type_enc,
        company_enc=company_enc,
        n_stops=n_stops,
        duration_min=duration_min,
        platform_change=platform_change,
        cancelled_stops=cancelled_stops,
        completely_cancelled=completely_cancelled,
        partly_cancelled=partly_cancelled,
        first_station_enc=0,
        last_station_enc=1,
        hist_delay_route=hist_delay_route,
        hist_delay_hour=hist_delay_hour,
        hist_delay_type=hist_delay_type,
    )

    pred_cls, probas = cascade_predict(X_input)

    # ── Result card ──────────────────────────────────────────────────────────
    CARD_CONFIG = {
        0: ("green",  "🟢", "À l'heure",       "Retard ≤ 5 min — Aucune compensation"),
        1: ("yellow", "🟡", "Retard mineur",    "Retard entre 5 et 30 min — Alerte passagers"),
        2: ("orange", "🟠", "Retard modéré",    "Retard entre 30 et 60 min — Compensation 50%"),
        3: ("red",    "🔴", "Retard sévère",    "Retard > 60 min — Compensation 100%"),
    }
    color, emoji, label, sublabel = CARD_CONFIG[pred_cls]

    st.markdown(f"""
    <div class="result-card {color}">
        <div class="emoji">{emoji}</div>
        <div class="label">{label}</div>
        <div class="sublabel">{sublabel}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Compensation ─────────────────────────────────────────────────────────
    AVG_PAX    = 200
    CLAIM_RATE = 0.30

    if pred_cls == 0 or pred_cls == 1:
        comp_amount = 0.0
        comp_label  = "Aucune compensation requise"
        comp_detail = "—"
    elif pred_cls == 2:
        comp_amount = ticket_price * AVG_PAX * CLAIM_RATE * 0.50
        comp_label  = f"Compensation 50% · Billet estimé €{ticket_price:.0f}"
        comp_detail = f"€{comp_amount:,.0f}"
    else:
        comp_amount = ticket_price * AVG_PAX * CLAIM_RATE * 1.00
        comp_label  = f"Compensation 100% · Billet estimé €{ticket_price:.0f}"
        comp_detail = f"€{comp_amount:,.0f}"

    st.markdown(f"""
    <div class="comp-box">
        <div>
            <div class="comp-label">Compensation estimée (barèmes NS)</div>
            <div class="comp-label">{comp_label}</div>
        </div>
        <div class="comp-value">{comp_detail}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Probability chart ────────────────────────────────────────────────────
    st.markdown('<div class="section-title">📊 Probabilités par classe</div>', unsafe_allow_html=True)

    fig = go.Figure(go.Bar(
        x=[f"Classe {i}<br>{CLASS_LABELS[i]}" for i in range(4)],
        y=[p * 100 for p in probas],
        marker_color=['#3fb950', '#f0b429', '#ff8c42', '#f85149'],
        text=[f"{p*100:.1f}%" for p in probas],
        textposition='outside',
        textfont=dict(color='#e6edf3', size=12),
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#8b949e', family='DM Sans'),
        yaxis=dict(
            showgrid=True, gridcolor='#21262d',
            ticksuffix='%', range=[0, 105],
            color='#8b949e'
        ),
        xaxis=dict(color='#8b949e'),
        margin=dict(t=20, b=10, l=0, r=0),
        height=280,
        showlegend=False,
    )
    # Highlight predicted class
    colors = ['#3fb950', '#f0b429', '#ff8c42', '#f85149']
    marker_colors = [c if i == pred_cls else c + '55' for i, c in enumerate(colors)]
    fig.data[0].marker.color = marker_colors

    st.plotly_chart(fig, use_container_width=True)

    # ── Risk summary ─────────────────────────────────────────────────────────
    risk_30 = (probas[2] + probas[3]) * 100
    risk_60 = probas[3] * 100
    st.info(
        f"🔎 **Risque de retard ≥ 30 min** : {risk_30:.1f}%  ·  "
        f"**Risque de retard ≥ 60 min** : {risk_60:.1f}%"
    )

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER — Model stats
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="stats-row">
    <div class="stat-box">
        <div class="stat-val">0.3045</div>
        <div class="stat-lbl">F1 Macro (test set)</div>
    </div>
    <div class="stat-box">
        <div class="stat-val">€845K</div>
        <div class="stat-lbl">Coût BRF vs €1.02M naïf</div>
    </div>
    <div class="stat-box">
        <div class="stat-val">+17%</div>
        <div class="stat-lbl">Économie vs baseline</div>
    </div>
    <div class="stat-box">
        <div class="stat-val">€179K</div>
        <div class="stat-lbl">Économie annualisée</div>
    </div>
</div>
""", unsafe_allow_html=True)
