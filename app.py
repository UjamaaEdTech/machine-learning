"""
app.py
======
Smart Crop Recommendation System — Streamlit App
Matches all guide requirements:
  ✅ Title: "Smart Crop Recommendation System"
  ✅ Instruction text for farmers
  ✅ All 7 input parameters (N, P, K, temperature, humidity, rainfall, pH)
  ✅ Predict button
  ✅ Predicted Crop Name with success message
  ✅ Confidence Score
  ✅ Dark theme for high contrast and readability

Run:
    streamlit run app.py
"""

import pickle
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# ── Page config — dark theme ──────────────────────────────────────
st.set_page_config(
    page_title="Smart Crop Recommendation System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Dark theme CSS ────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Lato:wght@300;400;700&display=swap');

  /* Dark background */
  .stApp { background-color: #1a1a1a; }
  section[data-testid="stSidebar"] { background-color: #111111 !important; }
  html, body, [class*="css"] { font-family: 'Lato', sans-serif; color: #e8e0d0; }
  h1, h2, h3 { font-family: 'Playfair Display', serif !important; color: #e8b84b !important; }

  /* Input labels */
  label { color: #c9bfb0 !important; font-weight: 600 !important; }

  /* Sliders */
  .stSlider > div > div { accent-color: #52b788 !important; }

  /* Number inputs */
  input[type="number"] {
    background-color: #2a2a2a !important;
    color: #e8e0d0 !important;
    border: 1px solid #444 !important;
    border-radius: 6px !important;
  }

  /* Predict button */
  .stButton > button {
    background: linear-gradient(135deg, #2d6a2f, #52b788) !important;
    color: white !important;
    border-radius: 10px !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    padding: 0.7rem 2.5rem !important;
    border: none !important;
    width: 100% !important;
    letter-spacing: 0.05em !important;
    transition: opacity 0.2s !important;
  }
  .stButton > button:hover { opacity: 0.88 !important; }

  /* Success box */
  .crop-result {
    background: linear-gradient(135deg, #1a3a1a, #0d2b0d);
    border: 2px solid #52b788;
    border-radius: 14px;
    padding: 28px 32px;
    text-align: center;
    margin: 20px 0;
  }
  .crop-emoji  { font-size: 3.5rem; display: block; margin-bottom: 8px; }
  .crop-title  { font-family: 'Playfair Display', serif; font-size: 2rem; color: #e8b84b; font-weight: 900; }
  .crop-sub    { color: #a8d8a8; font-size: 1rem; margin-top: 6px; }

  /* Confidence bar container */
  .conf-bar-bg {
    background: #2a2a2a;
    border-radius: 100px;
    height: 22px;
    margin: 12px 0;
    overflow: hidden;
    border: 1px solid #444;
  }
  .conf-bar-fill {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, #2d6a2f, #52b788, #a8d8a8);
    transition: width 0.8s ease;
  }

  /* Section divider */
  .section-head {
    color: #52b788 !important;
    font-size: 0.78rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    font-family: 'Lato', monospace !important;
    margin-bottom: 10px;
    border-left: 3px solid #e8b84b;
    padding-left: 10px;
  }

  /* Info cards */
  .info-card {
    background: #222;
    border: 1px solid #333;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 10px;
  }
  .info-card h4 { color: #e8b84b !important; margin-bottom: 4px; font-size: 0.95rem; }
  .info-card p  { color: #a09080; font-size: 0.85rem; line-height: 1.5; margin: 0; }

  /* Instruction box */
  .instruction-box {
    background: #1e1e1e;
    border: 1px solid #3a3a2a;
    border-left: 4px solid #e8b84b;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 24px;
    color: #c9bfb0;
    font-size: 0.95rem;
    line-height: 1.7;
  }
</style>
""", unsafe_allow_html=True)

# ── Load model artefacts ──────────────────────────────────────────
@st.cache_resource
def load_artefacts():
    model  = pickle.load(open('model.pkl',         'rb'))
    scaler = pickle.load(open('scaler.pkl',        'rb'))
    le     = pickle.load(open('label_encoder.pkl', 'rb'))
    return model, scaler, le

try:
    model, scaler, le = load_artefacts()
    model_ready = True
except FileNotFoundError:
    model_ready = False

# ── Crop emoji & info map ─────────────────────────────────────────
CROP_INFO = {
    'rice':        ('🍚', 'Cereal grain — thrives in high rainfall, flooded paddies'),
    'maize':       ('🌽', 'Corn — versatile crop, needs warm climate & moderate rain'),
    'chickpea':    ('🫘', 'Legume — drought tolerant, thrives in dry cool weather'),
    'kidneybeans': ('🫘', 'Legume — needs warm temp, moderate rainfall & loamy soil'),
    'pigeonpeas':  ('🌿', 'Legume — drought resistant, grows in tropical regions'),
    'mothbeans':   ('🌱', 'Legume — extremely drought tolerant, arid conditions'),
    'mungbean':    ('🌱', 'Legume — fast growing, warm & humid climate'),
    'blackgram':   ('🌱', 'Legume — warm climate, moderate rainfall'),
    'lentil':      ('🍵', 'Legume — cool season crop, well-drained soil'),
    'pomegranate': ('🍎', 'Fruit — drought tolerant, hot & dry climate'),
    'banana':      ('🍌', 'Fruit — tropical crop, high humidity & rainfall'),
    'mango':       ('🥭', 'Fruit — tropical tree, dry season needed for flowering'),
    'grapes':      ('🍇', 'Fruit — warm & dry climate, well-drained soil'),
    'watermelon':  ('🍉', 'Fruit — hot climate, sandy loam soil'),
    'muskmelon':   ('🍈', 'Fruit — warm & dry, sandy soil preferred'),
    'apple':       ('🍏', 'Fruit — cool climate, well-drained hilly terrain'),
    'orange':      ('🍊', 'Citrus fruit — subtropical, moderate rainfall'),
    'papaya':      ('🪴', 'Fruit — tropical, warm & well-drained soil'),
    'coconut':     ('🥥', 'Tree — coastal tropical regions, high humidity'),
    'cotton':      ('🌸', 'Fibre crop — black soil, hot & dry climate'),
    'jute':        ('🌿', 'Fibre crop — high rainfall, humid tropical climate'),
    'coffee':      ('☕', 'Beverage crop — cool temp, high humidity, acidic soil'),
}


# ══════════════════════════════════════════════════════════════════
# MAIN TITLE
# ══════════════════════════════════════════════════════════════════
st.markdown("<h1 style='font-size:2.4rem;margin-bottom:4px;'>🌾 Smart Crop Recommendation System</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#a09080;font-size:1rem;margin-bottom:20px;'>AI-powered advisory tool for farmers — powered by Machine Learning</p>", unsafe_allow_html=True)

# ── Instruction text ──────────────────────────────────────────────
st.markdown("""
<div class='instruction-box'>
  <strong style='color:#e8b84b;'>📋 How to use this tool:</strong><br>
  Enter your farm's <strong>soil test results</strong> (Nitrogen, Phosphorus, Potassium, and pH)
  and your <strong>local climate data</strong> (Temperature, Humidity, Rainfall) in the fields below.
  Once all values are entered, click <em>"🔍 Predict Best Crop"</em> to receive a personalised
  crop recommendation with a confidence score.
</div>
""", unsafe_allow_html=True)

if not model_ready:
    st.error("⚠️ Model files not found. Please run `01_crop_recommendation_notebook.ipynb` first to train and save the model.")
    st.stop()


# ══════════════════════════════════════════════════════════════════
# INPUT FORM — All 7 parameters
# ══════════════════════════════════════════════════════════════════
st.markdown("<div class='section-head'>🧱 Soil Nutrients</div>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    N = st.number_input("Nitrogen (N) — ratio", min_value=0, max_value=140, value=50,
                        help="Nitrogen content in soil. Range: 0–140")
with col2:
    P = st.number_input("Phosphorous (P) — ratio", min_value=5, max_value=145, value=50,
                        help="Phosphorus content in soil. Range: 5–145")
with col3:
    K = st.number_input("Potassium (K) — ratio", min_value=5, max_value=205, value=50,
                        help="Potassium content in soil. Range: 5–205")

st.markdown("<br><div class='section-head'>🌤️ Weather Conditions</div>", unsafe_allow_html=True)
col4, col5, col6 = st.columns(3)

with col4:
    temperature = st.number_input("Temperature (°C)", min_value=8.0, max_value=44.0, value=25.0, step=0.5,
                                  help="Average ambient temperature in Celsius. Range: 8–44°C")
with col5:
    humidity = st.number_input("Humidity (%)", min_value=14.0, max_value=100.0, value=70.0, step=0.5,
                               help="Relative humidity percentage. Range: 14–100%")
with col6:
    rainfall = st.number_input("Rainfall (mm)", min_value=20.0, max_value=300.0, value=100.0, step=1.0,
                               help="Average annual rainfall in mm. Range: 20–300mm")

st.markdown("<br><div class='section-head'>🧪 Soil pH</div>", unsafe_allow_html=True)
col7, _, __ = st.columns(3)
with col7:
    ph = st.number_input("Soil pH", min_value=3.5, max_value=9.9, value=6.5, step=0.1,
                         help="Soil acidity/alkalinity. 7 = neutral, <7 = acidic, >7 = alkaline. Range: 3.5–9.9")

st.markdown("<br>", unsafe_allow_html=True)

# ── Predict button ────────────────────────────────────────────────
predict_clicked = st.button("🔍 Predict Best Crop")


# ══════════════════════════════════════════════════════════════════
# PREDICTION OUTPUT
# ══════════════════════════════════════════════════════════════════
if predict_clicked:
    user_input  = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    user_scaled = scaler.transform(user_input)

    proba    = model.predict_proba(user_scaled)[0]
    top_idx  = np.argmax(proba)
    crop     = le.classes_[top_idx]
    conf     = proba[top_idx] * 100

    emoji, description = CROP_INFO.get(crop.lower(), ('🌿', 'A suitable crop for your conditions.'))

    st.markdown("---")

    # ── Primary result — success message with crop name ───────────
    col_res, col_chart = st.columns([1, 1.1])

    with col_res:
        st.markdown(f"""
        <div class='crop-result'>
          <span class='crop-emoji'>{emoji}</span>
          <div class='crop-title'>{crop.title()}</div>
          <div class='crop-sub'>{description}</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Confidence score ──────────────────────────────────────
        st.markdown(f"<p style='color:#c9bfb0;font-size:0.9rem;margin-bottom:4px;'>Model Confidence Score</p>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class='conf-bar-bg'>
          <div class='conf-bar-fill' style='width:{conf:.1f}%;'></div>
        </div>
        <p style='color:#52b788;font-size:1.4rem;font-weight:700;margin:4px 0 0;'>
          {conf:.1f}% confident
        </p>
        """, unsafe_allow_html=True)

        # Confidence label
        if conf >= 80:
            st.success(f"✅ High confidence — **{crop.title()}** is strongly recommended for your farm!")
        elif conf >= 50:
            st.warning(f"⚠️ Moderate confidence — **{crop.title()}** is likely suitable. Consider verifying with local experts.")
        else:
            st.info(f"ℹ️ Low confidence — conditions are borderline. Check top alternatives below.")

    with col_chart:
        # Top 5 probability chart
        top5_idx   = np.argsort(proba)[::-1][:5]
        top5_crops = [le.classes_[i].title() for i in top5_idx]
        top5_probs = [proba[i] * 100 for i in top5_idx]

        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor('#1e1e1e')
        ax.set_facecolor('#1e1e1e')
        colors = ['#52b788','#2d6a2f','#1a4d20','#0d3314','#061a09']
        bars = ax.barh(top5_crops[::-1], top5_probs[::-1], color=colors[::-1], edgecolor='#333')
        for bar, val in zip(bars, top5_probs[::-1]):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{val:.1f}%', va='center', color='#c9bfb0', fontsize=9)
        ax.set_xlabel('Confidence (%)', color='#c9bfb0')
        ax.set_title('Top 5 Crop Recommendations', color='#e8b84b', fontweight='bold', fontsize=11)
        ax.tick_params(colors='#c9bfb0')
        ax.spines[:].set_color('#333')
        ax.set_xlim(0, 105)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Input summary ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("<div class='section-head'>📋 Your Input Summary</div>", unsafe_allow_html=True)
    summary_cols = st.columns(7)
    labels = ['N', 'P', 'K', 'Temp (°C)', 'Humidity (%)', 'pH', 'Rainfall (mm)']
    values = [N, P, K, temperature, humidity, ph, rainfall]
    for col, lbl, val in zip(summary_cols, labels, values):
        col.metric(lbl, val)


# ══════════════════════════════════════════════════════════════════
# BOTTOM INFO — What each parameter means
# ══════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("<div class='section-head'>📚 Parameter Guide</div>", unsafe_allow_html=True)
st.markdown("<h3 style='font-size:1.2rem;margin-bottom:14px;'>What does each input mean?</h3>", unsafe_allow_html=True)

info_cols = st.columns(4)
params = [
    ("Nitrogen (N)", "Helps plants grow leaves and stems. Too little = yellow leaves. Too much = all leaves, no fruit."),
    ("Phosphorus (P)", "Needed for root development and flowering. Important early in the growing season."),
    ("Potassium (K)", "Strengthens plants, improves disease resistance, and boosts fruit quality."),
    ("Soil pH", "Measures acidity. Most crops prefer pH 6–7 (slightly acidic). Extremes block nutrient uptake."),
    ("Temperature", "Average temperature in your region. Crops fail if too hot or too cold."),
    ("Humidity", "Amount of moisture in the air. High humidity suits tropical crops; low humidity suits arid crops."),
    ("Rainfall", "Annual rainfall received. Rice needs a lot; cotton needs relatively little."),
]
for i, (title, desc) in enumerate(params):
    with info_cols[i % 4]:
        st.markdown(f"""
        <div class='info-card'>
          <h4>{title}</h4>
          <p>{desc}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<p style='color:#555;font-size:0.78rem;text-align:center;margin-top:32px;'>Smart Crop Recommendation System · Powered by Random Forest · Thrive Student Project</p>", unsafe_allow_html=True)
