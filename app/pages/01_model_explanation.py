import streamlit as st
from pathlib import Path

# ── Configuração da página ───────────────────────────────────
st.set_page_config(
    page_title="Model Explanation",
    page_icon="📖",
    layout="wide"
)

ROOT_DIR   = Path(__file__).resolve().parent.parent.parent
IMAGES_DIR = ROOT_DIR / "app" / "assets"

# ── Header ───────────────────────────────────────────────────
st.title("📖 Model Explanation")
st.markdown("Understanding the data source, experimental setup and the predictive model.")
st.markdown("---")

# ════════════════════════════════════════════════════════════
# SEÇÃO 1 — RESUMO EXECUTIVO
# ════════════════════════════════════════════════════════════
st.markdown("## 📄 Executive Summary — Research Paper")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    **Title:** Machine Learning for Failure Prediction in Hydraulic Pumps
    """)

#with col2:
#    st.info(
#        "📌 **Dataset available at:**\n\n"
#        "[Kaggle — Valve Plate Failure Prediction]"
#        "(https://www.kaggle.com/datasets/mbjunior/"
#        "valve-plate-failure-prediction-in-hydraulic-pumps)"
#    )

st.markdown("### 🎯 Objective")
st.markdown("""
The research evaluates the application of **machine learning methods** to predict
**valve plate failures** in hydraulic piston pumps used in industrial settings such as
mining, automotive, and metal processing.

The core motivation is **predictive maintenance** — detecting component degradation
early enough to avoid unplanned downtime and significant economic losses.
""")

st.markdown("### 🔬 Experimental Setup")
st.markdown("""
A **dedicated laboratory test bench** was developed to generate the dataset under
controlled conditions. The bench consisted of:

- A **piston pump** (37 kW motor, 1485 RPM, max pressure 27 MPa, displacement 45 cm³/rev)
  as the **test unit**
- A **drive pump** providing hydraulic power to the circuit
- **11 sensors** distributed across the hydraulic circuit measuring:
  pressure, temperature, and flow at key points
- **3 additional vibration sensors** (not included in this dataset version)

The pump was operated under **4 distinct conditions**:
""")

# Cards das 4 classes
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.success("""
    **✅ Normal (dane_OT)**

    Pump operating under standard conditions.
    No damage to the valve plate.

    ~68,000 samples
    """)

with c2:
    st.warning("""
    **⚠️ Valve Plate Wear (dane_ut1)**

    Progressive wear of the valve plate.
    Gradual degradation of sealing surfaces.

    ~85,000 samples
    """)

with c3:
    st.error("""
    **🔴 Simulated Failure 1 (dane_ut2)**

    First level of forced failure.
    Physical damage induced on valve plate.

    ~14,000 samples
    """)

with c4:
    st.error("""
    **🔴 Simulated Failure 2 (dane_ut3)**

    Second level of forced failure.
    More severe physical damage induced.

    ~16,000 samples
    """)

st.markdown("### 📊 Dataset Characteristics")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    | Property | Value |
    |---|---|
    | Total samples | ~153,000 |
    | Time resolution | 1 second (averaged) |
    | Number of features | 7 sensor readings |
    | Target variable | 4 operational states |
    | Format | CSV per condition |
    """)

with col2:
    st.markdown("""
    | Feature | Description |
    |---|---|
    | Pressure - leak line | Pressure at leak line [MPa] |
    | Temperature - leak line | Temperature at leak line [°C] |
    | Pressure - output | Pump output pressure [MPa] |
    | Temperature - output | Output temperature [°C] |
    | Flow - leak line | Flow at leak line [L/min] |
    | Flow - output | Pump output flow [L/min] |
    | Temp. diff | Temperature differential [°C] |
    """)

st.markdown("### 🏆 Key Findings from the Paper")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Model Performance (original paper):**
    - Neural Network: **89% accuracy** (best)
    - kNN: second best
    - Tree-based models: ~85% (4% below NN)

    **Our MVP Performance:**
    - XGBoost Tuned: **99.72% F1-macro**
    - ROC-AUC: **100%** across all classes
    """)

    st.caption(
        "The significant improvement over the paper's results is attributed to "
        "Borderline-SMOTE balancing, Optuna hyperparameter tuning, "
        "and 30 engineered features (rolling stats, ratios, deltas)."
    )

with col2:
    st.markdown("""
    **Most Important Features (paper):**
    1. 🥇 **Flow - leak line** — strongest predictor
    2. 🥈 **Pressure - output** — second most important
    3. 🥉 **Flow - output** — third most important
    4. **Temperature - leak line** — relevant for degradation
    """)

    st.info(
        "💡 These findings align with our Feature Importance analysis "
        "from the XGBoost model — validating our pipeline."
    )

st.markdown("---")

# ════════════════════════════════════════════════════════════
# SEÇÃO 2 — DIAGRAMA DO BANCO DE TESTES
# ════════════════════════════════════════════════════════════
st.markdown("## 🔧 Test Bench — Sensor Layout")
st.markdown(
    "The diagram below shows the hydraulic circuit of the test bench "
    "and the position of each sensor used to generate the dataset."
)

# Exibe a imagem do diagrama
image_path = IMAGES_DIR / "test_bench_diagram.png"

if image_path.exists():
    st.image(
        str(image_path),
        caption="Figure 1 — Hydraulic test bench diagram with sensor positions "
                "(Source: Rojek & Blachnik, 2024)",
        use_container_width=True
    )
else:
    st.warning(
        f"⚠️ Image not found at: {image_path}\n\n"
        "Please save the diagram image as: `app/assets/test_bench_diagram.png`"
    )

# Legenda dos sensores
st.markdown("### 📍 Sensor Legend")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **Suction Side**
    | ID | Sensor | Type |
    |---|---|---|
    | 1 | T11-AI11 | Temperature |
    | 2 | PA01-AI09 | Pressure |
    | 4 | T10-AI12 | Temperature |
    | 5 | P03-AI10 | Pressure |
    """)

with col2:
    st.markdown("""
    **Return / Leak Line**
    | ID | Sensor | Type |
    |---|---|---|
    | 6 | STAUFF-AI19 | Flow (leak) |
    | 7 | Filter | — |
    | 8 | T13-AI06 | Temperature |
    | 9 | P17-AI05 | Pressure |
    | 11 | DZR-AI08 | Flow (output) |
    """)

with col3:
    st.markdown("""
    **Output / Drive Side**
    | ID | Sensor | Type |
    |---|---|---|
    | 12 | T06-AI13 | Temperature |
    | 13 | P19-AI01 | Pressure |
    | 14 | Valve | — |
    | 15 | Nm | Torque |
    | 16 | M 3~ | Motor |
    | 18 | P14-AI02 | Pressure |
    | 19 | T12-AI14 | Temperature |
    """)

st.markdown("---")

# ════════════════════════════════════════════════════════════
# SEÇÃO 3 — ARQUITETURA DO MVP
# ════════════════════════════════════════════════════════════
st.markdown("## 🏗️ MVP Architecture")

st.markdown("""
The predictive model was built following a structured ML pipeline:
""")

st.code("""
[Raw Sensor Data — 7 features]
         │
         ▼
[Pre-processing]
  • Remove irrelevant columns (timestamps, redundant sensors)
  • Handle missing values (median imputation)
  • Clip outliers (Z-score threshold = 3.0)
  • Stratified split: 70% train / 15% val / 15% test
  • StandardScaler (fit on train only)
  • Borderline-SMOTE (train only — avoids data leakage)
         │
         ▼
[Feature Engineering — 37 features]
  • 3 Ratio features  (flow/pressure relationships)
  • 24 Rolling stats  (mean + std over 5s, 10s, 30s windows)
  • 3 Delta features  (sample-to-sample variation)
         │
         ▼
[XGBoost Classifier — Optuna Tuned (50 trials)]
  n_estimators=588 | max_depth=10 | learning_rate=0.073
         │
         ▼
[Output: 4-class diagnosis + confidence score]
  ✅ Normal | ⚠️ Valve Plate Wear | 🔴 FF1 | 🔴 FF2
""", language="text")

st.markdown("---")

# ════════════════════════════════════════════════════════════
# SEÇÃO 4 — PERFORMANCE FINAL
# ════════════════════════════════════════════════════════════
st.markdown("## 📈 Final Model Performance (Test Set)")

m1, m2, m3, m4 = st.columns(4)
m1.metric("F1-Macro",        "99.72%", "↑ vs paper 89%")
m2.metric("ROC-AUC",         "100.0%")
m3.metric("Accuracy",        "99.80%")
m4.metric("F1-Falha Forçada","99.58%", "Hardest class")

st.markdown("""
| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| ✅ Normal | 99.9% | 99.9% | 99.9% |
| ⚠️ Valve Plate Wear | 99.8% | 99.8% | 99.8% |
| 🔴 Simulated Failure 1 | 99.6% | 99.6% | 99.6% |
| 🔴 Simulated Failure 2 | 99.7% | 99.7% | 99.7% |
""")

st.caption(
    "Source: Rojek & Blachnik (2024) — "
    "DOI: 10.20944/preprints202407.0284.v1"
)