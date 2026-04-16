import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px

# ── Paths ────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"

# ── Configuração da página ───────────────────────────────────
st.set_page_config(
    page_title="Pump Failure Prediction",
    page_icon="🔧",
    layout="wide",           # usa toda a largura da tela
    initial_sidebar_state="expanded"
)

# ── Cores por classe — usadas em todos os gráficos ───────────
CLASS_COLORS = {
    "Normal":                "#4CAF50",  # verde
    "Desgaste (Valve Plate)": "#FF9800",  # laranja
    "Falha Forçada 1":        "#F44336",  # vermelho
    "Falha Forçada 2":        "#9C27B0",  # roxo
}

# ── Ícones por classe ────────────────────────────────────────
CLASS_ICONS = {
    "Normal":                "✅",
    "Desgaste (Valve Plate)": "⚠️",
    "Falha Forçada 1":        "🔴",
    "Falha Forçada 2":        "🔴",
}


@st.cache_resource  

def load_artifacts():
    """
    Loads model, scaler and metadata from disk. 
    Runs only once per Streamlit session.
    """
    metadata = json.loads(
        (MODELS_DIR / "model_metadata.json").read_text(encoding="utf-8")
    )
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    model  = joblib.load(MODELS_DIR / "xgboost_model.joblib")

    return model, scaler, metadata


model, scaler, metadata = load_artifacts()

BASE_FEATURES  = metadata["base_feature_names"]   # 7 features brutas
MODEL_FEATURES = metadata["feature_names"]         # 37 features do modelo
LABEL_NAMES    = {int(k): v for k, v in metadata["label_names"].items()}

def run_prediction_pipeline(df_raw: pd.DataFrame) -> pd.DataFrame:

    # ── 1. Valida colunas ────────────────────────────────────
    missing_cols = set(BASE_FEATURES) - set(df_raw.columns)
    if missing_cols:
        st.error(f"❌ Missing columns in CSV: {missing_cols}")
        st.stop()

    df = df_raw[BASE_FEATURES].copy()

    # ── 2. Trata NaN ─────────────────────────────────────────
    if df.isnull().any().any():
        df = df.fillna(df.median())

    # ── 3. Normalização nas 7 features base ──────────────────
    # Scaler foi fitado nas 7 features brutas na Etapa 3
    df_scaled = pd.DataFrame(
        scaler.transform(df),
        columns=BASE_FEATURES
    )

    # ── 4. Feature Engineering → 37 features ─────────────────
    df_engineered = _apply_feature_engineering(df_scaled)

    # ── 5. Garante ordem correta das features ────────────────
    df_engineered = df_engineered[MODEL_FEATURES]

    # ── 6. Predição ──────────────────────────────────────────
    predictions   = model.predict(df_engineered)
    probabilities = model.predict_proba(df_engineered)

    # Monta DataFrame de resultados
    results = df_raw.copy()
    results["predicted_label"] = predictions
    results["predicted_class"] = [LABEL_NAMES[p] for p in predictions]
    results["confidence"]      = probabilities.max(axis=1)

    for i, name in LABEL_NAMES.items():
        results[f"prob_{name}"] = probabilities[:, i]

    return results

#For the MVP pupose I'm putting here the feature engineering
def _apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:

    # Ratios
    df["ratio_flow_leak_output"] = (
        df["Flow - leak line"] / (df["Flow - output"] + 1e-8)
    )
    df["ratio_pressure_leak_output"] = (
        df["Pressure - leak line"] / (df["Pressure - output"] + 1e-8)
    )
    df["ratio_temp_diff_output"] = (
        df["Temp. diff"] / (df["Temperature - output"] + 1e-8)
    )

    # Rolling features
    ROLLING_FEATURES = [
        "Pressure - leak line",
        "Temperature - leak line",
        "Flow - leak line",
        "Flow - output",
    ]

    for feature in ROLLING_FEATURES:
        prefix = feature.lower().replace(" ", "_").replace("-", "")
        for window in [5, 10, 30]:
            df[f"{prefix}_mean_{window}s"] = (
                df[feature].rolling(window=window, min_periods=1).mean()
            )
            df[f"{prefix}_std_{window}s"] = (
                df[feature].rolling(window=window, min_periods=1)
                .std().fillna(0)
            )

    # Delta features
    for feature in ["Pressure - output", "Pressure - leak line", "Flow - output"]:
        prefix = feature.lower().replace(" ", "_").replace("-", "")
        df[f"{prefix}_delta"] = df[feature].diff().fillna(0)

    return df

def render_sidebar():
    """
    Sidebar with model filters.
    """
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/pump.png", width=80)
        st.title("🔧 Pump Failure\nPrediction")
        st.markdown("---")

        st.markdown("### 📦 Model in use")
        st.info(f"**{metadata['model_name']}**")
        st.caption(f"Features: {metadata['n_features']}")

        st.markdown("---")
        st.markdown("### 🎨 Caption")
        for classe, color in CLASS_COLORS.items():
            icon = CLASS_ICONS[classe]
            st.markdown(
                f"<span style='color:{color}'>■</span> {icon} {classe}",
                unsafe_allow_html=True
            )

        st.markdown("---")
        st.caption("MVP — Pump Failure Prediction")
        st.caption("Copper Mining | Predictive Maintenance")

def render_summary_metrics(results: pd.DataFrame):
    """
    Cards com resumo das predições no topo do dashboard.
    """
    total      = len(results)
    n_normal   = (results["predicted_class"] == "Normal").sum()
    n_failures = total - n_normal
    avg_conf   = results["confidence"].mean()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="📊 Total Samples",
            value=f"{total:,}"
        )
    with col2:
        st.metric(
            label="✅ Normal Operation",
            value=f"{n_normal:,}",
            delta=f"{n_normal/total*100:.1f}%"
        )
    with col3:
        st.metric(
            label="🔴 Anomalies Detected",
            value=f"{n_failures:,}",
            delta=f"-{n_failures/total*100:.1f}%",
            delta_color="inverse"  # vermelho quando aumenta
        )
    with col4:
        st.metric(
            label="🎯 Mean Confidence",
            value=f"{avg_conf*100:.1f}%"
        )

def render_charts(results: pd.DataFrame):
    """
    Gráficos de distribuição e evolução temporal.
    """
    col1, col2 = st.columns(2)

    # ── Gráfico 1: Distribuição das classes ──────────────────
    with col1:
        st.markdown("#### 📊 Class Distribution")

        dist = results["predicted_class"].value_counts().reset_index()
        dist.columns = ["Class", "Count"]

        fig = px.bar(
            dist,
            x="Class",
            y="Count",
            color="Class",
            color_discrete_map=CLASS_COLORS,
            text="Count",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            showlegend=False,
            xaxis_title="",
            yaxis_title="Quantity",
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Gráfico 2: Confiança por classe ──────────────────────
    with col2:
        st.markdown("#### 🎯 Confidence per Class")

        fig = px.box(
            results,
            x="predicted_class",
            y="confidence",
            color="predicted_class",
            color_discrete_map=CLASS_COLORS,
        )
        fig.update_layout(
            showlegend=False,
            xaxis_title="",
            yaxis_title="Confidence",
            yaxis_tickformat=".0%",
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Gráfico 3: Evolução temporal das predições ────────────
    st.markdown("#### 📈 Temporal Evolution of Predictions")

    # Mapeia classe → número para o gráfico de linha
    class_to_num = {v: k for k, v in LABEL_NAMES.items()}
    results["class_num"] = results["predicted_class"].map(class_to_num)

    fig = px.scatter(
        results.reset_index(),
        x="index",
        y="class_num",
        color="predicted_class",
        color_discrete_map=CLASS_COLORS,
        hover_data=["predicted_class", "confidence"],
    )
    fig.update_layout(
        xaxis_title="SAmple (time →)",
        yaxis_title="Estate",
        yaxis=dict(
            tickvals=[0, 1, 2, 3],
            ticktext=list(LABEL_NAMES.values())
        ),
        height=300,
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)

def render_alerts_table(results: pd.DataFrame):

    st.markdown("#### 🚨 Alertas — Anomalies Samples")

    # Filtra apenas anomalias
    anomalies = results[results["predicted_class"] != "Normal"].copy()

    if anomalies.empty:
        st.success("✅ No anomalies detected in the samples sent!")
        return

    st.warning(f"⚠️ {len(anomalies)} samples with anomaly detected")

    # Colunas relevantes para o operador
    display_cols = BASE_FEATURES + ["predicted_class", "confidence"]
    display_df   = anomalies[display_cols].copy()
    display_df["confidence"] = display_df["confidence"].map("{:.1%}".format)

    # Coloriza por classe
    def highlight_class(row):
        color_map = {
            "Desgaste (Valve Plate)": "background-color: #FFBA4B",
            "Falha Forçada 1":        "background-color: #FF6D82",
            "Falha Forçada 2":        "background-color: #CC6FF1",
        }
        color = color_map.get(row["predicted_class"], "")
        return [color] * len(row)

    st.dataframe(
        display_df.style.apply(highlight_class, axis=1),
        use_container_width=True,
        height=300
    )

    # Botão de download
    csv = anomalies.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download the alerts in CSV",
        data=csv,
        file_name="alertas_pump_failure.csv",
        mime="text/csv"
    )


def main():
    """
    Função principal — orquestra todos os blocos do dashboard.
    """
    render_sidebar()

    st.title("🔧 Pump Failure Prediction Dashboard")
    st.markdown(
        "Uploading sensor readings for diagnostics"
        "the operational status of the hydraulic pump."
    )
    st.markdown("---")

    # ── Upload do CSV ────────────────────────────────────────
    st.markdown("### 📂 Data Upload")

    uploaded_file = st.file_uploader(
        label="Upload CSV with sensor readings",
        type=["csv"],
        help=f"The CSV must contain the columns: {', '.join(BASE_FEATURES)}"
    )

    # ── Exemplo de formato esperado ──────────────────────────
    with st.expander("ℹ️ View expected CSV format"):
        example = pd.DataFrame(
            columns=BASE_FEATURES,
            data=[[0.0] * len(BASE_FEATURES)]
        )
        st.dataframe(example)
        st.caption(
                "The CSV must contain exactly these columns." 
                "Extra columns will be ignored."
        )

    # ── Processamento ────────────────────────────────────────
    if uploaded_file is not None:

        with st.spinner("🔄 Processing predictions..."):
            df_raw  = pd.read_csv(uploaded_file)
            results = run_prediction_pipeline(df_raw)

        st.success(f"✅ {len(results):,} samples processed!")
        st.markdown("---")

        # Renderiza os blocos
        render_summary_metrics(results)
        st.markdown("---")
        render_charts(results)
        st.markdown("---")
        render_alerts_table(results)

    else:
        # Estado inicial — sem upload
        st.info(
            "👆 Upload a CSV with sensor readings "
            "to start the diagnosis."
        )

        # Preview das métricas do modelo
        st.markdown("---")
        st.markdown("### 🏆 Model Performance (test set)")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("F1-Macro",  "99.72%")
        col2.metric("ROC-AUC",   "100.0%")
        col3.metric("Acurácia",  "99.80%")
        col4.metric("F1-Falha1", "99.61%")


if __name__ == "__main__":
    main()