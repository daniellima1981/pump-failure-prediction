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
    page_title="Failure Prediction in Hydraulic Pumps",
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

@st.cache_data  
def generate_test_sample() -> pd.DataFrame:
    """
    Gera um sample de teste diretamente dos dados processados.
    Inclui amostras de todas as classes para demonstração completa.
    """
    import random

    # Carrega o dataset processado
    processed_path = ROOT_DIR / "data" / "processed" / "dataset_merged.csv"

    if not processed_path.exists():
        st.error(
            f"Dataset not found at: {processed_path}\n\n"
            "Please run the data pipeline first."
        )
        st.stop()

    df = pd.read_csv(processed_path)

    # Garante amostras de todas as classes — melhor para demonstração
    # 50 amostras de cada classe → 200 amostras no total
    sample = (
        df.groupby("label_name", group_keys=False)
        .apply(lambda x: x.sample(min(50, len(x)), random_state=42))
        .reset_index(drop=True)
    )

    # Retorna apenas as features brutas — simula um CSV real de entrada
    return sample[BASE_FEATURES]


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
    with st.sidebar:
        # Carrega imagem local do repositório
        icon_path = ROOT_DIR / "app" / "assets" / "pump.jpg"

        if icon_path.exists():
            st.image(str(icon_path), width=80)
        else:
            st.markdown("# 🔧")  # fallback se imagem não encontrada

        st.title("Pump Failure\nPrediction")
        st.markdown("---")

        st.markdown("### 📦 Model in use")
        st.info(f"**{metadata['model_name']}**")
        st.caption(f"Features: {metadata['n_features']}")

        st.markdown("---")
        st.markdown("### 🎨 Legend")
        for classe, color in CLASS_COLORS.items():
            icon = CLASS_ICONS[classe]
            st.markdown(
                f"<span style='color:{color}'>■</span> {icon} {classe}",
                unsafe_allow_html=True
            )

        st.markdown("---")
        st.caption("MVP — Pump Failure Prediction")
        st.caption("Predictive Maintenance")

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

def render_maintenance_kpis(results: pd.DataFrame):
    """
    KPIs essenciais para o gestor de manutenção.
    Responde: quando começou, qual tipo, quão severo e
    por quanto tempo persistiu.
    """
    st.markdown("### 🔧 Maintenance KPIs")

    total        = len(results)
    failures     = results[results["predicted_class"] != "Normal"]
    n_failures   = len(failures)
    pct_failures = n_failures / total * 100

    # ── Primeiro sinal de anomalia ───────────────────────────
    # Índice da primeira amostra fora do estado normal
    first_anomaly_idx = failures.index.min() if not failures.empty else None

    # ── Classe dominante de falha ────────────────────────────
    dominant_class = (
        failures["predicted_class"].mode()[0]
        if not failures.empty else "None"
    )

    # ── Sequência máxima consecutiva de falhas ───────────────
    # Indica persistência — falhas intermitentes vs contínuas
    # Técnica: identifica blocos consecutivos com groupby + cumsum
    results["is_failure"] = (results["predicted_class"] != "Normal").astype(int)
    results["block"]      = (results["is_failure"] != results["is_failure"].shift()).cumsum()

    max_consecutive = (
        results[results["is_failure"] == 1]
        .groupby("block")
        .size()
        .max()
    ) if n_failures > 0 else 0

    # ── Confiança média nas falhas ───────────────────────────
    avg_failure_confidence = (
        failures["confidence"].mean()
        if not failures.empty else 0
    )

    # ── Renderiza cards ──────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)

    k1.metric(
        label="⚠️ Anomaly Rate",
        value=f"{pct_failures:.1f}%",
        delta=f"{n_failures} samples",
        delta_color="inverse"
    )

    k2.metric(
        label="🕐 First Anomaly at Sample",
        value=f"#{first_anomaly_idx}" if first_anomaly_idx is not None else "None",
        help="Index of the first sample classified outside Normal state"
    )

    k3.metric(
        label="🔁 Max Consecutive Failures",
        value=f"{max_consecutive}",
        help="Longest uninterrupted sequence of anomalous samples"
    )

    k4.metric(
        label="🎯 Dominant Failure Class",
        value=dominant_class,
        help="Most frequent anomaly type detected"
    )

    # ── Tabela resumo por classe ─────────────────────────────
    st.markdown("#### 📋 Summary per Class")

    summary = (
        results.groupby("predicted_class")
        .agg(
            Count=("predicted_class", "count"),
            Pct=("predicted_class", lambda x: f"{len(x)/total*100:.1f}%"),
            Avg_Confidence=("confidence", lambda x: f"{x.mean()*100:.1f}%"),
            Max_Consecutive=("block", lambda x: (
                results.loc[x.index][results.loc[x.index, "is_failure"] == 1]
                .groupby("block").size().max()
                if any(results.loc[x.index, "is_failure"] == 1) else 0
            ))
        )
        .reset_index()
    )

    summary.columns = [
        "Class", "Count", "% of Total",
        "Avg Confidence", "Max Consecutive"
    ]

    # Coloriza por classe
    def color_class(val):
        colors = {
            "Normal":                 "color: #4CAF50; font-weight: bold",
            "Desgaste (Valve Plate)": "color: #FF9800; font-weight: bold",
            "Falha Forçada 1":        "color: #F44336; font-weight: bold",
            "Falha Forçada 2":        "color: #9C27B0; font-weight: bold",
        }
        return colors.get(val, "")

    st.dataframe(
        summary.style.applymap(color_class, subset=["Class"]),
        use_container_width=True,
        hide_index=True
    )

def render_sensor_trends(df_raw: pd.DataFrame, results: pd.DataFrame):
    """
    Evolução temporal dos valores brutos dos sensores.
    Permite ao técnico ver QUANDO o sensor começou a desviar
    — informação que a classificação sozinha não dá.
    """
    st.markdown("### 📈 Sensor Trend Analysis")

    # ── Seletor de sensor ────────────────────────────────────
    selected_sensor = st.selectbox(
        "Select sensor to analyze:",
        options=BASE_FEATURES,
        index=0,
        help="Visualize the raw sensor readings over time, "
             "colored by the model's prediction"
    )

    # ── Monta DataFrame combinado ────────────────────────────
    # Une os valores brutos do sensor com as predições
    trend_df = pd.DataFrame({
        "sample":        range(len(df_raw)),
        "value":         df_raw[selected_sensor].values,
        "predicted_class": results["predicted_class"].values,
        "confidence":    results["confidence"].values,
    })

    # ── Gráfico principal — scatter colorido por classe ──────
    import plotly.graph_objects as go

    fig = go.Figure()

    for classe, color in CLASS_COLORS.items():
        mask = trend_df["predicted_class"] == classe
        subset = trend_df[mask]

        if subset.empty:
            continue

        fig.add_trace(go.Scatter(
            x=subset["sample"],
            y=subset["value"],
            mode="markers",
            name=f"{CLASS_ICONS[classe]} {classe}",
            marker=dict(
                color=color,
                size=4,
                opacity=0.7
            ),
            hovertemplate=(
                f"<b>{classe}</b><br>"
                "Sample: %{x}<br>"
                f"{selected_sensor}: %{{y:.3f}}<br>"
                "Confidence: %{customdata:.1%}<extra></extra>"
            ),
            customdata=subset["confidence"]
        ))

    # ── Linha de média móvel (rolling mean 30s) ───────────────
    # Ajuda a ver a tendência de longo prazo sem ruído
    rolling_mean = trend_df["value"].rolling(window=30, min_periods=1).mean()

    fig.add_trace(go.Scatter(
        x=trend_df["sample"],
        y=rolling_mean,
        mode="lines",
        name="Rolling Mean (30s)",
        line=dict(color="white", width=1.5, dash="dot"),
        opacity=0.6
    ))

    # ── Linha de threshold (média ± 2std do estado Normal) ───
    # Referência visual do range normal de operação
    normal_vals = trend_df[
        trend_df["predicted_class"] == "Normal"
    ]["value"]

    if not normal_vals.empty:
        mean_normal = normal_vals.mean()
        std_normal  = normal_vals.std()
        upper_bound = mean_normal + 2 * std_normal
        lower_bound = mean_normal - 2 * std_normal

        # Banda de normalidade
        fig.add_hrect(
            y0=lower_bound, y1=upper_bound,
            fillcolor="green", opacity=0.05,
            line_width=0,
            annotation_text="Normal range (±2σ)",
            annotation_position="top right",
            annotation_font_color="lightgreen"
        )

        # Linhas de threshold
        fig.add_hline(
            y=upper_bound,
            line_dash="dash",
            line_color="lightgreen",
            line_width=1,
            opacity=0.5
        )
        fig.add_hline(
            y=lower_bound,
            line_dash="dash",
            line_color="lightgreen",
            line_width=1,
            opacity=0.5
        )

    fig.update_layout(
        title=f"Sensor Trend: {selected_sensor}",
        xaxis_title="Sample (time →)",
        yaxis_title=selected_sensor,
        height=400,
        template="plotly_dark",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ── Estatísticas do sensor selecionado ───────────────────
    st.markdown(f"#### 📊 Statistics — {selected_sensor}")

    stats_cols = st.columns(4)

    # Estatísticas por classe para o sensor selecionado
    for i, (classe, color) in enumerate(CLASS_COLORS.items()):
        mask   = trend_df["predicted_class"] == classe
        subset = df_raw.loc[mask, selected_sensor]

        if subset.empty:
            continue

        with stats_cols[i % 4]:
            st.markdown(
                f"<p style='color:{color}; font-weight:bold'>"
                f"{CLASS_ICONS[classe]} {classe}</p>",
                unsafe_allow_html=True
            )
            st.markdown(f"""
            - **Mean:** {subset.mean():.3f}
            - **Std:** {subset.std():.3f}
            - **Min:** {subset.min():.3f}
            - **Max:** {subset.max():.3f}
            """)

    # ── Proxy da curva P-F ───────────────────────────────────
    st.markdown("#### ⏱️ P-F Interval Proxy")
    st.markdown(
        "Consecutive anomalous samples since last Normal state "
        "— estimates how long the pump has been in degradation."
    )

    # Calcula sequências consecutivas por classe ao longo do tempo
    pf_df = trend_df[["sample", "predicted_class"]].copy()
    pf_df["is_anomaly"] = (pf_df["predicted_class"] != "Normal").astype(int)

    # Acumulador — reseta quando volta ao Normal
    cumulative = []
    count = 0
    for val in pf_df["is_anomaly"]:
        if val == 1:
            count += 1
        else:
            count = 0
        cumulative.append(count)

    pf_df["consecutive_anomalies"] = cumulative

    fig_pf = go.Figure()

    fig_pf.add_trace(go.Scatter(
        x=pf_df["sample"],
        y=pf_df["consecutive_anomalies"],
        mode="lines",
        fill="tozeroy",
        line=dict(color="#F44336", width=2),
        fillcolor="rgba(244, 67, 54, 0.2)",
        name="Consecutive anomalies",
        hovertemplate=(
            "Sample: %{x}<br>"
            "Consecutive anomalies: %{y}<extra></extra>"
        )
    ))

    fig_pf.update_layout(
        title="P-F Interval Proxy — Consecutive Anomalous Samples",
        xaxis_title="Sample (time →)",
        yaxis_title="Consecutive anomalies",
        height=250,
        template="plotly_dark",
        showlegend=False
    )

    st.plotly_chart(fig_pf, use_container_width=True)

    st.caption(
        "💡 A rising curve indicates the pump has been in an anomalous state "
        "for an increasing number of consecutive samples. "
        "A drop to zero means the pump returned to Normal. "
        "A curve that never drops to zero = persistent failure = immediate action required."
    )

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
    render_sidebar()

    st.title("🔧 Pump Failure Prediction Dashboard")
    st.markdown(
        "Upload sensor readings for hydraulic pump operational state diagnosis."
    )
    st.markdown("---")

    # ── Seção de entrada de dados ────────────────────────────
    st.markdown("### 📂 Data Input")

    col_upload, col_test = st.columns([3, 1])

    with col_upload:
        uploaded_file = st.file_uploader(
            label="Upload CSV with sensor readings",
            type=["csv"],
            help=f"CSV must contain: {', '.join(BASE_FEATURES)}"
        )

    with col_test:
        st.markdown("**Or run a quick test:**")
        run_test = st.button(
            label="🧪 Run Test Sample",
            help="Generates 200 samples (50 per class) from the dataset "
                 "and runs the full prediction pipeline automatically.",
            use_container_width=True,
            type="primary"    # botão azul destacado
        )

    # ── Formato esperado ─────────────────────────────────────
    with st.expander("ℹ️ Expected CSV format"):
        example = pd.DataFrame(
            columns=BASE_FEATURES,
            data=[[0.0] * len(BASE_FEATURES)]
        )
        st.dataframe(example)
        st.caption(
            "The CSV must contain exactly these columns. "
            "Extra columns will be ignored."
        )

    # ── Lógica de entrada ────────────────────────────────────
    df_input = None
    input_source = None

    if run_test:
        with st.spinner("🔄 Generating test sample..."):
            df_input     = generate_test_sample()
            input_source = "test"

        st.info(
            f"🧪 **Test mode** — {len(df_input)} samples generated "
            f"(50 per class) from the training dataset."
        )

    elif uploaded_file is not None:
        df_input     = pd.read_csv(uploaded_file)
        input_source = "upload"

    # ── Pipeline de predição ─────────────────────────────────
    if df_input is not None:

        with st.spinner("🔄 Running predictions..."):
            results = run_prediction_pipeline(df_input)

        # Badge de fonte dos dados
        if input_source == "test":
            st.success(
                f"✅ Test completed — {len(results):,} samples processed | "
                f"Source: internal test sample"
            )
        else:
            st.success(
                f"✅ {len(results):,} samples processed | "
                f"Source: {uploaded_file.name}"
            )

        st.markdown("---")

        # Renderiza todos os blocos
        render_summary_metrics(results)
        st.markdown("---")
        render_maintenance_kpis(results)
        st.markdown("---")
        render_sensor_trends(df_input, results)
        st.markdown("---")
        render_charts(results)
        st.markdown("---")
        render_alerts_table(results)

    else:
        # Estado inicial — sem input
        st.info(
            "👆 Upload a CSV with sensor readings or click "
            "**Run Test Sample** to see a demonstration."
        )
        st.markdown("---")
        st.markdown("### 🏆 Model Performance (test set)")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("F1-Macro",  "99.72%")
        col2.metric("ROC-AUC",   "100.0%")
        col3.metric("Accuracy",  "99.80%")
        col4.metric("F1-Falha1", "99.61%")


if __name__ == "__main__":
    main()



