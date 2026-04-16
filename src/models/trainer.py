import pandas as pd
import numpy as np
import logging
import mlflow
import mlflow.sklearn
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    f1_score, roc_auc_score, classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import optuna
from optuna.samplers import TPESampler



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT_DIR    = Path(__file__).resolve().parent.parent.parent 
MODELS_DIR = ROOT_DIR / "models"
REPORT_DIR = ROOT_DIR / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

LABEL_NAMES = {
    0: "Normal",
    1: "Desgaste (Valve Plate)",
    2: "Falha Forçada 1",
    3: "Falha Forçada 2"
}


def setup_mlflow(experiment_name: str = "pump-failure-prediction") -> None:

    db_path = (ROOT_DIR / "mlflow.db").as_posix()
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")

    mlflow.set_experiment(experiment_name)
    logger.info(f"✅ MLflow configurado — experimento: '{experiment_name}'")




def plot_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    model_name: str
) -> str:
    """
    Gera, salva e retorna o path da confusion matrix.

    O que é confusion matrix?
    Uma tabela onde:
    - Linhas = classes reais
    - Colunas = classes preditas
    - Diagonal = acertos
    - Fora da diagonal = erros

    Exemplo de leitura:
    Se linha "Falha Forçada 1" × coluna "Normal" = 50,
    significa que o modelo errou 50 falhas classificando como Normal
    → esse é o tipo de erro mais perigoso no nosso problema

    Por que salvar como imagem?
    O MLflow armazena artefatos — você consegue visualizar
    a confusion matrix de qualquer run histórico na UI,
    mesmo semanas depois do treino.
    """
    cm = confusion_matrix(y_true, y_pred)
    labels = list(LABEL_NAMES.values())

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,       # exibe o número em cada célula
        fmt="d",          # formato inteiro (não científico)
        cmap="Blues",     # azul mais escuro = mais ocorrências
        xticklabels=labels,
        yticklabels=labels,
        ax=ax
    )
    ax.set_title(f"Confusion Matrix — {model_name}")
    ax.set_ylabel("Real")      # eixo Y = o que realmente era
    ax.set_xlabel("Predito")   # eixo X = o que o modelo disse
    plt.tight_layout()

    path = str(REPORT_DIR / f"confusion_matrix_{model_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()  # libera memória — importante em loops de treino

    return path


def plot_feature_importance(
    model,
    feature_names: list,
    model_name: str,
    top_n: int = 20
) -> str:

    importance = pd.Series(
        model.feature_importances_,  # array com importância de cada feature
        index=feature_names
    ).sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    importance.plot(kind="barh", ax=ax, color="steelblue")
    ax.invert_yaxis()  # feature mais importante no topo
    ax.set_title(f"Top {top_n} Features — {model_name}")
    ax.set_xlabel("Importância")
    plt.tight_layout()

    path = str(REPORT_DIR / f"feature_importance_{model_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()

    return path


def run_cross_validation(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = 5
) -> dict:

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,      # embaralha antes de dividir
        random_state=42    # reprodutibilidade
    )

    scores = cross_validate(
        model, X_train, y_train,
        cv=cv,
        scoring={
            # f1_macro: média do F1 de todas as classes com peso igual
            # é nossa métrica principal — não favorece classes majoritárias
            "f1_macro": "f1_macro",

            # roc_auc_ovr: One-vs-Rest — calcula AUC para cada classe
            # contra todas as outras e tira a média
            "roc_auc":  "roc_auc_ovr",

            # accuracy: informativa mas não principal
            # (modelo que sempre diz Normal teria ~44% de acurácia)
            "accuracy": "accuracy"
        },
        return_train_score=False,  # só precisamos do score de validação
        n_jobs=-1
    )

    results = {
        "cv_f1_macro_mean":  scores["test_f1_macro"].mean(),
        "cv_f1_macro_std":   scores["test_f1_macro"].std(),
        "cv_roc_auc_mean":   scores["test_roc_auc"].mean(),
        "cv_roc_auc_std":    scores["test_roc_auc"].std(),
        "cv_accuracy_mean":  scores["test_accuracy"].mean(),
        "cv_accuracy_std":   scores["test_accuracy"].std(),
    }

    logger.info(f"\n📊 Cross-Validation ({n_splits} folds):")
    for k, v in results.items():
        logger.info(f"  {k}: {v:.4f}")

    return results

# Suprime logs verbosos do Optuna — só mostra o essencial
optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> float:
    """
    Função objetivo do Optuna — executada a cada trial.

    O que é um trial?
    Cada trial é uma tentativa do Optuna com um conjunto
    diferente de hiperparâmetros. O Optuna aprende com os
    resultados anteriores para sugerir combinações melhores
    a cada tentativa.

    O que é TPE (Tree-structured Parzen Estimator)?
    É o algoritmo padrão do Optuna para sugerir hiperparâmetros.
    Em vez de busca aleatória, ele constrói um modelo probabilístico
    dos hiperparâmetros que funcionaram bem e foca nessa região.
    É muito mais eficiente que grid search ou random search.

    Returns:
        f1_macro médio do cross-validation — o Optuna vai MAXIMIZAR esse valor
    """

    # trial.suggest_* → Optuna sugere um valor dentro do range definido
    # e aprende com o resultado para sugerir melhor na próxima vez
    params = {
        "n_estimators": trial.suggest_int(
            "n_estimators", 100, 1000
            # int → valores inteiros entre 100 e 1000
            # mais árvores = mais capacidade, mas mais lento
        ),
        "max_depth": trial.suggest_int(
            "max_depth", 3, 10
            # controla complexidade da árvore
            # muito alto = overfitting, muito baixo = underfitting
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.01, 0.3, log=True
            # log=True → busca em escala logarítmica
            # faz sentido porque 0.01 e 0.02 são muito diferentes
            # mas 0.20 e 0.21 têm impacto similar
        ),
        "subsample": trial.suggest_float(
            "subsample", 0.6, 1.0
            # % de amostras usadas por árvore
            # valores < 1.0 adicionam aleatoriedade → reduz overfitting
        ),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", 0.6, 1.0
            # % de features usadas por árvore
            # mesmo raciocínio do subsample
        ),
        "min_child_weight": trial.suggest_int(
            "min_child_weight", 1, 10
            # mínimo de amostras necessárias para criar um nó filho
            # valores maiores = árvore mais conservadora = menos overfitting
        ),
        "gamma": trial.suggest_float(
            "gamma", 0.0, 1.0
            # ganho mínimo para fazer um split
            # 0 = split sempre que possível
            # valores maiores = modelo mais conservador
        ),
        "reg_alpha": trial.suggest_float(
            "reg_alpha", 0.0, 1.0
            # regularização L1 (Lasso)
            # penaliza features com pouca contribuição → sparsity
        ),
        "reg_lambda": trial.suggest_float(
            "reg_lambda", 0.5, 2.0
            # regularização L2 (Ridge)
            # penaliza pesos grandes → modelo mais suave
        ),
        "eval_metric":  "mlogloss",
        "random_state": 42,
        "n_jobs":       -1,
    }

    model = XGBClassifier(**params)

    # Usa cross-validation para avaliar cada trial
    # Mais confiável do que um único split
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    # n_splits=3 no tuning (em vez de 5) → mais rápido
    # no treino final usamos 5 folds

    scores = cross_validate(
        model, X_train, y_train,
        cv=cv,
        scoring="f1_macro",  # métrica que o Optuna vai maximizar
        n_jobs=-1
    )

    return scores["test_score"].mean()


def run_optuna_tuning(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 50,       # número de combinações a testar
    experiment_name: str = "pump-failure-prediction"
) -> dict:
    """
    Executa o tuning de hiperparâmetros com Optuna
    e registra o melhor resultado no MLflow.

    Por que 50 trials?
    - 50 é um bom equilíbrio entre qualidade e tempo
    - Com TPE, os últimos 20 trials já convergem para
      uma região boa do espaço de hiperparâmetros
    - Você pode começar com n_trials=20 para testar
      e aumentar depois se tiver tempo

    Returns:
        dict com os melhores hiperparâmetros encontrados
    """
    setup_mlflow()

    logger.info(f"\n🔍 Iniciando Optuna — {n_trials} trials...\n")

    # Cria o estudo — direction="maximize" porque queremos
    # o maior f1_macro possível
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42),  # seed para reprodutibilidade
        study_name="xgboost_tuning"
    )

    # optimize() executa n_trials vezes a função objective
    study.optimize(
        lambda trial: objective(trial, X_train, y_train),
        n_trials=n_trials,
        show_progress_bar=True  # barra de progresso no terminal
    )

    # Melhor trial encontrado
    best_trial  = study.best_trial
    best_params = best_trial.params
    best_value  = best_trial.value

    logger.info(f"\n🏆 Melhor trial:")
    logger.info(f"  f1_macro: {best_value:.4f}")
    logger.info(f"  Params:   {best_params}")

    # Registra o resultado do tuning no MLflow
    # como um run separado — fica documentado
    with mlflow.start_run(run_name="Optuna_Tuning"):
        mlflow.log_param("n_trials",        n_trials)
        mlflow.log_param("best_params",     str(best_params))
        mlflow.log_metric("best_f1_macro",  best_value)

        # Loga a evolução do tuning — quanto o f1 melhorou
        # a cada trial — útil para ver a curva de convergência
        for i, trial in enumerate(study.trials):
            mlflow.log_metric("trial_f1_macro", trial.value, step=i)

    logger.info("\n✅ Tuning concluído — melhores params prontos para treino final")

    return best_params


def train_model(
    model,
    model_name: str,
    params: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val:   pd.DataFrame,
    y_val:   pd.Series,
    register: bool = True
) -> None:

    setup_mlflow()

    with mlflow.start_run(run_name=model_name):
        # O bloco 'with' garante que o run é fechado
        # corretamente mesmo se ocorrer um erro no meio

        # ── 1. Params ────────────────────────────────────────
        # log_params aceita um dict — loga tudo de uma vez
        mlflow.log_params(params)
        mlflow.log_param("model_type",       model_name)
        mlflow.log_param("n_features",       X_train.shape[1])
        mlflow.log_param("n_train_samples",  len(X_train))
        mlflow.log_param("n_val_samples",    len(X_val))
        mlflow.log_param("smote_applied",    True)

        # ── 2. Cross-validation ──────────────────────────────
        logger.info(f"\n🔄 Cross-Validation — {model_name}...")
        cv_results = run_cross_validation(model, X_train, y_train)
        mlflow.log_metrics(cv_results)

        # ── 3. Treino final ──────────────────────────────────
        # Após o CV, treina no conjunto completo de treino
        # para aproveitar todas as amostras disponíveis
        logger.info(f"\n🏋️ Treinando {model_name}...")
        model.fit(X_train, y_train)

        # ── 4. Avaliação na validação ────────────────────────
        y_pred       = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)
        # predict_proba retorna probabilidade para cada classe
        # ex: [0.85, 0.10, 0.03, 0.02] → 85% Normal

        val_metrics = {
            # macro: média simples entre classes — métrica principal
            "val_f1_macro":    f1_score(y_val, y_pred, average="macro"),

            # weighted: média ponderada pelo tamanho da classe
            # útil como métrica secundária
            "val_f1_weighted": f1_score(y_val, y_pred, average="weighted"),

            # roc_auc com estratégia One-vs-Rest para multiclasse
            "val_roc_auc":     roc_auc_score(
                                   y_val, y_pred_proba,
                                   multi_class="ovr"
                               ),

            "val_accuracy":    (y_val == y_pred).mean(),

            # F1 por classe — average=None retorna array por classe
            # [0] = Normal, [1] = Desgaste, [2] = FF1, [3] = FF2
            "val_f1_normal":   f1_score(y_val, y_pred, average=None)[0],
            "val_f1_desgaste": f1_score(y_val, y_pred, average=None)[1],
            "val_f1_falha1":   f1_score(y_val, y_pred, average=None)[2],
            "val_f1_falha2":   f1_score(y_val, y_pred, average=None)[3],
        }

        mlflow.log_metrics(val_metrics)

        logger.info(f"\n📊 Métricas — {model_name}:")
        for k, v in val_metrics.items():
            logger.info(f"  {k}: {v:.4f}")

        # Classification report: precision, recall e f1 por classe
        # Mais detalhado que as métricas individuais
        report = classification_report(
            y_val, y_pred,
            target_names=list(LABEL_NAMES.values())
        )
        logger.info(f"\n📋 Classification Report:\n{report}")

        # ── 5. Artefatos ─────────────────────────────────────
        cm_path = plot_confusion_matrix(y_val, y_pred, model_name)
        mlflow.log_artifact(cm_path)

        fi_path = plot_feature_importance(
            model, X_train.columns.tolist(), model_name
        )
        mlflow.log_artifact(fi_path)

        # Salva o scaler junto ao modelo
        # — necessário para inferência futura
        mlflow.log_artifact(str(MODELS_DIR / "scaler.joblib"))

        # ── 6. Modelo ────────────────────────────────────────
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=model_name,

            # Model Registry: cria/atualiza o modelo registrado
            # Permite gerenciar versões (Staging → Production)
            registered_model_name=(
                f"PumpFailurePredictor_{model_name}"
                if register else None
            ),

            # input_example: documenta o formato de entrada esperado
            # aparece na UI do MLflow como referência
            input_example=X_val.iloc[:5]
        )

        logger.info(f"\n✅ Run '{model_name}' registrado no MLflow!")




def run_training_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val:   pd.DataFrame,
    y_val:   pd.Series,
    n_trials: int = 50
) -> None:

    logger.info("🚀 Iniciando pipeline de modelagem...\n")

    # ── Passo 1: Tuning com Optuna ───────────────────────────
    # Encontra os melhores hiperparâmetros para o XGBoost
    best_params = run_optuna_tuning(X_train, y_train, n_trials=n_trials)

    # Adiciona params fixos que não entram no tuning
    best_params.update({
        "eval_metric":  "mlogloss",
        "random_state": 42,
        "n_jobs":       -1,
    })

    # ── Passo 2: XGBoost com melhores params ─────────────────
    train_model(
        model=XGBClassifier(**best_params),
        model_name="XGBoost_Tuned",
        params=best_params,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
    )

    # ── Passo 3: Random Forest como baseline ──────────────────
    # Não faz tuning no RF — serve como comparação base
    rf_params = {
        "n_estimators": 200,
        "max_depth":    10,
        "random_state": 42,
        "n_jobs":       -1,
    }

    train_model(
        model=RandomForestClassifier(**rf_params),
        model_name="RandomForest",
        params=rf_params,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
    )

    logger.info("\n🏁 Pipeline de modelagem concluído!")
    logger.info("👉 Rode 'mlflow ui' no terminal para comparar os modelos")


if __name__ == "__main__":
    run_training_pipeline()

