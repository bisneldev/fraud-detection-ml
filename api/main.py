"""
API REST de détection de fraude bancaire
Modèle : XGBoost (PR-AUC 0,76 — Precision 92%)
"""
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
import time

from api.schemas import (
    TransactionInput, ScoringResult, BatchInput, BatchResult,
    HealthResponse
)


# =====================================================================
# ÉTAT GLOBAL
# =====================================================================
class AppState:
    model = None
    scaler = None
    features = None
    version = "1.0.0"
    
    # Seuils de décision (calibrés sur le dataset)
    # Utilisés pour les décisions BUSINESS (pas les prédictions brutes)
    SEUIL_BLOQUER = 0.7     # Fraude quasi-certaine → blocage
    SEUIL_VERIFIER = 0.3    # Suspect → vérification manuelle
    SEUIL_SURVEILLER = 0.1  # Léger doute → surveillance
    # En dessous de 0.1 → VALIDER

state = AppState()


# =====================================================================
# LIFESPAN
# =====================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Chargement du modèle au démarrage"""
    print("🚀 Démarrage de l'API Détection de Fraude...")
    
    model_path = Path('models/fraud_model.pkl')
    if not model_path.exists():
        raise RuntimeError(
            f"❌ Modèle introuvable : {model_path}\n"
            f"Lance d'abord : python src/fraud_model.py"
        )
    
    data = joblib.load(model_path)
    state.model = data['model']
    state.scaler = data['scaler']
    state.features = data['features']
    
    print(f"✅ Modèle chargé ({type(state.model).__name__})")
    print(f"✅ Features : {len(state.features)}")
    print(f"✅ Seuils : BLOQUER≥{state.SEUIL_BLOQUER}, "
          f"VÉRIFIER≥{state.SEUIL_VERIFIER}, "
          f"SURVEILLER≥{state.SEUIL_SURVEILLER}")
    print("✅ API prête !\n")
    
    yield
    print("\n👋 Arrêt de l'API")


# =====================================================================
# APPLICATION
# =====================================================================
app = FastAPI(
    title="🚨 API Détection de Fraude Bancaire",
    description=(
        "API REST de détection de fraude en temps réel sur transactions bancaires. "
        "Modèle XGBoost entraîné sur 283 745 transactions (dataset ULB/Kaggle). "
        "PR-AUC 0,76 — Precision 92 %."
    ),
    version=state.version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================================
# LOGIQUE DE SCORING
# =====================================================================
def scorer_transaction(transaction_dict: dict) -> ScoringResult:
    """
    Score une transaction et retourne la décision.
    """
    # 1. Préparer les features
    df = pd.DataFrame([transaction_dict])[state.features]
    
    # 2. Scaling
    X_scaled = state.scaler.transform(df)
    
    # 3. Prédiction
    proba = float(state.model.predict_proba(X_scaled)[0, 1])
    score_risque = int(proba * 1000)
    
    # 4. Niveau de risque + décision
    if proba >= state.SEUIL_BLOQUER:
        niveau = 'CRITIQUE'
        decision = 'BLOQUER'
        action = "🚨 BLOQUER IMMÉDIATEMENT — Fraude quasi-certaine. Contacter le client."
    elif proba >= state.SEUIL_VERIFIER:
        niveau = 'ÉLEVÉ'
        decision = 'VÉRIFIER'
        action = "⚠️ VÉRIFICATION MANUELLE — Contacter le client avant validation."
    elif proba >= state.SEUIL_SURVEILLER:
        niveau = 'MOYEN'
        decision = 'SURVEILLER'
        action = "👀 SURVEILLANCE RENFORCÉE — Transaction validée mais tracée."
    else:
        niveau = 'FAIBLE'
        decision = 'VALIDER'
        action = "✅ VALIDER — Transaction conforme au comportement normal."
    
    return ScoringResult(
        probabilite_fraude=round(proba * 100, 4),
        niveau_risque=niveau,
        action_recommandee=action,
        decision=decision,
        score_risque=score_risque
    )


# =====================================================================
# ENDPOINTS
# =====================================================================

@app.get("/", tags=["Accueil"])
async def root():
    """Page d'accueil — liste des endpoints"""
    return {
        "api": "Détection de Fraude Bancaire",
        "version": state.version,
        "status": "✅ opérationnelle",
        "documentation": "/docs",
        "modele": type(state.model).__name__ if state.model else None,
        "endpoints": {
            "GET  /health": "État de l'API",
            "POST /scorer": "Scorer UNE transaction",
            "POST /scorer/batch": "Scorer plusieurs transactions (max 1000)",
            "GET  /seuils": "Voir les seuils de décision",
            "GET  /modele/info": "Informations sur le modèle ML"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health():
    """État de santé de l'API"""
    return HealthResponse(
        status="healthy",
        modele_charge=state.model is not None,
        type_modele=type(state.model).__name__ if state.model else "Non chargé",
        version=state.version,
        timestamp=datetime.now().isoformat()
    )


@app.post("/scorer", response_model=ScoringResult, tags=["Détection"])
async def scorer(transaction: TransactionInput):
    """
    Score UNE transaction bancaire.
    
    Retourne :
    - **probabilite_fraude** : probabilité de fraude (%)
    - **niveau_risque** : FAIBLE / MOYEN / ÉLEVÉ / CRITIQUE
    - **decision** : VALIDER / SURVEILLER / VÉRIFIER / BLOQUER
    - **action_recommandee** : action détaillée
    - **score_risque** : 0-1000
    """
    try:
        t0 = time.time()
        result = scorer_transaction(transaction.model_dump())
        duree = (time.time() - t0) * 1000
        print(f"⏱️  Scoring en {duree:.0f} ms — {result.decision} ({result.probabilite_fraude}%)")
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur de scoring : {str(e)}"
        )


@app.post("/scorer/batch", response_model=BatchResult, tags=["Détection"])
async def scorer_batch(batch: BatchInput):
    """
    Score plusieurs transactions en une seule requête (max 1000).
    Utile pour l'analyse de lots ou le traitement différé.
    """
    try:
        t0 = time.time()
        resultats = [
            scorer_transaction(t.model_dump()) for t in batch.transactions
        ]
        duree = (time.time() - t0) * 1000
        
        # Agrégation
        nb_fraudes = sum(1 for r in resultats if r.decision == 'BLOQUER')
        nb_suspectes = sum(1 for r in resultats if r.decision == 'VÉRIFIER')
        taux = (nb_fraudes + nb_suspectes) / len(resultats) * 100
        
        print(f"⏱️  Batch de {len(resultats)} transactions en {duree:.0f} ms")
        
        return BatchResult(
            total=len(resultats),
            nb_fraudes_detectees=nb_fraudes,
            nb_suspectes=nb_suspectes,
            taux_detection=round(taux, 2),
            resultats=resultats
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur batch : {str(e)}"
        )


@app.get("/seuils", tags=["Configuration"])
async def seuils():
    """Retourne les seuils de décision actuels"""
    return {
        "seuil_bloquer": state.SEUIL_BLOQUER,
        "seuil_verifier": state.SEUIL_VERIFIER,
        "seuil_surveiller": state.SEUIL_SURVEILLER,
        "description": {
            "BLOQUER": f"probabilité ≥ {state.SEUIL_BLOQUER}",
            "VÉRIFIER": f"{state.SEUIL_VERIFIER} ≤ probabilité < {state.SEUIL_BLOQUER}",
            "SURVEILLER": f"{state.SEUIL_SURVEILLER} ≤ probabilité < {state.SEUIL_VERIFIER}",
            "VALIDER": f"probabilité < {state.SEUIL_SURVEILLER}"
        }
    }


@app.get("/modele/info", tags=["Modèle"])
async def modele_info():
    """Informations sur le modèle ML"""
    if state.model is None:
        raise HTTPException(503, "Modèle non chargé")
    
    if hasattr(state.model, 'feature_importances_'):
        importances = state.model.feature_importances_
    else:
        importances = np.zeros(len(state.features))
    
    feature_imp = sorted(
        zip(state.features, importances.tolist()),
        key=lambda x: x[1], reverse=True
    )[:10]
    
    return {
        "type_modele": type(state.model).__name__,
        "nb_features": len(state.features),
        "features": state.features,
        "top_10_importances": [
            {"feature": f, "importance_pct": round(i * 100, 2)}
            for f, i in feature_imp
        ],
        "metriques": {
            "pr_auc": 0.7624,
            "roc_auc": 0.9729,
            "precision": 0.9167,
            "recall": 0.7154,
        }
    }


# =====================================================================
# LANCEMENT DIRECT
# =====================================================================
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="127.0.0.1",
        port=8001,
        reload=True
    )