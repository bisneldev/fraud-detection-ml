"""
Schémas Pydantic pour l'API de détection de fraude
"""
from pydantic import BaseModel, Field, ConfigDict
from typing import Literal


# =====================================================================
# TRANSACTION À SCORER
# =====================================================================
class TransactionInput(BaseModel):
    """Une transaction bancaire à analyser"""
    # Features V1-V28 (composantes PCA anonymisées)
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    # Features business
    Amount: float = Field(..., ge=0, description="Montant en €")
    hour: float = Field(..., ge=0, lt=24, description="Heure de la journée (0-23)")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "V1": -1.359807, "V2": -0.072781, "V3": 2.536347,
                "V4": 1.378155, "V5": -0.338321, "V6": 0.462388,
                "V7": 0.239599, "V8": 0.098698, "V9": 0.363787,
                "V10": 0.090794, "V11": -0.551600, "V12": -0.617801,
                "V13": -0.991390, "V14": -0.311169, "V15": 1.468177,
                "V16": -0.470401, "V17": 0.207971, "V18": 0.025791,
                "V19": 0.403993, "V20": 0.251412, "V21": -0.018307,
                "V22": 0.277838, "V23": -0.110474, "V24": 0.066928,
                "V25": 0.128539, "V26": -0.189115, "V27": 0.133558,
                "V28": -0.021053, "Amount": 149.62, "hour": 0.0
            }
        }
    )


# =====================================================================
# RÉSULTAT DE SCORING
# =====================================================================
class ScoringResult(BaseModel):
    """Résultat de l'analyse d'une transaction"""
    probabilite_fraude: float = Field(..., description="Probabilité de fraude (%)")
    niveau_risque: Literal['FAIBLE', 'MOYEN', 'ÉLEVÉ', 'CRITIQUE'] = Field(
        ..., description="Niveau de risque"
    )
    action_recommandee: str = Field(..., description="Action à effectuer")
    decision: Literal['VALIDER', 'SURVEILLER', 'VÉRIFIER', 'BLOQUER'] = Field(
        ..., description="Décision automatique"
    )
    score_risque: int = Field(..., ge=0, le=1000, description="Score 0-1000 (1000 = fraude certaine)")


# =====================================================================
# BATCH DE TRANSACTIONS
# =====================================================================
class BatchInput(BaseModel):
    """Liste de transactions à scorer en une seule requête"""
    transactions: list[TransactionInput] = Field(
        ..., min_length=1, max_length=1000
    )


class BatchResult(BaseModel):
    """Résultats du scoring batch"""
    total: int
    nb_fraudes_detectees: int
    nb_suspectes: int
    taux_detection: float
    resultats: list[ScoringResult]


# =====================================================================
# KPI DE L'API
# =====================================================================
class HealthResponse(BaseModel):
    """État de santé de l'API"""
    status: str
    modele_charge: bool
    type_modele: str
    version: str
    timestamp: str