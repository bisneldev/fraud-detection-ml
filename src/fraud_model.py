"""
Modèle de détection de fraude bancaire
Gestion du déséquilibre massif (1:578) via SMOTE
Comparaison : Random Forest / XGBoost / Isolation Forest
"""
import pandas as pd
import numpy as np
import joblib
import time
from pathlib import Path
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # à mettre AVANT les imports sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
    confusion_matrix, precision_recall_curve, f1_score, precision_score,
    recall_score, roc_curve
)
from imblearn.over_sampling import SMOTE

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠️  XGBoost non installé")


# =====================================================================
# FEATURES
# =====================================================================
def preparer_features(df):
    """Sélectionne les features pour le ML"""
    v_cols = [c for c in df.columns if c.startswith('V') and c[1:].isdigit()]
    features = v_cols + ['Amount', 'hour']
    # Note : on n'utilise PAS 'moment_journee' et 'tranche_montant' (catégorielles)
    # ni 'amount_log' (redondant avec Amount après scaling)
    
    X = df[features].copy()
    y = df['Class'].copy()
    
    return X, y, features


# =====================================================================
# ENTRAÎNEMENT
# =====================================================================
def entrainer_modeles(df, test_size=0.25, random_state=42):
    """
    Version OPTIMISÉE :
    - Random Forest plus léger (rapide, moins de surapprentissage)
    - XGBoost sans SMOTE (utilise scale_pos_weight natif)
    - SMOTE uniquement pour Logistic Regression
    """
    print("=" * 75)
    print("🤖 ENTRAÎNEMENT DES MODÈLES DE DÉTECTION DE FRAUDE (OPTIMISÉ)")
    print("=" * 75)
    
    # ---- Préparation ----
    X, y, features = preparer_features(df)
    
    print(f"\n📊 Dataset : {len(X):,} transactions × {len(features)} features")
    print(f"📉 Taux de fraude : {y.mean()*100:.4f}% ({y.sum()} fraudes)")
    
    # ---- Split stratifié ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"\n🔀 Split stratifié :")
    print(f"   • Train : {len(X_train):,} ({y_train.sum()} fraudes)")
    print(f"   • Test  : {len(X_test):,} ({y_test.sum()} fraudes)")
    
    # ---- Scaling ----
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # ---- SMOTE uniquement pour Logistic Regression ----
    print(f"\n⚖️  Application du SMOTE (pour Logistic Regression uniquement)...")
    smote = SMOTE(random_state=random_state, sampling_strategy=0.1)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
    
    print(f"   • Avant SMOTE : {len(X_train):,} ({y_train.sum()} fraudes)")
    print(f"   • Après SMOTE : {len(X_train_res):,} ({(y_train_res==1).sum()} fraudes, "
          f"{y_train_res.mean()*100:.2f}%)")
    
    # ---- Ratio pour XGBoost sans SMOTE ----
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"   • scale_pos_weight XGBoost : {scale_pos_weight:.1f}")
    
    # ---- Modèles optimisés ----
    modeles = {}
    
    # Logistic Regression (avec SMOTE)
    modeles['Logistic Regression'] = {
        'modele': LogisticRegression(
            max_iter=2000, class_weight='balanced', random_state=random_state
        ),
        'use_smote': True,
    }
    
    # Random Forest (léger mais performant)
    modeles['Random Forest'] = {
        'modele': RandomForestClassifier(
            n_estimators=100,           # Réduit de 200 → 100
            max_depth=10,               # Réduit de 15 → 10
            min_samples_split=10,       # Nouveau
            min_samples_leaf=4,         # Nouveau
            class_weight='balanced_subsample',  # Meilleur pour déséquilibre
            random_state=random_state,
            n_jobs=-1
        ),
        'use_smote': False,             # Pas besoin, class_weight gère
    }
    
    # XGBoost (natif, sans SMOTE)
    if HAS_XGBOOST:
        modeles['XGBoost'] = {
            'modele': XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                scale_pos_weight=scale_pos_weight,  # Gestion native
                subsample=0.8,              # Nouveau
                colsample_bytree=0.8,       # Nouveau
                reg_alpha=0.1,              # Nouveau (L1)
                reg_lambda=0.1,             # Nouveau (L2)
                random_state=random_state,
                eval_metric='aucpr',        # Optimise PR-AUC
                n_jobs=-1
            ),
            'use_smote': False,             # Gestion native
        }
    
    # ---- Entraînement & évaluation ----
    resultats = {}
    
    for nom, config in modeles.items():
        print(f"\n{'─' * 75}")
        print(f"🎯 {nom}")
        if config['use_smote']:
            print(f"   (avec SMOTE)")
        else:
            print(f"   (gestion native du déséquilibre)")
        print(f"{'─' * 75}")
        
        t0 = time.time()
        if config['use_smote']:
            config['modele'].fit(X_train_res, y_train_res)
        else:
            config['modele'].fit(X_train_scaled, y_train)
        duree_train = time.time() - t0
        
        # Prédictions
        y_proba = config['modele'].predict_proba(X_test_scaled)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)
        
        # Métriques
        roc_auc = roc_auc_score(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"   ⏱️  Temps d'entraînement : {duree_train:.1f}s")
        print(f"   🎯 ROC-AUC            : {roc_auc:.4f}")
        print(f"   🎯 PR-AUC  ⭐          : {pr_auc:.4f}")
        print(f"   🎯 Precision (fraude) : {precision:.4f}")
        print(f"   🎯 Recall (fraude)    : {recall:.4f}")
        print(f"   🎯 F1-score (fraude)  : {f1:.4f}")
        print(f"\n   Matrice de confusion :")
        print(f"      Vrais négatifs (TN) : {cm[0,0]:,}")
        print(f"      Faux positifs  (FP) : {cm[0,1]:,}")
        print(f"      Faux négatifs  (FN) : {cm[1,0]:,}  ← fraudes manquées")
        print(f"      Vrais positifs (TP) : {cm[1,1]:,}  ← fraudes détectées")
        
        resultats[nom] = {
            'modele': config['modele'],
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'y_proba': y_proba,
            'y_pred': y_pred,
            'duree_train': duree_train,
        }
    
    # ---- Meilleur modèle ----
    meilleur_nom = max(resultats, key=lambda k: resultats[k]['pr_auc'])
    meilleur = resultats[meilleur_nom]
    
    print(f"\n{'=' * 75}")
    print(f"🏆 MEILLEUR MODÈLE : {meilleur_nom}")
    print(f"   PR-AUC (métrique clé) = {meilleur['pr_auc']:.4f}")
    print(f"   ROC-AUC               = {meilleur['roc_auc']:.4f}")
    print(f"   Recall (fraudes)      = {meilleur['recall']:.4f}")
    print(f"{'=' * 75}")
    
    return {
        'resultats': resultats,
        'meilleur_nom': meilleur_nom,
        'meilleur_modele': meilleur['modele'],
        'scaler': scaler,
        'features': features,
        'X_test': X_test_scaled,
        'y_test': y_test,
    }

# =====================================================================
# ANALYSE DE L'IMPORTANCE DES FEATURES
# =====================================================================
def calculer_importance(modele, features, top_n=15):
    """Importance des features"""
    if hasattr(modele, 'feature_importances_'):
        imp = modele.feature_importances_
    elif hasattr(modele, 'coef_'):
        imp = np.abs(modele.coef_[0])
    else:
        return None
    
    df_imp = pd.DataFrame({
        'feature': features,
        'importance': imp
    }).sort_values('importance', ascending=False).head(top_n)
    
    df_imp['importance_pct'] = (
        df_imp['importance'] / df_imp['importance'].sum() * 100
    ).round(2)
    
    return df_imp


# =====================================================================
# OPTIMISATION DU SEUIL DE DÉCISION
# =====================================================================
def optimiser_seuil(y_test, y_proba):
    """
    Trouve le seuil optimal pour maximiser le F1-score.
    Analyse différents seuils et propose un compromis.
    """
    print("\n" + "=" * 75)
    print("🎯 OPTIMISATION DU SEUIL DE DÉCISION")
    print("=" * 75)
    
    seuils = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    resultats = []
    
    for seuil in seuils:
        y_pred = (y_proba >= seuil).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        fp = cm[0, 1]
        fn = cm[1, 0]
        
        resultats.append({
            'seuil': seuil,
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
            'faux_positifs': int(fp),
            'faux_negatifs': int(fn),
        })
    
    df = pd.DataFrame(resultats)
    print(df.to_string(index=False))
    
    meilleur_f1 = df.loc[df['f1'].idxmax()]
    print(f"\n   🏆 Meilleur F1 : seuil = {meilleur_f1['seuil']}")
    print(f"      Precision  : {meilleur_f1['precision']:.4f}")
    print(f"      Recall     : {meilleur_f1['recall']:.4f}")
    print(f"      F1         : {meilleur_f1['f1']:.4f}")
    
    # Recommandation business
    print(f"\n   💼 RECOMMANDATION BUSINESS :")
    print(f"      En fraude bancaire, on privilégie le RECALL (ne pas rater de fraudes)")
    print(f"      même si cela génère plus de FAUX POSITIFS (frictions clients).")
    print(f"      → Seuil recommandé : 0.3 (détecte {df[df['seuil']==0.3]['recall'].values[0]*100:.1f}% des fraudes)")
    
    return df, meilleur_f1['seuil']


# =====================================================================
# SCORING D'UNE TRANSACTION
# =====================================================================
def scorer_transaction(transaction_dict, modele, scaler, features):
    """Score une nouvelle transaction"""
    X = pd.DataFrame([transaction_dict])[features]
    X_scaled = scaler.transform(X)
    proba = modele.predict_proba(X_scaled)[0, 1]
    
    if proba >= 0.7:
        niveau = 'CRITIQUE'
        action = 'BLOQUER immédiatement'
    elif proba >= 0.3:
        niveau = 'ÉLEVÉ'
        action = 'Vérification manuelle requise'
    elif proba >= 0.1:
        niveau = 'MOYEN'
        action = 'Surveillance renforcée'
    else:
        niveau = 'FAIBLE'
        action = 'Transaction validée'
    
    return {
        'probabilite_fraude': round(proba * 100, 4),
        'niveau_risque': niveau,
        'action_recommandee': action,
    }


# =====================================================================
# SAUVEGARDE
# =====================================================================
def sauvegarder_modele(modele, scaler, features, path='models/fraud_model.pkl'):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        'model': modele,
        'scaler': scaler,
        'features': features,
    }, path)
    print(f"\n💾 Modèle sauvegardé : {path}")


# =====================================================================
# SCRIPT PRINCIPAL
# =====================================================================
if __name__ == '__main__':
    print("🔄 Chargement des données nettoyées...")
    df = pd.read_csv('data/processed/transactions_clean.csv')
    
    # Entraînement
    resultats = entrainer_modeles(df)
    
    # Importance des features
    print("\n" + "=" * 75)
    print("📊 IMPORTANCE DES FEATURES")
    print("=" * 75)
    imp = calculer_importance(resultats['meilleur_modele'], resultats['features'])
    if imp is not None:
        print(imp.to_string(index=False))
    
    # Optimisation du seuil
    df_seuils, seuil_optimal = optimiser_seuil(
        resultats['y_test'],
        resultats['resultats'][resultats['meilleur_nom']]['y_proba']
    )
    
    # Sauvegarde
    sauvegarder_modele(
        resultats['meilleur_modele'],
        resultats['scaler'],
        resultats['features']
    )
    
    print("\n" + "=" * 75)
    print("✅ MODÈLE DE DÉTECTION DE FRAUDE ENTRAÎNÉ")
    print("=" * 75)