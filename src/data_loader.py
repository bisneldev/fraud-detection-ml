"""
Chargement et diagnostic du dataset de détection de fraude
Dataset : Credit Card Fraud Detection (ULB Machine Learning Group - Kaggle)
"""
import pandas as pd
import numpy as np
from pathlib import Path


# =====================================================================
# CHARGEMENT
# =====================================================================
def charger_donnees(path='data/raw/creditcard.csv'):
    """Charge le dataset de transactions"""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"❌ Dataset introuvable : {path}\n"
            f"Télécharge-le sur : https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
        )
    
    print(f"🔄 Chargement du dataset...")
    df = pd.read_csv(path)
    print(f"✅ {len(df):,} transactions chargées")
    return df


# =====================================================================
# DIAGNOSTIC COMPLET
# =====================================================================
def diagnostic_complet(df, target_col='Class'):
    """
    Diagnostic complet du dataset de fraude.
    Retourne un dict avec toutes les métriques clés.
    """
    print("=" * 75)
    print("📊 DIAGNOSTIC DU DATASET — DÉTECTION DE FRAUDE")
    print("=" * 75)
    
    rapport = {}
    
    # ---- 1. Dimensions ----
    print(f"\n1️⃣ DIMENSIONS")
    print(f"   • Lignes    : {df.shape[0]:,}")
    print(f"   • Colonnes  : {df.shape[1]}")
    rapport['n_lignes'] = df.shape[0]
    rapport['n_colonnes'] = df.shape[1]
    
    # ---- 2. Colonnes ----
    print(f"\n2️⃣ COLONNES ({df.shape[1]} au total)")
    print(f"   • Time       : temps écoulé (secondes) depuis la 1ère transaction")
    print(f"   • V1 à V28   : composantes PCA (anonymisées par la banque)")
    print(f"   • Amount     : montant de la transaction (€)")
    print(f"   • Class      : 0 = légitime, 1 = fraude (variable cible)")
    
    # ---- 3. Valeurs manquantes ----
    print(f"\n3️⃣ VALEURS MANQUANTES")
    missing = df.isnull().sum().sum()
    if missing == 0:
        print("   ✅ Aucune valeur manquante")
    else:
        print(f"   ⚠️  {missing} valeurs manquantes")
    rapport['missing'] = missing
    
    # ---- 4. Doublons ----
    n_dup = df.duplicated().sum()
    print(f"\n4️⃣ DOUBLONS")
    print(f"   • {n_dup:,} lignes dupliquées ({n_dup/len(df)*100:.3f}%)")
    rapport['doublons'] = n_dup
    
    # ---- 5. DÉSÉQUILIBRE DES CLASSES (le point crucial) ----
    print(f"\n5️⃣ DÉSÉQUILIBRE DES CLASSES — ⚠️ POINT CRUCIAL")
    counts = df[target_col].value_counts().sort_index()
    total = len(df)
    
    for label, nb in counts.items():
        nom = "🚨 FRAUDE" if label == 1 else "✅ Légitime"
        pct = nb / total * 100
        print(f"   • {nom:12s} : {nb:>7,} ({pct:.4f}%)")
    
    ratio = counts[0] / counts[1]
    print(f"\n   📊 Ratio de déséquilibre : 1 fraude pour {ratio:.0f} transactions")
    print(f"   ⚠️  Déséquilibre EXTRÊME → stratégie ML spécifique requise")
    rapport['n_fraudes'] = int(counts[1])
    rapport['n_legitimes'] = int(counts[0])
    rapport['taux_fraude'] = counts[1] / total * 100
    rapport['ratio_desequilibre'] = ratio
    
    # ---- 6. Statistiques Time ----
    print(f"\n6️⃣ DIMENSION TEMPORELLE")
    time_hours = df['Time'].max() / 3600
    print(f"   • Période couverte : {time_hours:.1f} heures ({time_hours/24:.1f} jours)")
    print(f"   • Time min : {df['Time'].min():,.0f} s")
    print(f"   • Time max : {df['Time'].max():,.0f} s")
    
    # ---- 7. Statistiques Amount ----
    print(f"\n7️⃣ MONTANTS DES TRANSACTIONS (€)")
    print(f"   • Montant moyen   : {df['Amount'].mean():.2f} €")
    print(f"   • Montant médian  : {df['Amount'].median():.2f} €")
    print(f"   • Montant max     : {df['Amount'].max():,.2f} €")
    print(f"   • Montant min     : {df['Amount'].min():.2f} €")
    print(f"   • Total           : {df['Amount'].sum():,.2f} €")
    
    # ---- 8. Montants : fraude vs légitime ----
    print(f"\n8️⃣ MONTANTS : FRAUDE vs LÉGITIME")
    stats = df.groupby(target_col)['Amount'].agg(['count', 'mean', 'median', 'max'])
    stats.index = ['Légitime', 'Fraude']
    stats.columns = ['Nb', 'Moyen (€)', 'Médian (€)', 'Max (€)']
    print(stats.round(2).to_string())
    
    # ---- 9. Vérification des features V1-V28 ----
    print(f"\n9️⃣ FEATURES V1-V28 (composantes PCA)")
    v_cols = [c for c in df.columns if c.startswith('V') and c[1:].isdigit()]
    print(f"   • Nombre : {len(v_cols)} colonnes")
    print(f"   • Toutes numériques : {df[v_cols].select_dtypes(include=[np.number]).shape[1] == len(v_cols)}")
    print(f"   • Valeurs extrêmes : min={df[v_cols].min().min():.2f}, max={df[v_cols].max().max():.2f}")
    
    # ---- 10. Résumé ----
    print("\n" + "=" * 75)
    print("✅ Diagnostic terminé")
    print("=" * 75)
    print(f"\n⚠️  POINT D'ATTENTION PRINCIPAL :")
    print(f"   Le taux de fraude est de {rapport['taux_fraude']:.4f}%")
    print(f"   → L'accuracy ne sera PAS une métrique pertinente (99,83% en prédisant toujours 0)")
    print(f"   → On utilisera : PR-AUC, Recall, F1-score, matrice de confusion")
    
    return rapport


# =====================================================================
# SCRIPT PRINCIPAL
# =====================================================================
if __name__ == '__main__':
    # Chargement
    df = charger_donnees()
    
    # Diagnostic
    rapport = diagnostic_complet(df)
    
    # Aperçu
    print(f"\n📋 Aperçu (5 premières lignes) :")
    print(df.head().to_string())
    
    print(f"\n📋 Aperçu des 5 premières fraudes :")
    print(df[df['Class'] == 1].head().to_string())