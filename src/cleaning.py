"""
Nettoyage et préparation du dataset de fraude
Attention : préserver les fraudes (déséquilibre 1:578)
"""
import pandas as pd
import numpy as np
from pathlib import Path


# =====================================================================
# NETTOYAGE
# =====================================================================
def nettoyer_donnees(df, verbose=True):
    """
    Pipeline de nettoyage.
    PRIORITÉ : ne JAMAIS supprimer de fraudes.
    """
    df = df.copy()
    n_initial = len(df)
    n_fraudes_initial = int(df['Class'].sum())
    
    rapport = {'n_initial': n_initial, 'n_fraudes_initial': n_fraudes_initial}
    
    if verbose:
        print("=" * 75)
        print("🧹 NETTOYAGE DES TRANSACTIONS")
        print("=" * 75)
        print(f"\n📥 Point de départ : {n_initial:,} transactions")
        print(f"   dont {n_fraudes_initial} fraudes ({n_fraudes_initial/n_initial*100:.4f}%)")
    
    # ================================================================
    # ÉTAPE 1 : Suppression des doublons LÉGITIMES uniquement
    # ================================================================
    n_avant = len(df)
    n_fraudes_avant = int(df['Class'].sum())
    
    # Séparer fraudes et légitimes
    df_fraudes = df[df['Class'] == 1].copy()
    df_legit = df[df['Class'] == 0].copy()
    
    # Supprimer doublons uniquement sur légitimes
    df_legit = df_legit.drop_duplicates()
    df = pd.concat([df_legit, df_fraudes], ignore_index=True)
    
    doublons_supprimes = n_avant - len(df)
    fraudes_preservees = int(df['Class'].sum())
    
    if verbose:
        print(f"\n1️⃣ Suppression des doublons (légitimes uniquement)")
        print(f"   • Doublons supprimés : {doublons_supprimes:,}")
        print(f"   • Fraudes préservées : {fraudes_preservees} ✅")
    
    rapport['doublons_supprimes'] = doublons_supprimes
    
    # ================================================================
    # ÉTAPE 2 : Feature engineering sur Time
    # ================================================================
    # Convertir Time (secondes) en heure de la journée
    df['hour'] = (df['Time'] / 3600) % 24
    
    # Catégoriser le moment de la journée
    df['moment_journee'] = pd.cut(
        df['hour'],
        bins=[0, 6, 12, 18, 24],
        labels=['Nuit (0-6h)', 'Matin (6-12h)', 
                'Après-midi (12-18h)', 'Soir (18-24h)'],
        include_lowest=True
    )
    
    if verbose:
        print(f"\n2️⃣ Feature engineering temporel")
        print(f"   • Colonne 'hour' créée (0-24h)")
        print(f"   • Colonne 'moment_journee' créée (4 catégories)")
    
    # ================================================================
    # ÉTAPE 3 : Feature engineering sur Amount
    # ================================================================
    # Log du montant (réduit l'effet des valeurs extrêmes)
    df['amount_log'] = np.log1p(df['Amount'])
    
    # Catégoriser le montant
    df['tranche_montant'] = pd.cut(
        df['Amount'],
        bins=[-1, 1, 10, 100, 1000, np.inf],
        labels=['<1€', '1-10€', '10-100€', '100-1000€', '>1000€']
    )
    
    if verbose:
        print(f"\n3️⃣ Feature engineering sur les montants")
        print(f"   • Colonne 'amount_log' créée")
        print(f"   • Colonne 'tranche_montant' créée (5 catégories)")
    
    # ================================================================
    # ÉTAPE 4 : Vérification finale
    # ================================================================
    n_final = len(df)
    n_fraudes_final = int(df['Class'].sum())
    
    rapport['n_final'] = n_final
    rapport['n_fraudes_final'] = n_fraudes_final
    rapport['taux_fraude'] = n_fraudes_final / n_final * 100
    
    if verbose:
        print("\n" + "=" * 75)
        print("📊 RAPPORT DE NETTOYAGE")
        print("=" * 75)
        print(f"📥 Transactions initiales : {n_initial:,}")
        print(f"📤 Transactions finales   : {n_final:,}")
        print(f"🗑️  Doublons supprimés    : {doublons_supprimes:,}")
        print(f"🚨 Fraudes préservées     : {n_fraudes_final}/{n_fraudes_initial} "
              f"({n_fraudes_final/n_fraudes_initial*100:.1f}%)")
        print(f"📉 Nouveau taux de fraude : {rapport['taux_fraude']:.4f}%")
        print("=" * 75)
    
    return df, rapport


# =====================================================================
# SAUVEGARDE
# =====================================================================
def sauvegarder_donnees(df, chemin='data/processed/transactions_clean.csv'):
    """Sauvegarde le dataset nettoyé"""
    Path(chemin).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(chemin, index=False)
    print(f"\n💾 Données nettoyées sauvegardées : {chemin}")
    print(f"   Taille : {Path(chemin).stat().st_size / 1e6:.1f} Mo")


# =====================================================================
# SCRIPT PRINCIPAL
# =====================================================================
if __name__ == '__main__':
    # Chargement
    print("🔄 Chargement...")
    df_raw = pd.read_csv('data/raw/creditcard.csv')
    
    # Nettoyage
    df_clean, rapport = nettoyer_donnees(df_raw)
    
    # Sauvegarde
    sauvegarder_donnees(df_clean)
    
    # Aperçu
    print("\n📋 Aperçu :")
    print(df_clean.head())
    
    print(f"\n📋 Colonnes finales ({len(df_clean.columns)}) :")
    for i, col in enumerate(df_clean.columns, 1):
        print(f"   {i:2d}. {col}")
    
    # Vérification déséquilibre
    print(f"\n📊 Vérification déséquilibre :")
    print(df_clean['Class'].value_counts().to_string())