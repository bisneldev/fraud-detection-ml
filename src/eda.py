"""
Analyse exploratoire du dataset de fraude
Focus : comprendre le comportement des fraudes vs légitimes
"""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path


# Palette
COLORS = {
    'legit': '#2a9d8f',
    'fraude': '#e63946',
    'primaire': '#1f3a5f',
    'accent': '#f4a261',
}
COLOR_MAP = {0: COLORS['legit'], 1: COLORS['fraude']}


# =====================================================================
# ANALYSE GLOBALE
# =====================================================================
def analyse_globale(df):
    """KPI globaux + analyse rapide"""
    print("=" * 75)
    print("📊 ANALYSE GLOBALE DES TRANSACTIONS")
    print("=" * 75)
    
    stats = {
        'total': len(df),
        'fraudes': int(df['Class'].sum()),
        'legitimes': int((df['Class'] == 0).sum()),
        'taux_fraude': df['Class'].mean() * 100,
        'montant_total': df['Amount'].sum(),
        'montant_fraude': df.loc[df['Class'] == 1, 'Amount'].sum(),
    }
    
    print(f"\n📈 Indicateurs clés :")
    print(f"   • Total transactions     : {stats['total']:,}")
    print(f"   • Légitimes              : {stats['legitimes']:,}")
    print(f"   • Fraudes                : {stats['fraudes']:,} ({stats['taux_fraude']:.4f}%)")
    print(f"   • Montant total          : {stats['montant_total']:,.2f} €")
    print(f"   • Montant fraudé         : {stats['montant_fraude']:,.2f} €")
    print(f"   • Perte potentielle      : {stats['montant_fraude']:,.2f} €")
    print(f"   • % du montant total     : {stats['montant_fraude']/stats['montant_total']*100:.4f}%")
    
    return stats


# =====================================================================
# GRAPHIQUES
# =====================================================================
def graphique_desequilibre(df):
    """Visualisation du déséquilibre"""
    counts = df['Class'].value_counts().sort_index()
    labels = ['Légitime', 'Fraude']
    values = [counts[0], counts[1]]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Distribution (échelle log)', 'Zoom sur les fraudes'),
        specs=[[{'type': 'bar'}, {'type': 'pie'}]]
    )
    
    # Bar chart en log
    fig.add_trace(
        go.Bar(
            x=labels, y=values,
            marker_color=[COLORS['legit'], COLORS['fraude']],
            text=[f'{v:,}' for v in values],
            textposition='outside',
            showlegend=False
        ),
        row=1, col=1
    )
    fig.update_yaxes(type='log', title='Nombre (log)', row=1, col=1)
    
    # Pie chart
    fig.add_trace(
        go.Pie(
            labels=labels, values=values,
            marker_colors=[COLORS['legit'], COLORS['fraude']],
            hole=0.5,
            textinfo='percent+label',
            showlegend=False
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title_text="⚖️ Déséquilibre des classes — 1 fraude pour 578 transactions",
        height=450,
        template='plotly_white'
    )
    return fig


def graphique_fraudes_par_heure(df):
    """Distribution des fraudes par heure de la journée"""
    df_hour = df.copy()
    df_hour['heure'] = df_hour['hour'].astype(int)
    
    data = df_hour.groupby(['heure', 'Class']).size().reset_index(name='count')
    
    fig = px.bar(
        data, x='heure', y='count', color='Class',
        barmode='group',
        color_discrete_map=COLOR_MAP,
        title="🕐 Distribution des transactions par heure de la journée",
        labels={'heure': 'Heure', 'count': 'Nombre', 'Class': 'Type'}
    )
    fig.update_yaxes(type='log')
    fig.update_layout(
        template='plotly_white',
        height=450,
        legend_title="Type"
    )
    return fig


def graphique_taux_fraude_par_heure(df):
    """Taux de fraude par heure — plus parlant"""
    df_hour = df.copy()
    df_hour['heure'] = df_hour['hour'].astype(int)
    
    data = df_hour.groupby('heure')['Class'].agg(['count', 'sum', 'mean']).reset_index()
    data['taux_fraude_pct'] = (data['mean'] * 100).round(3)
    data = data.rename(columns={'count': 'nb_transactions', 'sum': 'nb_fraudes'})
    
    fig = px.bar(
        data, x='heure', y='taux_fraude_pct',
        color='taux_fraude_pct',
        color_continuous_scale='Reds',
        text='taux_fraude_pct',
        title="⚠️ Taux de fraude par heure (le vrai signal !)",
        labels={'heure': 'Heure', 'taux_fraude_pct': 'Taux de fraude (%)'}
    )
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig.update_layout(
        template='plotly_white',
        height=450,
        coloraxis_showscale=False
    )
    return fig, data


def graphique_montants(df):
    """Distribution des montants"""
    df_amount = df[df['Amount'] > 0].copy()
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Distribution des montants (log)', 'Boxplot par classe')
    )
    
    # Histogramme
    for classe, couleur in [(0, COLORS['legit']), (1, COLORS['fraude'])]:
        subset = df_amount[df_amount['Class'] == classe]
        fig.add_trace(
            go.Histogram(
                x=subset['Amount'],
                name=f"{'Légitime' if classe==0 else 'Fraude'}",
                marker_color=couleur,
                opacity=0.7,
                nbinsx=50
            ),
            row=1, col=1
        )
    fig.update_xaxes(type='log', title='Montant (€, log)', row=1, col=1)
    fig.update_yaxes(type='log', title='Fréquence (log)', row=1, col=1)
    
    # Boxplot
    fig.add_trace(
        go.Box(
            x=df_amount['Class'],
            y=df_amount['Amount'],
            marker_color=COLORS['fraude'],
            boxpoints=False,
            name='Montant'
        ),
        row=1, col=2
    )
    fig.update_xaxes(tickvals=[0, 1], ticktext=['Légitime', 'Fraude'], row=1, col=2)
    fig.update_yaxes(type='log', title='Montant (€, log)', row=1, col=2)
    
    fig.update_layout(
        title_text="💰 Distribution des montants selon le type de transaction",
        height=450,
        template='plotly_white',
        showlegend=False
    )
    return fig


def graphique_top_features_discriminantes(df, top_n=15):
    """Identifie les features V1-V28 les plus discriminantes"""
    v_cols = [c for c in df.columns if c.startswith('V') and c[1:].isdigit()]
    
    # Calcul de la différence de moyennes normalisée (Cohen's d)
    stats_legit = df[df['Class'] == 0][v_cols].mean()
    stats_fraude = df[df['Class'] == 1][v_cols].mean()
    
    pooled_std = np.sqrt(
        (df[df['Class'] == 0][v_cols].var() + df[df['Class'] == 1][v_cols].var()) / 2
    )
    
    cohens_d = ((stats_fraude - stats_legit) / pooled_std).abs()
    cohens_d = cohens_d.sort_values(ascending=False).head(top_n)
    
    fig = px.bar(
        x=cohens_d.values,
        y=cohens_d.index,
        orientation='h',
        color=cohens_d.values,
        color_continuous_scale='Reds',
        title=f"🎯 Top {top_n} features les plus discriminantes (Cohen's d)",
        labels={'x': "Cohen's d (écart normalisé)", 'y': ''}
    )
    fig.update_layout(
        template='plotly_white',
        height=500,
        coloraxis_showscale=False,
        yaxis={'categoryorder': 'total ascending'}
    )
    return fig, cohens_d


def graphique_moments_journee(df):
    """Taux de fraude par moment de la journée"""
    data = df.groupby('moment_journee', observed=True)['Class'].agg(
        ['count', 'sum', 'mean']
    ).reset_index()
    data['taux_fraude_pct'] = (data['mean'] * 100).round(3)
    data = data.rename(columns={'count': 'nb_transactions', 'sum': 'nb_fraudes'})
    
    fig = px.bar(
        data, x='moment_journee', y='taux_fraude_pct',
        color='taux_fraude_pct',
        color_continuous_scale='OrRd',
        text='taux_fraude_pct',
        title="🌙 Taux de fraude par moment de la journée",
        labels={'moment_journee': '', 'taux_fraude_pct': 'Taux de fraude (%)'}
    )
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig.update_layout(
        template='plotly_white',
        height=400,
        coloraxis_showscale=False
    )
    return fig, data


def graphique_correlation_target(df, top_n=20):
    """Corrélation des features numériques avec la target"""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Exclure Class lui-même
    if 'Class' in num_cols:
        num_cols.remove('Class')
    
    corr = df[num_cols + ['Class']].corr()[['Class']].drop('Class')
    corr = corr.reindex(corr['Class'].abs().sort_values(ascending=False).index)
    corr_top = corr.head(top_n)
    
    fig = px.bar(
        x=corr_top['Class'].values,
        y=corr_top.index,
        orientation='h',
        color=corr_top['Class'].values,
        color_continuous_scale='RdBu_r',
        title=f"📊 Top {top_n} corrélations avec la fraude",
        labels={'x': 'Corrélation avec la fraude', 'y': ''}
    )
    fig.update_layout(
        template='plotly_white',
        height=550,
        coloraxis_showscale=False,
        yaxis={'categoryorder': 'total ascending'}
    )
    return fig, corr


def graphique_tranches_montant(df):
    """Taux de fraude par tranche de montant"""
    data = df.groupby('tranche_montant', observed=True)['Class'].agg(
        ['count', 'sum', 'mean']
    ).reset_index()
    data['taux_fraude_pct'] = (data['mean'] * 100).round(4)
    data = data.rename(columns={'count': 'nb_transactions', 'sum': 'nb_fraudes'})
    
    fig = px.bar(
        data, x='tranche_montant', y='taux_fraude_pct',
        color='taux_fraude_pct',
        color_continuous_scale='Reds',
        text='taux_fraude_pct',
        title="💸 Taux de fraude par tranche de montant",
        labels={'tranche_montant': 'Tranche de montant', 
                'taux_fraude_pct': 'Taux de fraude (%)'}
    )
    fig.update_traces(texttemplate='%{text:.3f}%', textposition='outside')
    fig.update_layout(
        template='plotly_white',
        height=400,
        coloraxis_showscale=False
    )
    return fig, data


# =====================================================================
# RAPPORT COMPLET
# =====================================================================
def generer_rapport_eda(df, output_dir='reports/eda'):
    """Génère tous les graphiques et les sauvegarde en HTML"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 75)
    print("📈 GÉNÉRATION DES GRAPHIQUES EDA")
    print("=" * 75)
    
    graphiques = {}
    
    # 1. Déséquilibre
    fig = graphique_desequilibre(df)
    graphiques['01_desequilibre'] = fig
    print(f"\n✅ 01_desequilibre")
    
    # 2. Distribution par heure
    fig = graphique_fraudes_par_heure(df)
    graphiques['02_distribution_horaire'] = fig
    print(f"✅ 02_distribution_horaire")
    
    # 3. Taux fraude par heure
    fig, data = graphique_taux_fraude_par_heure(df)
    graphiques['03_taux_fraude_par_heure'] = fig
    print(f"✅ 03_taux_fraude_par_heure")
    print(f"\n   🕐 Taux de fraude par heure (top 5 heures les plus risquées) :")
    top_heures = data.nlargest(5, 'taux_fraude_pct')[
        ['heure', 'nb_transactions', 'nb_fraudes', 'taux_fraude_pct']
    ]
    print(top_heures.to_string(index=False))
    
    # 4. Montants
    fig = graphique_montants(df)
    graphiques['04_montants'] = fig
    print(f"\n✅ 04_montants")
    
    # 5. Top features discriminantes
    fig, cohens_d = graphique_top_features_discriminantes(df)
    graphiques['05_top_features'] = fig
    print(f"\n✅ 05_top_features")
    print(f"\n   🎯 Top 10 features discriminantes (Cohen's d) :")
    print(cohens_d.head(10).round(3).to_string())
    
    # 6. Moments journée
    fig, data = graphique_moments_journee(df)
    graphiques['06_moments_journee'] = fig
    print(f"\n✅ 06_moments_journee")
    print(f"\n   🌙 Taux de fraude par moment de la journée :")
    print(data[['moment_journee', 'nb_transactions', 'nb_fraudes', 
                'taux_fraude_pct']].to_string(index=False))
    
    # 7. Corrélations
    fig, corr = graphique_correlation_target(df)
    graphiques['07_correlations'] = fig
    print(f"\n✅ 07_correlations")
    print(f"\n   📊 Top 10 corrélations avec la fraude :")
    print(corr.head(10).round(4).to_string())
    
    # 8. Tranches montant
    fig, data = graphique_tranches_montant(df)
    graphiques['08_tranches_montant'] = fig
    print(f"\n✅ 08_tranches_montant")
    print(f"\n   💸 Taux de fraude par tranche de montant :")
    print(data[['tranche_montant', 'nb_transactions', 'nb_fraudes', 
                'taux_fraude_pct']].to_string(index=False))
    
    # Sauvegarde HTML
    print(f"\n💾 Sauvegarde des graphiques dans {output_dir}/")
    for nom, fig in graphiques.items():
        chemin = f"{output_dir}/{nom}.html"
        fig.write_html(chemin)
        print(f"   ✅ {nom}.html")
    
    return graphiques


# =====================================================================
# SCRIPT PRINCIPAL
# =====================================================================
if __name__ == '__main__':
    print("🔄 Chargement des données nettoyées...")
    df = pd.read_csv('data/processed/transactions_clean.csv')
    
    # Analyse globale
    stats = analyse_globale(df)
    
    # Graphiques
    graphiques = generer_rapport_eda(df)
    
    print("\n" + "=" * 75)
    print("✅ EDA TERMINÉE")
    print("=" * 75)
    print(f"\n📁 {len(graphiques)} graphiques générés dans reports/eda/")
    print("👉 Ouvre-les dans ton navigateur pour les visualiser")