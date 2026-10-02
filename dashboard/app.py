"""
Dashboard de supervision — Détection de fraude bancaire
Complément visuel de l'API REST
"""
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import joblib
import sqlite3
from pathlib import Path


# =====================================================================
# CHARGEMENT DES DONNÉES ET DU MODÈLE
# =====================================================================
print("🔄 Chargement...")

# Charger le dataset nettoyé
df = pd.read_csv('data/processed/transactions_clean.csv')

# Charger le modèle
model_data = joblib.load('models/fraud_model.pkl')
model = model_data['model']
scaler = model_data['scaler']
features = model_data['features']

# Prédire sur toutes les transactions (pour la démo)
X = df[features]
X_scaled = scaler.transform(X)
probas = model.predict_proba(X_scaled)[:, 1]

df['probabilite_fraude'] = probas
df['score_risque'] = (probas * 1000).round(0).astype(int)

# Décision
def decision(proba):
    if proba >= 0.7: return 'BLOQUER'
    if proba >= 0.3: return 'VÉRIFIER'
    if proba >= 0.1: return 'SURVEILLER'
    return 'VALIDER'

df['decision'] = df['probabilite_fraude'].apply(decision)

print(f"✅ {len(df):,} transactions scorées")
print(f"   Fraudes réelles : {df['Class'].sum()}")


# =====================================================================
# PALETTE
# =====================================================================
COLORS = {
    'primaire': '#1f3a5f',
    'accent': '#e63946',
    'success': '#2a9d8f',
    'warning': '#f4a261',
    'danger': '#9d0208',
    'light': '#f5f7fa',
}

DECISION_COLORS = {
    'VALIDER': COLORS['success'],
    'SURVEILLER': COLORS['warning'],
    'VÉRIFIER': '#ff8c42',
    'BLOQUER': COLORS['accent'],
}


# =====================================================================
# KPI CARD
# =====================================================================
def kpi_card(titre, valeur, sous_titre="", couleur=None):
    couleur = couleur or COLORS['primaire']
    return html.Div([
        html.P(titre, style={
            'margin': 0, 'color': '#7a8899', 'fontSize': '12px',
            'fontWeight': '600', 'letterSpacing': '0.5px',
            'textTransform': 'uppercase'
        }),
        html.H2(valeur, style={
            'margin': '8px 0 4px 0', 'color': couleur,
            'fontSize': '26px', 'fontWeight': '700'
        }),
        html.P(sous_titre, style={
            'margin': 0, 'color': '#a0aec0', 'fontSize': '12px'
        }),
    ], style={
        'background': 'white',
        'padding': '18px 20px',
        'borderRadius': '12px',
        'boxShadow': '0 2px 12px rgba(0,0,0,0.06)',
        'borderLeft': f'4px solid {couleur}',
        'flex': 1,
        'minWidth': '200px'
    })


# =====================================================================
# APPLICATION
# =====================================================================
app = dash.Dash(__name__, title="Détection Fraude — Supervision")
server = app.server

app.layout = html.Div([

    # HEADER
    html.Div([
        html.Div([
            html.H1("🚨 Détection de Fraude Bancaire",
                    style={'margin': 0, 'color': 'white',
                           'fontSize': '26px', 'fontWeight': '700'}),
            html.P("Dashboard de supervision temps réel",
                   style={'margin': '5px 0 0 0', 'color': '#b8c5d6',
                          'fontSize': '14px'}),
        ], style={'flex': 1}),
        html.Div([
            html.P(f"📊 {len(df):,} transactions analysées",
                   style={'color': 'white', 'margin': 0,
                          'fontSize': '14px', 'fontWeight': '600'}),
            html.P(f"🚨 {df['Class'].sum()} fraudes réelles",
                   style={'color': '#ff6b6b', 'margin': '4px 0 0 0',
                          'fontSize': '13px'}),
        ], style={'textAlign': 'right'})
    ], style={
        'background': f'linear-gradient(90deg, {COLORS["primaire"]}, #8b0000)',
        'padding': '25px 40px',
        'display': 'flex',
        'alignItems': 'center'
    }),

    # KPIs
    html.Div(id='kpi-container', style={
        'display': 'flex', 'gap': '18px',
        'padding': '25px 40px', 'flexWrap': 'wrap'
    }),

    # FILTRES
    html.Div([
        html.Div([
            html.Label("🚦 Décision",
                       style={'fontWeight': '600', 'fontSize': '13px',
                              'color': '#4a5568', 'marginBottom': '6px',
                              'display': 'block'}),
            dcc.Dropdown(
                id='filtre-decision',
                options=[{'label': 'Toutes', 'value': 'ALL'}] +
                        [{'label': d, 'value': d}
                         for d in ['VALIDER', 'SURVEILLER', 'VÉRIFIER', 'BLOQUER']],
                value='ALL', clearable=False,
                style={'fontSize': '14px'}
            )
        ], style={'flex': 1, 'minWidth': '200px'}),

        html.Div([
            html.Label("🎯 Type",
                       style={'fontWeight': '600', 'fontSize': '13px',
                              'color': '#4a5568', 'marginBottom': '6px',
                              'display': 'block'}),
            dcc.Dropdown(
                id='filtre-type',
                options=[
                    {'label': 'Toutes', 'value': 'ALL'},
                    {'label': 'Légitimes', 'value': 0},
                    {'label': 'Fraudes réelles', 'value': 1},
                ],
                value='ALL', clearable=False,
                style={'fontSize': '14px'}
            )
        ], style={'flex': 1, 'minWidth': '200px'}),

        html.Div([
            html.Label("📉 Seuil de probabilité minimum",
                       style={'fontWeight': '600', 'fontSize': '13px',
                              'color': '#4a5568', 'marginBottom': '6px',
                              'display': 'block'}),
            dcc.Slider(
                id='seuil-slider',
                min=0, max=1, step=0.05, value=0,
                marks={0: '0', 0.25: '0.25', 0.5: '0.5',
                       0.75: '0.75', 1: '1'},
                tooltip={'placement': 'bottom'}
            )
        ], style={'flex': 2, 'minWidth': '300px'}),
    ], style={
        'display': 'flex', 'gap': '20px',
        'padding': '0 40px 25px 40px', 'flexWrap': 'wrap'
    }),

    # LIGNE 1 : Distribution décisions + Analyse horaire
    html.Div([
        html.Div(dcc.Graph(id='graph-decisions'),
                 style={'flex': 1, 'minWidth': '450px'}),
        html.Div(dcc.Graph(id='graph-heure'),
                 style={'flex': 1, 'minWidth': '450px'}),
    ], style={
        'display': 'flex', 'gap': '20px',
        'padding': '0 40px 20px 40px', 'flexWrap': 'wrap'
    }),

    # LIGNE 2 : Distribution probabilités + Montants
    html.Div([
        html.Div(dcc.Graph(id='graph-probas'),
                 style={'flex': 1, 'minWidth': '450px'}),
        html.Div(dcc.Graph(id='graph-montants'),
                 style={'flex': 1, 'minWidth': '450px'}),
    ], style={
        'display': 'flex', 'gap': '20px',
        'padding': '0 40px 20px 40px', 'flexWrap': 'wrap'
    }),

    # TABLEAU ALERTES
    html.Div([
        html.H3("🚨 Top 20 des transactions suspectes",
                style={'color': COLORS['primaire'],
                       'padding': '0 40px', 'margin': '20px 0 10px 0'}),
        html.Div(id='tableau-alertes',
                 style={'padding': '0 40px 40px 40px'})
    ]),

    # FOOTER
    html.Div([
        html.P("📌 Projet Détection de Fraude — Modèle XGBoost (PR-AUC 0,76)",
               style={'color': '#718096', 'fontSize': '12px',
                      'textAlign': 'center', 'margin': 0}),
    ], style={'padding': '20px', 'background': '#e2e8f0'})

], style={
    'fontFamily': "'Segoe UI', Arial, sans-serif",
    'background': COLORS['light'],
    'minHeight': '100vh',
    'margin': 0
})


# =====================================================================
# CALLBACK
# =====================================================================
@app.callback(
    [Output('kpi-container', 'children'),
     Output('graph-decisions', 'figure'),
     Output('graph-heure', 'figure'),
     Output('graph-probas', 'figure'),
     Output('graph-montants', 'figure'),
     Output('tableau-alertes', 'children')],
    [Input('filtre-decision', 'value'),
     Input('filtre-type', 'value'),
     Input('seuil-slider', 'value')]
)
def update_dashboard(decision_filter, type_filter, seuil):
    # Filtrage
    dff = df.copy()
    if decision_filter != 'ALL':
        dff = dff[dff['decision'] == decision_filter]
    if type_filter != 'ALL':
        dff = dff[dff['Class'] == type_filter]
    if seuil > 0:
        dff = dff[dff['probabilite_fraude'] >= seuil]
    
    # KPIs
    total = len(dff)
    bloquees = len(dff[dff['decision'] == 'BLOQUER'])
    verif = len(dff[dff['decision'] == 'VÉRIFIER'])
    fraudes_reelles = int(dff['Class'].sum()) if type_filter == 'ALL' else total
    
    kpis = [
        kpi_card("📊 Transactions", f"{total:,}", 
                 "analysées", COLORS['primaire']),
        kpi_card("🚨 Bloquées", f"{bloquees:,}",
                 f"{bloquees/total*100:.3f}%" if total else "0%",
                 COLORS['accent']),
        kpi_card("⚠️ À vérifier", f"{verif:,}",
                 f"{verif/total*100:.2f}%" if total else "0%",
                 COLORS['warning']),
        kpi_card("✅ Fraudes détectées", f"{fraudes_reelles:,}",
                 "réelles" if type_filter == 'ALL' else "filtrées",
                 COLORS['success']),
    ]
    
    # Graph 1 : Distribution décisions
    dec_counts = dff['decision'].value_counts().reset_index()
    dec_counts.columns = ['decision', 'count']
    dec_counts['couleur'] = dec_counts['decision'].map(DECISION_COLORS)
    
    fig1 = go.Figure(data=[go.Pie(
        labels=dec_counts['decision'],
        values=dec_counts['count'],
        marker_colors=dec_counts['couleur'],
        hole=0.5,
        textinfo='label+percent',
    )])
    fig1.update_layout(
        title="🚦 Distribution des décisions",
        template='plotly_white', height=400
    )
    
    # Graph 2 : Analyse par heure
    dff_h = dff.copy()
    dff_h['heure'] = dff_h['hour'].astype(int)
    data_h = dff_h.groupby('heure').agg(
        nb=('Class', 'count'),
        nb_fraudes=('Class', 'sum')
    ).reset_index()
    data_h['taux'] = (data_h['nb_fraudes'] / data_h['nb'] * 100).round(4)
    
    fig2 = px.bar(
        data_h, x='heure', y='nb',
        color='taux',
        color_continuous_scale='Reds',
        title="🕐 Volume de transactions par heure",
        labels={'heure': 'Heure', 'nb': 'Nb transactions', 'taux': 'Taux fraude (%)'}
    )
    fig2.update_layout(template='plotly_white', height=400)
    
    # Graph 3 : Distribution des probabilités
    fig3 = go.Figure()
    for classe, couleur, nom in [(0, COLORS['success'], 'Légitime'),
                                  (1, COLORS['accent'], 'Fraude')]:
        subset = dff[dff['Class'] == classe]
        fig3.add_trace(go.Histogram(
            x=subset['probabilite_fraude'],
            name=nom,
            marker_color=couleur,
            opacity=0.7,
            nbinsx=50
        ))
    fig3.update_layout(
        title="📊 Distribution des probabilités de fraude",
        xaxis_title="Probabilité",
        yaxis_title="Nombre (log)",
        yaxis_type='log',
        barmode='overlay',
        template='plotly_white',
        height=400
    )
    
    # Graph 4 : Montants
    fig4 = px.scatter(
        dff.sample(min(2000, len(dff))),
        x='Amount', y='probabilite_fraude',
        color='Class',
        color_discrete_map={0: COLORS['success'], 1: COLORS['accent']},
        title="💰 Montant vs Probabilité de fraude",
        labels={'Amount': 'Montant (€)', 'probabilite_fraude': 'Probabilité',
                'Class': 'Type'},
        log_x=True,
        opacity=0.6
    )
    fig4.update_layout(template='plotly_white', height=400)
    
    # Tableau alertes
    alertes = dff.nlargest(20, 'probabilite_fraude')[
        ['hour', 'Amount', 'probabilite_fraude', 'decision', 'Class']
    ].copy()
    alertes['hour'] = alertes['hour'].round(1)
    alertes['Amount'] = alertes['Amount'].round(2)
    alertes['probabilite_fraude'] = (alertes['probabilite_fraude'] * 100).round(2)
    alertes['Class'] = alertes['Class'].map({0: '✅ Légitime', 1: '🚨 Fraude'})
    alertes.columns = ['Heure', 'Montant (€)', 'Proba (%)', 'Décision', 'Réalité']
    
    tableau = dash_table.DataTable(
        data=alertes.to_dict('records'),
        columns=[{'name': c, 'id': c} for c in alertes.columns],
        style_cell={
            'textAlign': 'left', 'padding': '10px',
            'fontFamily': "'Segoe UI', Arial, sans-serif",
            'fontSize': '13px'
        },
        style_header={
            'backgroundColor': COLORS['primaire'],
            'color': 'white', 'fontWeight': 'bold',
            'fontSize': '12px', 'textTransform': 'uppercase'
        },
        style_data_conditional=[
            {'if': {'row_index': 'odd'}, 'backgroundColor': '#f8f9fa'},
            {'if': {'filter_query': '{Décision} = "BLOQUER"'},
             'backgroundColor': '#ffe5e5', 'fontWeight': 'bold'},
        ],
        style_as_list_view=True,
    )
    
    return kpis, fig1, fig2, fig3, fig4, tableau


# =====================================================================
# LANCEMENT
# =====================================================================
if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("🚀 Dashboard Fraude démarré")
    print("=" * 70)
    print("👉 Ouvre ton navigateur : http://127.0.0.1:8051")
    print("⛔ Pour arrêter : Ctrl + C")
    print("=" * 70 + "\n")
    app.run(debug=True, port=8051)