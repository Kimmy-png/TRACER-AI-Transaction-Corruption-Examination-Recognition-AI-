"""
Train the two main models: project risk classifier and transaction risk scorer.
"""

import warnings
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings('ignore')

# Features to exclude (anti-leakage - we don't want label info in training)
CO_BANNED = [
    'true_company_type',
    'corrupt_project_count',
    'shell_connections',
    'shell_network_flag',
]
OFF_BANNED = [
    'illicit_recv_count',
    'illicit_recv_amount',
    'unusual_transfer_flag',
    'shell_proximity',
    'shell_contact_count',
    'risky_node_distance',
]


def train_ai1(
    df_persons: pd.DataFrame,
    df_companies: pd.DataFrame,
    df_projects: pd.DataFrame,
) -> tuple:
    print("=" * 60)
    print("  Training Project Risk Model (AI-1)")
    print("=" * 60)

    print("\n[1/5] Preparing features...")

    CO_FEATS = [c for c in [
        'company_id',
        'company_type',
        'industry', 'legal_form', 'company_age', 'is_active', 'domicile_province',
        'annual_revenue', 'employee_count', 'asset_value', 'profit_margin', 'company_size_category',
        'director_count', 'commissioner_count', 'has_foreign_ownership',
        'ownership_complexity', 'management_change_count',
        'bank_account_count', 'has_offshore_account', 'has_escrow_account',
        'kyc_status', 'compliance_score', 'is_blacklisted', 'adverse_media_flag', 'risk_tier',
        'govt_project_win_count', 'total_project_value', 'avg_project_value', 'govt_revenue_ratio',
        'supplier_count', 'client_count', 'govt_official_links', 'related_companies_count',
        'total_inflow', 'total_outflow', 'inflow_count', 'outflow_count',
    ] if c in df_companies.columns and c not in CO_BANNED]

    co_clean = (df_companies[CO_FEATS].copy()
                .rename(columns={c: f'co_{c}' for c in CO_FEATS if c != 'company_id'}))

    OFF_FEATS = [c for c in [
        'person_id',
        'age', 'education_level', 'years_of_service', 'job_grade',
        'monthly_income', 'income_percentile', 'has_side_income', 'side_income_amount',
        'spending_to_income_ratio', 'savings_rate', 'has_investment', 'risk_appetite',
        'property_count', 'vehicle_count', 'has_luxury_goods',
        'total_estimated_asset', 'asset_to_income_ratio',
        'bank_account_count', 'has_offshore_account', 'credit_score',
        'degree_centrality', 'betweenness_proxy',
        'risky_entity_contact_count', 'high_risk_entity_flag',
        'inflow_outflow_ratio', 'trx_degree', 'total_inflow', 'total_outflow',
        'lifestyle_index', 'frequent_travel', 'luxury_spending_ratio',
        'income_anomaly_score', 'wealth_growth_rate',
        'procurement_auth_level', 'lhkpn_reported_asset',
        'is_politically_exposed', 'biz_partners_count',
    ] if c in df_persons.columns and c not in OFF_BANNED]

    off_clean = (df_persons[OFF_FEATS].copy()
                 .rename(columns={c: f'off_{c}' for c in OFF_FEATS if c != 'person_id'}))

    # Encode categorical features
    print("[2/5] Processing categorical data...")
    ID_COLS    = ['project_id', 'official_id', 'winner_company_id']
    TARGET_COL = 'is_corrupt'
    cat_cols   = [c for c in X_full.select_dtypes('object').columns if c not in ID_COLS]
    le_store   = {}
    for col in cat_cols:
        le = LabelEncoder()
        X_full[col] = le.fit_transform(X_full[col].astype(str))
        le_store[col] = le

    FEATURE_COLS = [c for c in X_full.columns if c not in ID_COLS + [TARGET_COL]]
    X = X_full[FEATURE_COLS].fillna(0)
    y = X_full[TARGET_COL]

    print(f"  {X.shape[0]} projects × {X.shape[1]} features")
    print(f"  {(y == 1).sum()} corrupt, {(y == 0).sum()} clean")

    # Train the model
    print("[3/5] Training model...")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_tr)

    ai1 = GradientBoostingClassifier(
        n_estimators=350, learning_rate=0.04, max_depth=4,
        min_samples_leaf=4, subsample=0.8, random_state=42)
    ai1.fit(X_tr, y_tr, sample_weight=sample_weights)

    # Evaluate
    print("[4/5] Evaluating...")
    y_prob_te = ai1.predict_proba(X_te)[:, 1]
    y_pred_te = (y_prob_te >= 0.35).astype(int)
    print(f"\n  ROC-AUC: {roc_auc_score(y_te, y_prob_te):.4f}")
    print(classification_report(y_te, y_pred_te, target_names=['clean', 'corrupt']))

    fi = pd.Series(ai1.feature_importances_, index=FEATURE_COLS)
    print("\n  Most important features:")
    print(fi.nlargest(10).to_string())

    # Score all projects
    print("\n[5/5] Scoring all projects...")
    X_all = X_full[FEATURE_COLS].fillna(0)
    X_full['ai1_score']   = ai1.predict_proba(X_all)[:, 1]
    X_full['ai1_flagged'] = (X_full['ai1_score'] >= 0.35).astype(int)

    df_projects_scored = df_projects.merge(
        X_full[['project_id', 'ai1_score', 'ai1_flagged']],
        on='project_id', how='left')

    flagged_ids = df_projects_scored[df_projects_scored['ai1_flagged'] == 1]['project_id'].tolist()

    print(f"\n✅ AI-1 training done!")
    print(f"  Flagged projects: {len(flagged_ids)}")
    print(f"  Actually corrupt: {df_projects_scored['is_corrupt'].sum()}")

    return ai1, df_projects_scored, flagged_ids, FEATURE_COLS, le_store


def train_ai2(
    df_persons: pd.DataFrame,
    df_companies: pd.DataFrame,
    df_all_transactions: pd.DataFrame,
    df_projects: pd.DataFrame,
    flagged_ids: list,
) -> tuple:
    print("=" * 60)
    print("  Training Transaction Risk Model (AI-2)")
    print("=" * 60)

    person_id_set   = set(df_persons['person_id'])
    company_id_set  = set(df_companies['company_id'])
    official_id_set = set(df_persons[df_persons['is_official'] == 1]['person_id'])
    shell_id_set_true = set(df_companies[df_companies['true_company_type'] == 'Shell_Company']['company_id'])

    # Compute risk scores for each node
    print("\n[1/6] Computing node risk scores...")

    _p = df_persons.copy()
    p_risk = (
        _p['income_anomaly_score'].clip(0, 1)    * 0.30 +
        _p['high_risk_entity_flag']              * 0.25 +
        _p['wealth_growth_rate'].clip(0, 1)      * 0.25 +
        _p['betweenness_proxy'].clip(0, 1)       * 0.20
    )
    node_risk_person = pd.Series(p_risk.values, index=_p['person_id']).clip(0, 1)

    _c = df_companies.copy()
    _kyc_enc  = _c['kyc_status'].map({'verified': 0, 'pending': 0.3, 'expired': 0.5, 'rejected': 1.0}).fillna(0.3)
    _tier_enc = _c['risk_tier'].map({'low': 0, 'medium': 0.5, 'high': 1.0}).fillna(0)
    c_risk = (
        (100 - _c['compliance_score'].clip(0, 100)) / 100 * 0.40 +
        _tier_enc    * 0.35 +
        _c['is_blacklisted'] * 0.15 +
        _kyc_enc     * 0.10
    )
    node_risk_company = pd.Series(c_risk.values, index=_c['company_id']).clip(0, 1)
    node_risk_gov     = pd.Series({'GOV_TREASURY': 0.0})
    all_node_risk     = pd.concat([node_risk_person, node_risk_company, node_risk_gov])

    # Prepare training data
    print("[2/6] Building training data...")
    trx_proj = df_all_transactions[df_all_transactions['project_ref'].notna()].copy()
    trx_proj = trx_proj.merge(
        df_projects[['project_id', 'budget']],
        left_on='project_ref', right_on='project_id', how='left')

    TRX_TYPE_MAP = {t: i for i, t in enumerate(sorted(trx_proj['trx_type'].unique()))}
    trx_proj['trx_type_enc']     = trx_proj['trx_type'].map(TRX_TYPE_MAP).fillna(-1).astype(int)
    trx_proj['src_risk']         = trx_proj['source_id'].map(all_node_risk).fillna(0.05)
    trx_proj['tgt_risk']         = trx_proj['target_id'].map(all_node_risk).fillna(0.05)
    trx_proj['src_is_official']  = trx_proj['source_id'].isin(official_id_set).astype(int)
    trx_proj['tgt_is_official']  = trx_proj['target_id'].isin(official_id_set).astype(int)
    trx_proj['src_is_gov']       = (trx_proj['source_id'] == 'GOV_TREASURY').astype(int)
    trx_proj['log_amount']       = np.log1p(trx_proj['amount'])
    trx_proj['amount_to_budget'] = (trx_proj['amount'] / trx_proj['budget'].clip(1)).clip(0, 1)
    trx_proj['risk_product']     = trx_proj['src_risk'] * trx_proj['tgt_risk']
    trx_proj['risk_sum']         = trx_proj['src_risk'] + trx_proj['tgt_risk']

    AI2_FEATURES = [
        'trx_type_enc', 'log_amount', 'amount_to_budget',
        'src_risk', 'tgt_risk', 'risk_product', 'risk_sum',
        'src_is_official', 'tgt_is_official', 'src_is_gov',
        'day',
    ]

    X2 = trx_proj[AI2_FEATURES].fillna(0)
    y2 = trx_proj['is_illicit']

    print(f"  {len(X2):,} edges total, {y2.sum():,} illicit ({y2.mean() * 100:.1f}%)")

    # Train the model
    print("\n[3/6] Training random forest...")
    X2_tr, X2_te, y2_tr, y2_te = train_test_split(X2, y2, test_size=0.20, stratify=y2, random_state=42)

    ai2 = RandomForestClassifier(
        n_estimators=400, max_depth=10,
        min_samples_leaf=2, class_weight='balanced',
        n_jobs=-1, random_state=42)
    ai2.fit(X2_tr, y2_tr)

    y2_prob_te = ai2.predict_proba(X2_te)[:, 1]
    print(f"  ROC-AUC: {roc_auc_score(y2_te, y2_prob_te):.4f}")
    print(classification_report(y2_te, (y2_prob_te >= 0.5).astype(int),
                                 target_names=['clean', 'illicit']))

    # Score edges from flagged projects
    print("\n[4/6] Scoring edges from flagged projects...")
    trx_flagged = trx_proj[trx_proj['project_ref'].isin(flagged_ids)].copy()
    X2_flag = trx_flagged[AI2_FEATURES].fillna(0)
    trx_flagged['ai2_edge_score']   = ai2.predict_proba(X2_flag)[:, 1]
    trx_flagged['ai2_edge_flagged'] = (trx_flagged['ai2_edge_score'] >= 0.50).astype(int)

    print(f"  {len(trx_flagged):,} edges from flagged projects")
    print(f"  {trx_flagged['ai2_edge_flagged'].sum():,} high-risk edges")

    # Helper to classify node types
    def get_node_type(node_id):
        if node_id == 'GOV_TREASURY':         return 'gov'
        if node_id in shell_id_set_true:      return 'shell'
        if node_id in official_id_set:        return 'official'
        if node_id in person_id_set:          return 'person'
        return 'company'

    # Build graph
    print("\n[5/6] Building network graph...")
    G = nx.DiGraph()
    for _, row in trx_flagged.iterrows():
        src, tgt = row['source_id'], row['target_id']
        for nid in [src, tgt]:
            if nid not in G:
                G.add_node(nid,
                           node_type=get_node_type(nid),
                           risk=float(all_node_risk.get(nid, 0.0)))
        if G.has_edge(src, tgt):
            G[src][tgt]['ai2_score'] = max(G[src][tgt]['ai2_score'], row['ai2_edge_score'])
            G[src][tgt]['amount']   += row['amount']
            G[src][tgt]['trx_count'] += 1
        else:
            G.add_edge(src, tgt,
                       ai2_score   = float(row['ai2_edge_score']),
                       trx_type    = row['trx_type'],
                       amount      = float(row['amount']),
                       project_ref = row['project_ref'],
                       is_illicit  = int(row['is_illicit']),
                       trx_count   = 1)

    print(f"\n✅ AI-2 training done!")
    print(f"  Network nodes: {G.number_of_nodes():,}")
    print(f"  Network edges: {G.number_of_edges():,}")

    return ai2, G, trx_flagged, all_node_risk, get_node_type
