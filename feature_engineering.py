"""
Engineer features from transactions and network patterns.
"""

import pandas as pd
import numpy as np


def _drop_if_exists(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    return df.drop(columns=[c for c in cols if c in df.columns])


def engineer_features(
    df_persons: pd.DataFrame,
    df_companies: pd.DataFrame,
    df_all_transactions: pd.DataFrame,
    df_projects: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    print("=" * 60)
    print("  Engineering Features")
    print("=" * 60)

    person_id_set   = set(df_persons['person_id'])
    company_id_set  = set(df_companies['company_id'])
    official_id_set = set(df_persons[df_persons['is_official'] == 1]['person_id'])

    # Pre-slice transaksi
    trx_to_person    = df_all_transactions[df_all_transactions['target_id'].isin(person_id_set)]
    trx_from_person  = df_all_transactions[df_all_transactions['source_id'].isin(person_id_set)]
    trx_to_company   = df_all_transactions[df_all_transactions['target_id'].isin(company_id_set)]
    trx_from_company = df_all_transactions[df_all_transactions['source_id'].isin(company_id_set)]

    # Transaction patterns for people
    print("\n[1/8]  Transaction patterns...")
    df_persons = _drop_if_exists(df_persons, [
        'total_inflow', 'inflow_count', 'avg_inflow', 'max_inflow',
        'total_outflow', 'outflow_count', 'avg_outflow', 'max_outflow'])

    _inflow = (trx_to_person.groupby('target_id')['amount']
               .agg(total_inflow='sum', inflow_count='count', avg_inflow='mean', max_inflow='max')
               .rename_axis('person_id').reset_index())
    _outflow = (trx_from_person.groupby('source_id')['amount']
                .agg(total_outflow='sum', outflow_count='count', avg_outflow='mean', max_outflow='max')
                .rename_axis('person_id').reset_index())

    df_persons = df_persons.merge(_inflow, on='person_id', how='left').merge(_outflow, on='person_id', how='left')
    _flow_cols = ['total_inflow', 'inflow_count', 'avg_inflow', 'max_inflow',
                  'total_outflow', 'outflow_count', 'avg_outflow', 'max_outflow']
    # Network signals
    print("[2/8]  Network connections...")
    df_persons = _drop_if_exists(df_persons, ['unique_senders', 'unique_receivers', 'trx_degree'])

    _senders   = (trx_to_person.groupby('target_id')['source_id'].nunique()
                  .rename('unique_senders').rename_axis('person_id').reset_index())
    _receivers = (trx_from_person.groupby('source_id')['target_id'].nunique()
                  .rename('unique_receivers').rename_axis('person_id').reset_index())

    df_persons = df_persons.merge(_senders, on='person_id', how='left').merge(_receivers, on='person_id', how='left')
    df_persons[['unique_senders', 'unique_receivers']] = (
        df_persons[['unique_senders', 'unique_receivers']].fillna(0).astype(int))
    df_persons['trx_degree'] = df_persons['unique_senders'] + df_persons['unique_receivers']

    # Centrality measures
    print("[3/8]  Computing centrality...")
    df_persons = _drop_if_exists(df_persons, ['degree_centrality', 'betweenness_proxy'])

    _max_deg = df_persons['trx_degree'].max()
    df_persons['degree_centrality'] = (df_persons['trx_degree'] / max(_max_deg, 1)).round(4)
    df_persons['betweenness_proxy'] = (
        (df_persons['unique_senders'] * df_persons['unique_receivers']) /
        df_persons['trx_degree'].clip(1).pow(2)
    ).round(4)

    # Risky connections
    print("[4/8]  Finding risky entity contacts...")
    df_persons = _drop_if_exists(df_persons, [
        'shell_proximity', 'shell_contact_count', 'risky_node_distance',
        'unknown_entity_trx_count'])

    _risky_co_ids = set(df_companies[
        (df_companies['compliance_score'] < 50) |
        (df_companies['kyc_status'].isin(['rejected', 'expired'])) |
        (df_companies['is_blacklisted'] == 1)
    ]['company_id'])

    _risky_trx = df_all_transactions[
        (df_all_transactions['source_id'].isin(_risky_co_ids) & df_all_transactions['target_id'].isin(person_id_set)) |
        (df_all_transactions['target_id'].isin(_risky_co_ids) & df_all_transactions['source_id'].isin(person_id_set))
    ]
    _risky_cnt = pd.concat([
        _risky_trx[_risky_trx['target_id'].isin(person_id_set)].rename(columns={'target_id': 'person_id'})[['person_id']],
        _risky_trx[_risky_trx['source_id'].isin(person_id_set)].rename(columns={'source_id': 'person_id'})[['person_id']],
    ]).groupby('person_id').size().rename('risky_entity_contact_count').reset_index()

    df_persons = df_persons.merge(_risky_cnt, on='person_id', how='left')
    df_persons['risky_entity_contact_count'] = df_persons['risky_entity_contact_count'].fillna(0).astype(int)
    df_persons['high_risk_entity_flag']      = (df_persons['risky_entity_contact_count'] > 0).astype(int)

    # Cash flow ratio
    print("[5/8]  Inflow/outflow analysis...")
    df_persons = _drop_if_exists(df_persons, ['inflow_outflow_ratio'])
    df_persons['inflow_outflow_ratio'] = (
        df_persons['total_inflow'] / df_persons['total_outflow'].clip(1)
    ).round(3)

    # Wealth anomalies
    print("[6/8]  Income anomalies and wealth changes...")
    df_persons = _drop_if_exists(df_persons, [
        'actual_monthly_inflow', 'income_anomaly_score', 'wealth_growth_rate'])

    df_persons['actual_monthly_inflow'] = (df_persons['total_inflow'] / 12).round(-4)
    df_persons['income_anomaly_score']  = (
        (df_persons['actual_monthly_inflow'] - df_persons['monthly_income']) /
        df_persons['monthly_income'].clip(1)
    ).clip(-1, 10).round(3)
    df_persons['wealth_growth_rate'] = (
        df_persons['income_anomaly_score'].clip(0, 1)    * 0.40 +
        df_persons['asset_to_income_ratio'].clip(0, 50) * 0.01
    ).clip(0).round(4)

    # Company project activity
    print("\n[7/8]  Project involvement...")
    df_companies = _drop_if_exists(df_companies, [
        'govt_project_win_count', 'total_project_value', 'avg_project_value',
        'corrupt_project_count', 'govt_revenue_ratio'])

    _proj_agg = (df_projects.groupby('winner_company_id').agg(
        govt_project_win_count=('project_id', 'count'),
        total_project_value   =('budget', 'sum'),
        avg_project_value     =('budget', 'mean'),
        corrupt_project_count =('is_corrupt', 'sum'),
    ).rename_axis('company_id').reset_index())

    df_companies = df_companies.merge(_proj_agg, on='company_id', how='left')
    for c in ['govt_project_win_count', 'total_project_value', 'avg_project_value', 'corrupt_project_count']:
        df_companies[c] = df_companies[c].fillna(0)
    df_companies['govt_revenue_ratio'] = (
        df_companies['total_project_value'] / df_companies['annual_revenue'].clip(1)
    ).clip(0, 1).round(3)

    # Company transactions
    print("[8/8]  Transactions and network ties...")
    df_companies = _drop_if_exists(df_companies, [
        'total_inflow', 'inflow_count', 'total_outflow', 'outflow_count'])

    _co_in  = (trx_to_company.groupby('target_id')['amount']
               .agg(total_inflow='sum', inflow_count='count')
               .rename_axis('company_id').reset_index())
    _co_out = (trx_from_company.groupby('source_id')['amount']
               .agg(total_outflow='sum', outflow_count='count')
               .rename_axis('company_id').reset_index())

    df_companies = df_companies.merge(_co_in, on='company_id', how='left').merge(_co_out, on='company_id', how='left')
    df_companies[['total_inflow', 'inflow_count', 'total_outflow', 'outflow_count']] = \
        df_companies[['total_inflow', 'inflow_count', 'total_outflow', 'outflow_count']].fillna(0)

    # Network relations
    df_companies = _drop_if_exists(df_companies, [
        'supplier_count', 'client_count', 'govt_official_links', 'related_companies_count'])

    _suppliers = (trx_to_company.groupby('target_id')['source_id'].nunique()
                  .rename('supplier_count').rename_axis('company_id').reset_index())
    _clients   = (trx_from_company.groupby('source_id')['target_id'].nunique()
                  .rename('client_count').rename_axis('company_id').reset_index())
    df_companies = df_companies.merge(_suppliers, on='company_id', how='left').merge(_clients, on='company_id', how='left')
    df_companies[['supplier_count', 'client_count']] = \
        df_companies[['supplier_count', 'client_count']].fillna(0).astype(int)

    _gov_links = (df_all_transactions
                  .query("source_id in @official_id_set and target_id in @company_id_set")
                  .groupby('target_id')['source_id'].nunique()
                  .rename('govt_official_links').rename_axis('company_id').reset_index())
    df_companies = df_companies.merge(_gov_links, on='company_id', how='left')
    df_companies['govt_official_links'] = df_companies['govt_official_links'].fillna(0).astype(int)

    _ubo_share = (df_companies.groupby('ubo_id')['company_id'].count()
                  .rename('ubo_portfolio').reset_index())
    df_companies = df_companies.merge(_ubo_share, on='ubo_id', how='left')
    df_companies['related_companies_count'] = (df_companies['ubo_portfolio'] - 1).clip(0).astype(int)

    print("\n✅ Feature engineering complete!")
    print(f"  Persons  : {df_persons.shape}")
    print(f"  Companies: {df_companies.shape}")

    return df_persons, df_companies
