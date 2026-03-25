"""
Generate the base entities: people, companies, and org hierarchy.
Returns dataframes ready for the transaction generator.
"""

import pandas as pd
import numpy as np
from faker import Faker
import random

fake = Faker('id_ID')


def generate_entities(
    n_persons: int = 100_000,
    n_companies: int = 4_500,
    n_officials: int = 3_000,
    seed: int = 42,
):
    np.random.seed(seed)
    random.seed(seed)

    print("=" * 60)
    print("  Generating Entities")
    print("=" * 60)

    # Generate person records
    print(f"\n[1/3] Creating {n_persons:,} people...")

    person_ids = [f"P{str(i).zfill(6)}" for i in range(1, n_persons + 1)]

    ages           = np.random.normal(40, 10, n_persons).clip(22, 65).astype(int)
    genders        = np.random.choice(['male', 'female'], n_persons, p=[0.55, 0.45])
    marital_status = np.random.choice(
        ['single', 'married', 'divorced', 'widowed'],
        n_persons, p=[0.28, 0.60, 0.09, 0.03])
    num_dependents = np.where(
        marital_status == 'married',
        np.random.poisson(2, n_persons),
        np.random.choice([0, 1], n_persons, p=[0.85, 0.15]))
    city_zones     = np.random.choice(['north', 'south', 'east', 'west', 'central'], n_persons)
    domicile_types = np.random.choice(['urban', 'suburban', 'rural'], n_persons, p=[0.55, 0.35, 0.10])

    edu_levels     = np.random.choice(
        ['high_school', 'diploma', 'bachelor', 'master', 'phd'],
        n_persons, p=[0.35, 0.10, 0.43, 0.10, 0.02])
    edu_inst_type  = np.random.choice(['public', 'private'], n_persons, p=[0.55, 0.45])
    _grad_age_map  = {'high_school': 18, 'diploma': 20, 'bachelor': 22, 'master': 25, 'phd': 30}
    base_grad_ages = np.array([_grad_age_map[e] for e in edu_levels])
    graduation_year = (2024 - (ages - base_grad_ages)).clip(1985, 2024)

    employment_type = np.random.choice(
        ['private_employee', 'self_employed', 'entrepreneur', 'freelance'],
        n_persons, p=[0.50, 0.20, 0.20, 0.10])
    job_grades = np.random.choice(
        ['junior', 'mid', 'senior', 'manager', 'executive'],
        n_persons, p=[0.30, 0.30, 0.25, 0.12, 0.03])
    years_of_svc  = np.clip(ages - base_grad_ages - np.random.randint(0, 3, n_persons), 0, 40)
    occupation_fld = np.random.choice(
        ['finance', 'engineering', 'trade', 'services', 'agriculture',
         'education', 'health', 'government'],
        n_persons, p=[0.12, 0.18, 0.20, 0.15, 0.10, 0.10, 0.08, 0.07])

    monthly_incomes   = np.random.lognormal(mean=16, sigma=0.8, size=n_persons).round(-5)
    income_percentile = pd.Series(monthly_incomes).rank(pct=True).values.round(4)
    annual_inc_est    = (monthly_incomes * 13).round(-5)
    has_side_income   = np.random.choice([0, 1], n_persons, p=[0.65, 0.35])
    side_income_amount = np.where(
        has_side_income,
        (monthly_incomes * np.random.uniform(0.1, 0.5, n_persons)).round(-4), 0)

    spend_inc_ratio = np.random.uniform(0.40, 0.95, n_persons).round(3)
    savings_rate    = np.clip(1 - spend_inc_ratio - np.random.uniform(0, 0.08, n_persons), 0.01, 0.50).round(3)
    has_investment  = np.random.choice([0, 1], n_persons, p=[0.65, 0.35])
    investment_type = np.where(
        has_investment,
        np.random.choice(['stocks', 'mutual_fund', 'property', 'gold', 'mixed'], n_persons),
        'none')
    risk_appetite = np.random.choice(
        ['conservative', 'moderate', 'aggressive'], n_persons, p=[0.45, 0.40, 0.15])

    property_count   = np.random.choice([0, 1, 2, 3, 4], n_persons, p=[0.30, 0.45, 0.15, 0.07, 0.03])
    vehicle_count    = np.random.choice([0, 1, 2, 3], n_persons, p=[0.25, 0.50, 0.20, 0.05])
    has_luxury_goods = np.random.choice([0, 1], n_persons, p=[0.85, 0.15])
    _prop_val  = np.random.lognormal(19.5, 0.8, n_persons)
    _veh_val   = np.random.lognormal(17.5, 0.6, n_persons)
    _lux_val   = np.where(has_luxury_goods, np.random.lognormal(17, 1, n_persons), 0)
    total_asset_val = (property_count * _prop_val + vehicle_count * _veh_val + _lux_val).round(-6)
    asset_inc_ratio = (total_asset_val / (monthly_incomes * 12).clip(1)).round(2)

    bank_acc_count   = np.random.poisson(1.5, n_persons) + 1
    has_offshore_acc = np.random.choice([0, 1], n_persons, p=[0.97, 0.03])
    primary_bank     = np.random.choice(
        ['BRI', 'BCA', 'Mandiri', 'BNI', 'CIMB', 'Danamon', 'BTN', 'Other'],
        n_persons, p=[0.25, 0.22, 0.20, 0.15, 0.05, 0.04, 0.04, 0.05])
    has_digital_wlt = np.random.choice([0, 1], n_persons, p=[0.30, 0.70])
    credit_scores   = np.random.normal(700, 80, n_persons).clip(300, 850).astype(int)

    monthly_trx_cnt  = np.random.poisson(50, n_persons)
    pref_trx_channel = np.random.choice(
        ['mobile_banking', 'atm', 'internet_banking', 'cash', 'digital_wallet'],
        n_persons, p=[0.35, 0.20, 0.15, 0.15, 0.15])
    large_trx_freq = np.random.choice(
        ['never', 'rarely', 'sometimes', 'often'], n_persons, p=[0.50, 0.30, 0.15, 0.05])

    family_conn      = np.random.poisson(3, n_persons)
    biz_partners_cnt = np.random.poisson(2, n_persons)
    org_membership   = np.random.choice([0, 1], n_persons, p=[0.60, 0.40])

    lifestyle_index = np.clip(
        income_percentile * 0.6 + np.random.uniform(0, 0.4, n_persons), 0, 1).round(3)
    frequent_travel = np.random.choice([0, 1], n_persons, p=[0.75, 0.25])
    luxury_spend_r  = np.where(
        has_luxury_goods,
        np.random.uniform(0.05, 0.30, n_persons),
        np.random.uniform(0.00, 0.05, n_persons)).round(3)
    social_media_act = np.random.choice(['low', 'medium', 'high'], n_persons, p=[0.30, 0.45, 0.25])

    df_persons = pd.DataFrame({
        'person_id'               : person_ids,
        'name'                    : [fake.name() for _ in range(n_persons)],
        'age'                     : ages,
        'gender'                  : genders,
        'marital_status'          : marital_status,
        'num_dependents'          : num_dependents,
        'city_zone'               : city_zones,
        'domicile_type'           : domicile_types,
        'education_level'         : edu_levels,
        'education_institution'   : edu_inst_type,
        'graduation_year'         : graduation_year,
        'employment_type'         : employment_type,
        'job_grade'               : job_grades,
        'years_of_service'        : years_of_svc,
        'occupation_field'        : occupation_fld,
        'monthly_income'          : monthly_incomes,
        'income_percentile'       : income_percentile,
        'annual_income_estimate'  : annual_inc_est,
        'has_side_income'         : has_side_income,
        'side_income_amount'      : side_income_amount,
        'spending_to_income_ratio': spend_inc_ratio,
        'savings_rate'            : savings_rate,
        'has_investment'          : has_investment,
        'investment_type'         : investment_type,
        'risk_appetite'           : risk_appetite,
        'property_count'          : property_count,
        'vehicle_count'           : vehicle_count,
        'has_luxury_goods'        : has_luxury_goods,
        'total_estimated_asset'   : total_asset_val,
        'asset_to_income_ratio'   : asset_inc_ratio,
        'bank_account_count'      : bank_acc_count,
        'has_offshore_account'    : has_offshore_acc,
        'primary_bank'            : primary_bank,
        'has_digital_wallet'      : has_digital_wlt,
        'credit_score'            : credit_scores,
        'monthly_trx_count_seed'  : monthly_trx_cnt,
        'preferred_trx_channel'   : pref_trx_channel,
        'large_trx_frequency'     : large_trx_freq,
        'family_connections'      : family_conn,
        'biz_partners_count'      : biz_partners_cnt,
        'org_membership'          : org_membership,
        'is_politically_exposed'  : np.zeros(n_persons, dtype=int),
        'lifestyle_index'         : lifestyle_index,
        'frequent_travel'         : frequent_travel,
        'luxury_spending_ratio'   : luxury_spend_r,
        'social_media_activity'   : social_media_act,
        'is_official'             : np.zeros(n_persons, dtype=int),
        'official_role'           : 'none',
        'procurement_auth_level'  : np.zeros(n_persons, dtype=int),
        'lhkpn_reported_asset'    : np.zeros(n_persons),
        'total_inflow'            : 0.0,
        'total_outflow'           : 0.0,
        'inflow_count'            : 0,
        'outflow_count'           : 0,
        'avg_inflow'              : 0.0,
        'avg_outflow'             : 0.0,
        'max_inflow'              : 0.0,
        'max_outflow'             : 0.0,
        'unique_senders'          : 0,
        'unique_receivers'        : 0,
        'trx_degree'              : 0,
        'degree_centrality'       : 0.0,
        'betweenness_proxy'       : 0.0,
        'inflow_outflow_ratio'    : 0.0,
        'income_anomaly_score'    : 0.0,
        'wealth_growth_rate'      : 0.0,
        'actual_monthly_inflow'   : 0.0,
    })

    print(f"\n[2/3] Creating {n_companies:,} companies...")

    company_ids = [f"C{str(i).zfill(5)}" for i in range(1, n_companies + 1)]
    _industries = ['construction', 'IT', 'consulting', 'procurement',
                   'logistics', 'general_trading', 'finance', 'property']

    co_types_true = np.random.choice(
        ['HQ', 'Branch', 'Shell_Company', 'Subsidiary'],
        n_companies, p=[0.65, 0.22, 0.05, 0.08])
    co_industry   = np.random.choice(
        _industries, n_companies, p=[0.35, 0.18, 0.10, 0.12, 0.05, 0.10, 0.05, 0.05])
    legal_forms   = np.random.choice(
        ['PT', 'CV', 'Koperasi', 'Yayasan', 'Firma'], n_companies, p=[0.60, 0.25, 0.05, 0.05, 0.05])
    estab_years   = np.random.randint(1990, 2024, n_companies)
    is_active     = np.random.choice([1, 0], n_companies, p=[0.90, 0.10])
    domicile_prov = np.random.choice(
        ['Jawa Timur', 'DKI Jakarta', 'Jawa Barat', 'Jawa Tengah',
         'Bali', 'Sulawesi Selatan', 'Sumatera Utara', 'Other'],
        n_companies, p=[0.25, 0.20, 0.15, 0.12, 0.07, 0.05, 0.05, 0.11])

    annual_rev   = np.random.lognormal(22, 1.5, n_companies).round(-6)
    emp_count    = np.random.lognormal(4, 1.2, n_companies).clip(1, 50_000).astype(int)
    asset_values = (annual_rev * np.random.uniform(0.5, 3.0, n_companies)).round(-6)
    profit_margs = np.random.beta(2, 5, n_companies).round(3)
    co_size_cat  = np.where(emp_count < 10, 'micro',
                   np.where(emp_count < 50, 'small',
                   np.where(emp_count < 250, 'medium', 'large')))
    co_ages = 2024 - estab_years

    ubo_ids    = np.random.choice(person_ids, n_companies)
    dir_count  = np.random.choice([1, 2, 3, 4, 5], n_companies, p=[0.30, 0.35, 0.20, 0.10, 0.05])
    comm_count = np.random.choice([0, 1, 2, 3], n_companies, p=[0.30, 0.35, 0.25, 0.10])
    foreign_own = np.random.choice([0, 1], n_companies, p=[0.85, 0.15])
    own_complex  = np.random.choice(
        ['simple', 'moderate', 'complex', 'opaque'], n_companies, p=[0.50, 0.30, 0.15, 0.05])
    mgmt_changes = np.random.poisson(1.5, n_companies)

    co_bank_acc  = np.random.poisson(3, n_companies) + 1
    co_offshore  = np.random.choice([0, 1], n_companies, p=[0.93, 0.07])
    co_prim_bank = np.random.choice(
        ['BRI', 'BCA', 'Mandiri', 'BNI', 'CIMB', 'Other'],
        n_companies, p=[0.20, 0.25, 0.25, 0.15, 0.08, 0.07])
    co_has_escrow = np.random.choice([0, 1], n_companies, p=[0.75, 0.25])

    kyc_status    = np.random.choice(
        ['verified', 'pending', 'rejected', 'expired'], n_companies, p=[0.80, 0.10, 0.03, 0.07])
    compliance_sc = np.random.normal(70, 15, n_companies).clip(0, 100).round(1)
    is_blacklisted = np.random.choice([0, 1], n_companies, p=[0.97, 0.03])
    adverse_media  = np.random.choice([0, 1], n_companies, p=[0.92, 0.08])

    _kyc_risk_arr = pd.Series(kyc_status).map(
        {'verified': 0, 'pending': 0.3, 'expired': 0.5, 'rejected': 1.0}).fillna(0.3).values
    _risk_score_arr = (
        (100 - compliance_sc) / 100 * 0.50 +
        is_blacklisted * 0.30 +
        _kyc_risk_arr  * 0.20
    )
    risk_tier = np.where(_risk_score_arr > 0.6, 'high',
                np.where(_risk_score_arr > 0.3, 'medium', 'low'))

    # ANTI-LEAKAGE: company_type di-mask
    co_types_masked = co_types_true.copy()
    _shell_idx = co_types_true == 'Shell_Company'
    co_types_masked[_shell_idx] = np.random.choice(
        ['HQ', 'Branch'], size=_shell_idx.sum(), p=[0.6, 0.4])

    df_companies = pd.DataFrame({
        'company_id'              : company_ids,
        'company_name'            : [fake.company() for _ in range(n_companies)],
        'true_company_type'       : co_types_true,
        'company_type'            : co_types_masked,
        'industry'                : co_industry,
        'legal_form'              : legal_forms,
        'establishment_year'      : estab_years,
        'company_age'             : co_ages,
        'is_active'               : is_active,
        'domicile_province'       : domicile_prov,
        'annual_revenue'          : annual_rev,
        'employee_count'          : emp_count,
        'asset_value'             : asset_values,
        'profit_margin'           : profit_margs,
        'company_size_category'   : co_size_cat,
        'ubo_id'                  : ubo_ids,
        'director_count'          : dir_count,
        'commissioner_count'      : comm_count,
        'has_foreign_ownership'   : foreign_own,
        'ownership_complexity'    : own_complex,
        'management_change_count' : mgmt_changes,
        'bank_account_count'      : co_bank_acc,
        'has_offshore_account'    : co_offshore,
        'primary_bank'            : co_prim_bank,
        'has_escrow_account'      : co_has_escrow,
        'kyc_status'              : kyc_status,
        'compliance_score'        : compliance_sc,
        'is_blacklisted'          : is_blacklisted,
        'adverse_media_flag'      : adverse_media,
        'risk_tier'               : risk_tier,
        'govt_project_win_count'  : 0,
        'total_project_value'     : 0.0,
        'avg_project_value'       : 0.0,
        'corrupt_project_count'   : 0,
        'govt_revenue_ratio'      : 0.0,
        'total_inflow'            : 0.0,
        'total_outflow'           : 0.0,
        'inflow_count'            : 0,
        'outflow_count'           : 0,
        'supplier_count'          : 0,
        'client_count'            : 0,
        'govt_official_links'     : 0,
        'related_companies_count' : 0,
    })

    # Org hierarchy setup
    print(f"\n[3/3] Setting up hierarchy with {n_officials} officials...")

    off_idx   = np.random.choice(df_persons.index, n_officials, replace=False)
    off_ids   = df_persons.loc[off_idx, 'person_id'].tolist()

    mayor_list = off_ids[0:1]
    kadis_list = off_ids[1:16]
    ppk_list   = off_ids[16:166]
    staff_list = off_ids[166:]

    role_map = (
        {p: 'Mayor'             for p in mayor_list} |
        {p: 'Head_of_Dept'      for p in kadis_list} |
        {p: 'PPK'               for p in ppk_list}   |
        {p: 'Procurement_Staff' for p in staff_list}
    )
    auth_map = (
        {p: 4 for p in mayor_list} |
        {p: 3 for p in kadis_list} |
        {p: 2 for p in ppk_list}   |
        {p: 1 for p in staff_list}
    )

    df_persons.loc[off_idx, 'is_official']           = 1
    df_persons.loc[off_idx, 'employment_type']        = 'civil_servant'
    df_persons.loc[off_idx, 'occupation_field']       = 'government'
    df_persons.loc[off_idx, 'is_politically_exposed'] = 1
    df_persons.loc[off_idx, 'monthly_income']         = np.random.lognormal(15.8, 0.3, n_officials).round(-5)
    df_persons.loc[off_idx, 'lhkpn_reported_asset']   = (
        df_persons.loc[off_idx, 'total_estimated_asset'] *
        np.random.uniform(0.4, 0.9, n_officials)).round(-6)
    df_persons['official_role']          = df_persons['person_id'].map(role_map).fillna('none')
    df_persons['procurement_auth_level'] = df_persons['person_id'].map(auth_map).fillna(0).astype(int)

    dept_pool  = [f"DEPT_{i:02d}" for i in range(1, 16)]
    kadis_dept = {k: random.choice(dept_pool) for k in kadis_list}
    ppk_dept   = {p: kadis_dept[random.choice(kadis_list)] for p in ppk_list}

    hierarchy_rows = (
        [{'person_id': mayor_list[0], 'role': 'Mayor',             'reports_to': None,                      'department': 'City_Office'}] +
        [{'person_id': p,             'role': 'Head_of_Dept',      'reports_to': mayor_list[0],             'department': kadis_dept[p]} for p in kadis_list] +
        [{'person_id': p,             'role': 'PPK',               'reports_to': random.choice(kadis_list), 'department': ppk_dept[p]}   for p in ppk_list] +
        [{'person_id': p,             'role': 'Procurement_Staff', 'reports_to': random.choice(ppk_list),  'department': ppk_dept.get(random.choice(ppk_list), 'DEPT_01')} for p in staff_list]
    )
    df_hierarchy = pd.DataFrame(hierarchy_rows)

    print("\n✅ Done generating entities!")
    print(f"  Persons   : {df_persons.shape}")
    print(f"  Companies : {df_companies.shape}")
    print(f"  Hierarchy : {df_hierarchy.shape}")
    print(f"  Shell companies masked: {_shell_idx.sum()}")
    print(f"\n  Official roles:")
    print(df_hierarchy['role'].value_counts().to_string())

    return df_persons, df_companies, df_hierarchy
