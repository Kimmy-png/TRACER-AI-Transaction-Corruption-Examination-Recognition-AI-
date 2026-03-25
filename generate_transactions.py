"""
Generate synthetic transactions including salaries, spending, B2B transfers, and corruption patterns.
"""

import pandas as pd
import numpy as np
import random


def generate_transactions(df_persons: "pd.DataFrame", df_companies: "pd.DataFrame", seed: int = 42):
    np.random.seed(seed)
    random.seed(seed)

    print("=" * 60)
    print("  Generating Transactions")
    print("=" * 60)

    ppk_list   = df_persons[df_persons['official_role'] == 'PPK']['person_id'].tolist()
    kadis_list = df_persons[df_persons['official_role'] == 'Head_of_Dept']['person_id'].tolist()
    staff_list = df_persons[df_persons['official_role'] == 'Procurement_Staff']['person_id'].tolist()

    assert len(ppk_list) > 0,   "no PPKs found"
    assert len(shell_cos) > 0,  "no shell companies"
    assert len(normal_cos) > 0, "no normal companies"

    print(f"  PPKs            : {len(ppk_list)}")
    print(f"  Shell companies : {len(shell_cos)}")
    print(f"  Normal companies: {len(normal_cos)}")

    # Part 1: Normal/legitimate transactions
    print("\n[Part 1] Generating legitimate transactions...")

    # Monthly salaries
    print("  -> Salary payments...")
    employers = np.where(is_official_arr == 1, 'GOV_TREASURY',
                         np.random.choice(company_ids_arr, len(persons_ids_arr)))
    paydays   = [25, 55, 85, 115, 145, 175, 205, 235, 265, 295, 325, 355]

    df_salaries = pd.DataFrame({
        'day'        : np.repeat(paydays, len(persons_ids_arr)),
        'source_id'  : np.tile(employers, MONTHS),
        'target_id'  : np.tile(persons_ids_arr, MONTHS),
        'amount'     : np.tile(incomes, MONTHS),
        'trx_type'   : 'salary',
        'project_ref': None,
        'is_illicit' : 0,
    })

    # Daily spending  
    print("  -> Daily spending...")
    trx_yearly       = trx_counts_mo * MONTHS
    total_spend_rows = int(trx_yearly.sum())
    avg_trx_val      = (incomes * spend_ratios) / trx_counts_mo

    spend_sources = np.repeat(persons_ids_arr, trx_yearly)
    base_amounts  = np.repeat(avg_trx_val, trx_yearly)
    noise         = np.random.lognormal(0, 0.8, total_spend_rows)
    spend_amounts = np.round(base_amounts * noise, -3)

    df_spending = pd.DataFrame({
        'day'        : np.random.randint(1, DAYS_IN_YEAR + 1, total_spend_rows),
        'source_id'  : spend_sources,
        'target_id'  : np.random.choice(company_ids_arr, total_spend_rows),
        'amount'     : spend_amounts,
        'trx_type'   : 'daily_consumption',
        'project_ref': None,
        'is_illicit' : 0,
    })
    df_spending = df_spending[df_spending['amount'] > 1_000]

    # Company-to-company transfers
    print("  -> B2B transfers...")
    N_B2B = len(company_ids_arr) * 120
    df_b2b = pd.DataFrame({
        'day'        : np.random.randint(1, DAYS_IN_YEAR + 1, N_B2B),
        'source_id'  : np.random.choice(company_ids_arr, N_B2B),
        'target_id'  : np.random.choice(company_ids_arr, N_B2B),
        'amount'     : np.round(np.random.lognormal(17, 1.5, N_B2B), -5),
        'trx_type'   : 'b2b_operational',
        'project_ref': None,
        'is_illicit' : 0,
    })

    # Part 2: Inject corruption patterns
    print("\n[Part 2] Injecting corruption patterns...")

    N_PROJECTS      = 500
    CORRUPTION_RATE = 0.15

    projects_list    = []
    corruption_edges = []

    for i in range(N_PROJECTS):
        is_corrupt = int(random.random() < CORRUPTION_RATE)
        prj_day    = random.randint(1, DAYS_IN_YEAR - 60)
        budget     = round(np.random.lognormal(22, 1.5), -6)
        pic_ppk    = random.choice(ppk_list)
        winner_co  = random.choice(normal_cos)
        prj_id     = f"PRJ{str(i).zfill(4)}"

        projects_list.append({
            'project_id'        : prj_id,
            'day'               : prj_day,
            'budget'            : budget,
            'official_id'       : pic_ppk,
            'winner_company_id' : winner_co,
            'is_corrupt'        : is_corrupt,
        })

        payout_day = min(prj_day + 30, DAYS_IN_YEAR)
        corruption_edges.append({
            'day': payout_day, 'source_id': 'GOV_TREASURY', 'target_id': winner_co,
            'amount': budget, 'trx_type': 'project_payout',
            'project_ref': prj_id, 'is_illicit': 0,
        })

        if is_corrupt:
            kickback  = budget * random.uniform(0.10, 0.20)
            shell_tgt = random.choice(shell_cos)
            l1_day    = min(payout_day + random.randint(2, 7), DAYS_IN_YEAR)

            corruption_edges.append({
                'day': l1_day, 'source_id': winner_co, 'target_id': shell_tgt,
                'amount': kickback, 'trx_type': 'subcontractor_fee_fake',
                'project_ref': prj_id, 'is_illicit': 1,
            })

            proxies = random.sample(persons_ids_arr.tolist(), 3)
            split   = round(kickback / 3, -2)
            for proxy in proxies:
                l2_day = min(l1_day + random.randint(1, 14), DAYS_IN_YEAR)
                l3_day = min(l2_day + random.randint(5, 30), DAYS_IN_YEAR)
                corruption_edges.append({
                    'day': l2_day, 'source_id': shell_tgt, 'target_id': proxy,
                    'amount': split, 'trx_type': 'consulting_fee_fake',
                    'project_ref': prj_id, 'is_illicit': 1,
                })
                corruption_edges.append({
                    'day': l3_day, 'source_id': proxy, 'target_id': pic_ppk,
                    'amount': round(split * 0.9, -2), 'trx_type': 'asset_transfer_or_cash',
                    'project_ref': prj_id, 'is_illicit': 1,
                })

    df_needles  = pd.DataFrame(corruption_edges)
    df_projects = pd.DataFrame(projects_list)

    # Combine all and sort
    print("\n[Part 3] Finalizing transaction records...")
    df_all_transactions = pd.concat(
        [df_salaries, df_spending, df_b2b, df_needles], ignore_index=True
    ).sort_values('day').reset_index(drop=True)

    print("\n✅ Done generating transactions!")
    print(f"  Total projects   : {len(df_projects):,}")
    print(f"  Total transactions: {len(df_all_transactions):,}")
    print(f"  Flagged as illicit: {(df_all_transactions['is_illicit'] == 1).sum():,}")
    print(f"  Corrupt projects : {df_projects['is_corrupt'].sum()}")

    return df_all_transactions, df_projects
