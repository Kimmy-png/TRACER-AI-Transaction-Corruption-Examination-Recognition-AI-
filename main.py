"""
Run the corruption risk detection pipeline end-to-end.

Usage:
    python main.py                  # Run everything
    python main.py --steps ai1 ai2  # Only train models
    python main.py --no-visual      # Skip visualization

"""

import argparse
import time

from generate_entities import generate_entities
from generate_transactions import generate_transactions
from feature_engineering import engineer_features
from models import train_ai1, train_ai2
from visualize import visualize_network


ALL_STEPS = ['entities', 'transactions', 'features', 'ai1', 'ai2', 'visualize']


def parse_args():
    parser = argparse.ArgumentParser(description='Corruption Risk Detection Pipeline')
    parser.add_argument(
        '--steps', nargs='+', choices=ALL_STEPS, default=ALL_STEPS,
        help='which steps to run')
    parser.add_argument(
        '--no-visual', action='store_true',
        help='skip visualization')
    parser.add_argument(
        '--top-n', type=int, default=25,
        help='how many top projects to show in the graph')
    parser.add_argument(
        '--output', type=str, default='corruption_risk_network.png',
        help='output file for the visualization')
    return parser.parse_args()


def main():
    args  = parse_args()
    steps = set(args.steps)
    t0    = time.time()

    print("\n" + "=" * 60)
    print("  CORRUPTION RISK DETECTION")
    print("=" * 60)
    print(f"  Steps    : {', '.join(args.steps)}")
    print(f"  Top-N    : {args.top_n}")
    print(f"  Output   : {args.output}")
    print("=" * 60 + "\n")

    # Generate entities first (required)
    if 'entities' in steps:
        df_persons, df_companies, df_hierarchy = generate_entities()
    else:
        raise RuntimeError("need to generate entities first")

    # Generate transactions
    if 'transactions' in steps:
        df_all_transactions, df_projects = generate_transactions(df_persons, df_companies)
    else:
        raise RuntimeError("need to generate transactions")

    # Engineer features
    if 'features' in steps:
        df_persons, df_companies = engineer_features(
            df_persons, df_companies, df_all_transactions, df_projects)

    # Train AI-1 (project risk)
    if 'ai1' in steps:
        ai1, df_projects_scored, flagged_ids, feature_cols, le_store = train_ai1(
            df_persons, df_companies, df_projects)
    else:
        flagged_ids = []
        df_projects_scored = df_projects.copy()

    # Train AI-2 (transaction risk)
    if 'ai2' in steps:
        ai2, G, df_trx_flagged, all_node_risk, get_node_type = train_ai2(
            df_persons, df_companies, df_all_transactions,
            df_projects, flagged_ids)

    # Visualize
    if 'visualize' in steps and not args.no_visual:
        visualize_network(
            df_projects_scored=df_projects_scored,
            df_trx_flagged=df_trx_flagged,
            G=G,
            all_node_risk=all_node_risk,
            get_node_type=get_node_type,
            top_n=args.top_n,
            output_path=args.output,
        )

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  Done in {elapsed / 60:.1f} minutes")
    print(f"{'=' * 60}\n")


if __name__ == '__main__':
    main()
