"""
Visualize the risk network as an interactive graph.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from sklearn.metrics import roc_auc_score


NODE_COLOR = {
    'gov'     : '#2196F3',
    'company' : '#4CAF50',
    'shell'   : '#F44336',
    'official': '#FF9800',
    'person'  : '#9E9E9E',
}

LAYER_Y = {
    'gov': 0.92, 'company': 0.70, 'shell': 0.48,
    'person': 0.26, 'official': 0.06,
}


def visualize_network(
    df_projects_scored,
    df_trx_flagged,
    G: nx.DiGraph,
    all_node_risk,
    get_node_type,
    top_n: int = 25,
    output_path: str = 'corruption_risk_network.png',
):
    print("=" * 60)
    print("  Visualizing Risk Network")
    print("=" * 60)

    # Get top-N flagged projects
    top_proj_ids = (df_projects_scored
                    .sort_values('ai1_score', ascending=False)
                    .head(top_n)['project_id'].tolist())

    trx_top = df_trx_flagged[df_trx_flagged['project_ref'].isin(top_proj_ids)]

    # Build subgraph with these transactions
    G_sub = nx.DiGraph()
    for _, row in trx_top.iterrows():
        src, tgt = row['source_id'], row['target_id']
        for nid in [src, tgt]:
            if nid not in G_sub:
                G_sub.add_node(nid,
                               node_type=get_node_type(nid),
                               risk=float(all_node_risk.get(nid, 0.0)))
        if G_sub.has_edge(src, tgt):
            G_sub[src][tgt]['ai2_score'] = max(G_sub[src][tgt]['ai2_score'], row['ai2_edge_score'])
            G_sub[src][tgt]['is_illicit'] = max(G_sub[src][tgt]['is_illicit'], int(row['is_illicit']))
        else:
            G_sub.add_edge(src, tgt,
                           ai2_score  = float(row['ai2_edge_score']),
                           is_illicit = int(row['is_illicit']),
                           trx_type   = row['trx_type'])

    print(f"  Nodes: {G_sub.number_of_nodes()} | Edges: {G_sub.number_of_edges()}")

    # Layout with layers by entity type
    init_pos = {}
    _layers  = {}
    for n, d in G_sub.nodes(data=True):
        _layers.setdefault(d.get('node_type', 'company'), []).append(n)
    for nt, nodes in _layers.items():
        ys = np.linspace(0.05, 0.95, len(nodes))
        for n, x in zip(nodes, ys):
            init_pos[n] = (x, LAYER_Y.get(nt, 0.5))

    pos = nx.spring_layout(G_sub, pos=init_pos, k=0.40, iterations=80, seed=42, weight=None)

    # Color and size nodes by type and risk
    node_types  = [G_sub.nodes[n].get('node_type', 'company') for n in G_sub.nodes()]
    node_colors = [NODE_COLOR.get(t, '#9E9E9E') for t in node_types]
    node_risks  = [G_sub.nodes[n].get('risk', 0) for n in G_sub.nodes()]
    node_sizes  = [200 + 2800 * r for r in node_risks]

    # Color edges by risk score
    edge_scores = [G_sub[u][v]['ai2_score'] for u, v in G_sub.edges()]
    edge_widths = [0.5 + 4.0 * s for s in edge_scores]
    cmap_edge   = plt.cm.RdYlGn_r
    edge_colors = [cmap_edge(s) for s in edge_scores]

    # Create figure
    BG  = '#0f1724'
    fig = plt.figure(figsize=(26, 15))
    fig.patch.set_facecolor(BG)

    ax_net = fig.add_axes([0.01, 0.05, 0.68, 0.90])
    ax_net.set_facecolor('#141e30')

    # Draw edges and nodes
    nx.draw_networkx_edges(G_sub, pos, ax=ax_net,
        edge_color=edge_colors, width=edge_widths, alpha=0.82,
        arrows=True, arrowsize=14, arrowstyle='-|>',
        connectionstyle='arc3,rad=0.12',
        node_size=node_sizes,
        min_source_margin=8, min_target_margin=8)

    nx.draw_networkx_nodes(G_sub, pos, ax=ax_net,
        node_color=node_colors, node_size=node_sizes,
        alpha=0.93, linewidths=1.0, edgecolors='white')

    _label_nodes = {n for n in G_sub.nodes()
                    if G_sub.nodes[n].get('risk', 0) >= 0.45
                    or n == 'GOV_TREASURY'
                    or G_sub.nodes[n].get('node_type') == 'shell'}
    label_dict = {n: ('GOV\nTREASURY' if n == 'GOV_TREASURY' else n[-7:])
                  for n in _label_nodes}

    nx.draw_networkx_labels(G_sub, pos, label_dict, ax=ax_net,
        font_size=6, font_color='white', font_weight='bold')

    for y_frac, label in LAYER_Y.items():
        ys = [v[1] for v in pos.values()]
        if not ys:
            continue
        y_mn, y_mx = min(ys), max(ys)
        y_data = y_mn + (label / 1.0) * (y_mx - y_mn)
        ax_net.axhline(y_data, color='white', alpha=0.06, lw=0.6, linestyle='--')
        ax_net.text(ax_net.get_xlim()[0], y_data + 0.008,
                    f'  {y_frac}', color='white', alpha=0.40, fontsize=7)

    ax_net.set_title(
        f'Corruption Risk Network  ·  Top-{top_n} Flagged Projects\n'
        f'Node size = behavioral risk score  ·  Edge color = AI-2 edge risk',
        color='white', fontsize=13, fontweight='bold', pad=14)
    ax_net.axis('off')

    sm = ScalarMappable(cmap=cmap_edge, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax_net, shrink=0.40, pad=0.01, aspect=22)
    cbar.set_label('AI-2 Edge Risk Score', color='white', fontsize=9)
    cbar.ax.yaxis.set_tick_params(color='white', labelsize=7)
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    leg1 = ax_net.legend(
        handles=[mpatches.Patch(color=v, label=k.capitalize()) for k, v in NODE_COLOR.items()],
        loc='upper left', framealpha=0.25, facecolor='#111', labelcolor='white',
        fontsize=8, title='Node type', title_fontsize=8)
    leg1.get_title().set_color('white')
    ax_net.add_artist(leg1)
    ax_net.legend(
        handles=[Line2D([0], [0], marker='o', color='w', markerfacecolor='#888',
                        markersize=s, label=l, linestyle='None')
                 for s, l in [(5, 'Low risk'), (10, 'Medium risk'), (15, 'High risk')]],
        loc='lower left', framealpha=0.25, facecolor='#111',
        labelcolor='white', fontsize=8, title='Node size', title_fontsize=8
    ).get_title().set_color('white')

    # Right panel: stats and histograms
    ax_ai1 = fig.add_axes([0.72, 0.60, 0.25, 0.28])
    ax_ai2 = fig.add_axes([0.72, 0.22, 0.25, 0.28])
    ax_txt = fig.add_axes([0.72, 0.05, 0.25, 0.14])
    for ax_ in [ax_ai1, ax_ai2, ax_txt]:
        ax_.set_facecolor('#141e30')
        for sp in ax_.spines.values():
            sp.set_color('#333')

    # AI-1 scores histogram
    s_all = df_projects_scored['ai1_score']
    ax_ai1.hist(s_all[df_projects_scored['is_corrupt'] == 0], bins=25, alpha=0.65, color='#4CAF50', label='Clean')
    ax_ai1.hist(s_all[df_projects_scored['is_corrupt'] == 1], bins=25, alpha=0.65, color='#F44336', label='Corrupt')
    ax_ai1.axvline(0.35, color='white', ls='--', lw=0.8, alpha=0.6)
    ax_ai1.set_title('Project Risk (AI-1)', color='white', fontsize=9, pad=5)
    ax_ai1.tick_params(colors='white', labelsize=7)
    ax_ai1.legend(fontsize=7, labelcolor='white', facecolor='#111', framealpha=0.4)
    ax_ai1.spines[['top', 'right']].set_visible(False)

    # AI-2 scores histogram
    e_clean   = [G_sub[u][v]['ai2_score'] for u, v in G_sub.edges() if G_sub[u][v]['is_illicit'] == 0]
    e_illicit = [G_sub[u][v]['ai2_score'] for u, v in G_sub.edges() if G_sub[u][v]['is_illicit'] == 1]
    if e_clean:   ax_ai2.hist(e_clean,   bins=20, alpha=0.65, color='#4CAF50', label='Clean')
    if e_illicit: ax_ai2.hist(e_illicit, bins=20, alpha=0.65, color='#F44336', label='Illicit')
    ax_ai2.axvline(0.50, color='white', ls='--', lw=0.8, alpha=0.6)
    ax_ai2.set_title('Transaction Risk (AI-2)', color='white', fontsize=9, pad=5)
    ax_ai2.tick_params(colors='white', labelsize=7)
    ax_ai2.legend(fontsize=7, labelcolor='white', facecolor='#111', framealpha=0.4)
    ax_ai2.spines[['top', 'right']].set_visible(False)

    # Summary text
    n_hi_edges = sum(1 for _, _, d in G_sub.edges(data=True) if d['ai2_score'] >= 0.5)
    ai1_auc    = roc_auc_score(df_projects_scored['is_corrupt'], df_projects_scored['ai1_score'])
    flagged_ids = df_projects_scored[df_projects_scored['ai1_flagged'] == 1]['project_id'].tolist()

    ax_txt.axis('off')
    ax_txt.text(0.05, 0.95,
        f"SUMMARY\n"
        f"{'─' * 28}\n"
        f"Total projects    : {len(df_projects_scored)}\n"
        f"Flagged by AI-1   : {len(flagged_ids)}\n"
        f"AI-1 ROC-AUC      : {ai1_auc:.3f}\n"
        f"Network nodes     : {G_sub.number_of_nodes()}\n"
        f"Network edges     : {G_sub.number_of_edges()}\n"
        f"High-risk edges   : {n_hi_edges}\n",
        transform=ax_txt.transAxes,
        color='white', fontsize=8.5, va='top', fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#0f1724', alpha=0.9))

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.show()
    print(f"\n✅ Visualisasi selesai! File: {output_path}")
