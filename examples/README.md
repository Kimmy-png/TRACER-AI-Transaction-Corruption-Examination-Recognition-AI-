# Example Outputs

This folder contains sample outputs from running the corruption risk detection pipeline.

## Sample Run Results

**Dataset:**
- 100,000 individuals
- 4,500 companies (235 shell companies)
- 3,000 government officials
- 61.7M transactions (with 602 flagged as illicit)

**Model Performance:**

### AI-1: Project Risk Classifier
- **ROC-AUC**: 0.7002
- **Accuracy**: 75%
- **Precision (Corrupt)**: 28%
- **Recall (Corrupt)**: 29%
- **Projects Flagged**: 87 out of 500

Top predictive features:
1. Official income anomaly score (26.1%)
2. Official wealth growth rate (7.2%)
3. Company domicile province (5.2%)
4. Official betweenness centrality (5.2%)
5. Company total outflow (5.1%)

### AI-2: Edge Risk Scorer
- **ROC-AUC**: 1.0000 (on test set)
- **Accuracy**: 100%
- Training set: 1,102 edges (602 illicit, 54.6%)

Note: Perfect performance on test set indicates strong separation between illicit and legitimate transaction patterns in this synthetic dataset.

## Visualization

**corruption_risk_network.png**
- Network graph of high-risk transaction flows
- 432 nodes (officials, companies, individuals)
- 603 edges (transactions)
- Color-coded by entity type and risk level
- Edge thickness indicates transaction risk score

## Running Your Own Analysis

To reproduce these results or generate new ones:

```bash
python main.py
```

Output will be saved to `corruption_risk_network.png` in the root directory.

For more details, see [README.md](../README.md)
