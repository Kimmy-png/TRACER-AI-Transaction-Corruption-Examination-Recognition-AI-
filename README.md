# TRACER-AI (Transaction Corruption Examination & Recognition AI)

A machine learning and network analysis-based platform for detecting corruption risks in government financial transactions using synthetic transaction data.

## Project Overview

This project implements a comprehensive system for identifying fraudulent patterns and corruption risks within a realistic government financial ecosystem. It simulates a large-scale transaction network comprising **100,000 individuals**, **4,500 companies**, and **3,000 government officials**, then applies advanced machine learning and network analysis techniques to detect suspicious activities.

The system employs two complementary AI models:

- **AI-1 — Project Risk Classifier** (Gradient Boosting): Evaluates the corruption risk score for each procurement project based on historical transaction patterns and financial indicators.
- **AI-2 — Edge Risk Scorer** (Random Forest + NetworkX): Identifies suspicious transactions by analyzing individual relationships and transaction flows, visualizing the risk network as an interactive graph.

All data is fully **synthetic** (generated using Faker with Indonesian locale) and designed following **strict anti-leakage principles** — models do not use features derived directly from target labels, ensuring realistic predictive performance.

## Project Structure

The codebase is organized as follows:

```
files/
├── main.py                      # Entry point — executes the complete pipeline
├── generate_entities.py         # Step 1: Generate persons, companies, and organizational hierarchy
├── generate_transactions.py     # Step 2: Generate synthetic transactions (legitimate + fraudulent)
├── feature_engineering.py       # Step 3: Extract and engineer features from raw transactions
├── models.py                    # Step 4 & 5: Train AI-1 (Gradient Boosting) and AI-2 (Random Forest)
├── visualize.py                 # Step 6: Generate network visualization and risk analysis charts
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Key Features

- **Synthetic Data Generation**: Creates realistic government financial transaction networks with no real-world data exposure
- **Dual AI Architecture**: Combines project-level risk scoring with transaction-level anomaly detection
- **Network Analysis**: Visualizes risk patterns as interactive network graphs using NetworkX and Matplotlib
- **Anti-Leakage Design**: Ensures models learn generalizable patterns rather than exploiting data artifacts
- **Production-Ready Pipeline**: Modular design allows selective execution of individual steps

## Installation

### Prerequisites
- Python 3.8 or higher
- pip or conda package manager

### Setup Instructions

Clone or download the repository and install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Running the Complete Pipeline

Execute all steps from data generation to visualization:

```bash
python main.py
```

### Running Specific Pipeline Steps

Execute only selected steps (useful for iterative development):

```bash
python main.py --steps entities transactions features ai1 ai2 visualize
```

### Headless Mode (Server Environments)

Run without generating GUI visualizations:

```bash
python main.py --no-visual
```

### Advanced Options

Customize output and filtering:

```bash
python main.py --top-n 50 --output risk_report.png
```

**Parameters:**
- `--steps`: Specify which pipeline steps to execute (space-separated)
- `--top-n`: Display top N highest-risk transactions/entities
- `--output`: Specify output filename for network visualization
- `--no-visual`: Suppress graphical output

## Technical Architecture

### Data Flow Pipeline

```
Entity Generation          Transaction Generation        Feature Engineering
    ↓                          ↓                              ↓
100K Persons           Legitimate Transactions         8–14 Dimensional
4.5K Companies         (Salary, Spending, B2B,         Feature Vectors
3K Officials           Procurement Projects, etc.)     
                                ↓
                    ┌───────────────────────┐
                    │   Model Training      │
                    ├───────────────────────┤
                    │ AI-1: Gradient        │ → Project Risk Scores
                    │ Boosting Classifier   │
                    │                       │
                    │ AI-2: Random Forest + │ → Transaction Edge Risk
                    │ NetworkX Analysis     │   Scores
                    └───────────────────────┘
                                ↓
                    Network Visualization
                    (Interactive Risk Graph)
```

### Model Descriptions

**AI-1 (Project Risk Classifier)**
- Algorithm: Gradient Boosting
- Task: Binary classification of procurement projects as high-risk or low-risk
- Output: Probability scores for corruption risk
- Features: Aggregated transaction metrics, organizational hierarchy indicators

**AI-2 (Edge Risk Scorer)**
- Algorithms: Random Forest for risk scoring + NetworkX for relationship mapping
- Task: Identify suspicious transactions between entities
- Output: Edge weights representing transaction anomaly scores
- Visualization: Network graph where nodes are entities and edges are suspicious transactions

## Output and Results

The system generates comprehensive analysis outputs:

### Console Output
- Model evaluation metrics (ROC-AUC scores, Precision, Recall, F1-Score)
- Classification reports and confusion matrices for both AI models
- Feature importance rankings showing which transaction patterns best predict corruption risk
- Top high-risk entities and transactions

### Generated Artifacts
- **corruption_risk_network.png**: Interactive network visualization showing:
  - Nodes: Government officials, companies, and individuals
  - Edges: Suspicious transactions between entities
  - Node size: Proportional to transaction volume
  - Edge color/width: Indicates risk score magnitude
  - Node coloring: Identifies entity type (official, company, individual) and risk category

### Example Results

See [examples/README.md](examples/README.md) for sample outputs from a full pipeline run, including:
- Model performance metrics
- Feature importance analysis
- Network visualization statistics

## Data Integrity and Validation

### Anti-Data-Leakage Principles

The project strictly adheres to best practices for preventing data leakage:

- **true_company_type** (indicating shell companies): Excluded from model training; used only for ground truth validation
- **is_illicit** (transaction fraud label): Not used as a feature in AI-2; available only for model evaluation
- **Shell Company Detection**: Achieved through behavioral pattern analysis, not direct feature inclusion

This approach ensures models learn generalizable corruption detection patterns rather than exploiting artifacts in the training data, resulting in realistic performance metrics applicable to production scenarios.

## Data Characteristics

### Dataset Scale
- **100,000 individuals**: Citizens and government staff
- **4,500 companies**: Mix of legitimate vendors and shell companies
- **3,000 government officials**: Decision-makers in procurement processes
- **~500,000+ transactions**: Diverse transaction types including salary, procurement, spending, and B2B transfers

### Transaction Types
- Salary payments to officials
- Vendor spending and procurement
- Company-to-company transfers
- Embezzlement attempts
- Layered transactions (money laundering patterns)

## Methodology and Evaluation

The project validates model performance using industry-standard metrics:

- **ROC-AUC Score**: Measures the model's ability to distinguish between corrupt and legitimate transactions across all threshold values
- **Precision and Recall**: Evaluates the trade-off between false positives and false negatives
- **Feature Importance Analysis**: Identifies the most predictive transaction patterns for corruption detection
- **Network Analysis Metrics**: Analyzes community structure and centrality measures in the transaction network

## Use Cases

This system is designed for:

- **Fraud Detection Teams**: Automated screening of suspicious financial patterns in government institutions
- **Compliance and Risk Management**: Identifying high-risk procurement projects and entities
- **Training and Research**: Educational platform for machine learning and network analysis applications
- **System Evaluation**: Benchmark for testing new corruption detection algorithms

## Technical Requirements

- **Python 3.8+**
- **Key Libraries**: scikit-learn, XGBoost, NetworkX, Pandas, NumPy, Matplotlib, Faker

See `requirements.txt` for complete dependency list with specific versions.

## Performance Considerations

- **Runtime**: Complete pipeline execution typically requires 2-5 minutes on standard hardware
- **Memory Usage**: Approximately 4-8 GB RAM for full dataset generation and model training
- **Scalability**: Architecture supports scaling to larger entity and transaction volumes with minimal code modifications

## Future Enhancements

Planned improvements for this project:
- Deep learning models for temporal pattern analysis
- Real-time transaction monitoring capabilities
- Interactive web-based dashboard for risk visualization
- Integration with additional data sources and external APIs
- Explainability features using SHAP values

## License and Attribution

This project is provided as-is for research, educational, and evaluation purposes. All data is fully synthetic and contains no real-world information.

## Support

For questions, issues, or suggestions, please refer to the project documentation or contact the development team.

---

**Project Status**: Active Development  
**Last Updated**: March 2026  
**Version**: 1.0
