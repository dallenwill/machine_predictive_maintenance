# Machine Predictive Maintenance

A machine learning system to predict industrial machine failures using time-series sensor data. The project covers the full data science lifecycle — from exploratory analysis through feature engineering to model training and evaluation — with the goal of enabling proactive maintenance scheduling and reducing unplanned downtime.

## Project Overview

Unplanned machine failures are costly in industrial settings. This project frames the problem as a **binary classification task**: given real-time sensor readings, predict whether a machine is likely to fail. By flagging at-risk equipment before failure occurs, maintenance teams can intervene early and avoid expensive emergency repairs.

## Repository Structure

```
machine_predictive_maintenance/
├── data/                   # Raw and processed datasets
├── model/                  # Saved model artifacts
├── notebooks/
│   ├── mpm_02_exploratory_data_analysis.ipynb   # EDA: distributions, correlations, class imbalance
│   ├── mpm_03_data_processing.ipynb             # Feature engineering and preprocessing pipeline
│   └── mpm_04_modeling.ipynb                    # Model training, evaluation, and selection
└── reports/
    ├── mpm_problem_statement.pdf    # Business problem framing and objectives
    ├── mpm_presentation.pdf         # Summary slide deck
    └── mpm_project_report.pdf       # Full technical write-up
```

## Notebooks

| Notebook | Description |
|---|---|
| `mpm_02_exploratory_data_analysis` | Univariate and multivariate analysis of sensor features, failure rates, and class distributions |
| `mpm_03_data_processing` | Data cleaning, feature engineering, scaling, and train/test splitting |
| `mpm_04_modeling` | Model training, hyperparameter tuning, and performance evaluation |

## Methodology

1. **Exploratory Data Analysis** — Examined sensor readings (e.g., temperature, rotational speed, torque, tool wear) for patterns associated with machine failures. Investigated class imbalance in the target variable.

2. **Data Processing** — Engineered features from raw sensor data, handled class imbalance, and built a reproducible preprocessing pipeline.

3. **Modeling** — Trained and compared classification models, optimizing for recall to minimize missed failure predictions. Evaluated with standard metrics (precision, recall, F1, ROC-AUC).

## Reports

- **Problem Statement** — Defines the business context, objectives, and success criteria.
- **Project Report** — Full technical documentation including methodology, results, and recommendations.
- **Presentation** — High-level summary of findings for a non-technical audience.

## Getting Started

### Prerequisites

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### Running the Notebooks

Clone the repository and run the notebooks in order:

```bash
git clone https://github.com/dallenwill/machine_predictive_maintenance.git
cd machine_predictive_maintenance
jupyter notebook
```

Start with `mpm_02_exploratory_data_analysis.ipynb`, then proceed through `mpm_03` and `mpm_04`.

## Tech Stack

- **Python** — Core language
- **Jupyter Notebooks** — Analysis and modeling environment
- **scikit-learn** — Preprocessing and ML models
- **pandas / NumPy** — Data manipulation
- **Matplotlib / Seaborn** — Visualization
