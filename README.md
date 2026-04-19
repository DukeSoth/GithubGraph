# GithubGraph

# Mapping Collaboration Risk in Open-Source Projects

This project analyzes open-source collaboration patterns on GitHub using graph-based methods and machine learning to identify repositories that are at risk of structural decline.

## 📌 Overview

Open-source projects often depend on a small number of contributors. If these contributors reduce their activity, the project may become inactive or fail.

This project models collaboration as a graph and uses structural features to predict repository-level risk.

The pipeline combines:

- Graph construction (developer–repository relationships)
- Feature engineering (centralization, inequality, robustness, temporal trends)
- Supervised learning (risk prediction)
- Interpretable scoring (explainable risk components)

---

## Problem Definition

We formulate this as a **binary classification problem**:

A repository is labeled as **at risk (y = 1)** if:

- Future activity drops below 50% of observation period, OR  
- The repository becomes inactive for ≥ 90 consecutive days  

Otherwise:

- y = 0 (stable)

---

## Dataset

We use the **GH Archive dataset**, which contains public GitHub events.

### Event Types Used
- PushEvent
- PullRequestEvent
- IssuesEvent
- ReleaseEvent

### Time Windows
- **Observation Window:** Jan 1 – Jun 30, 2015  
- **Future Window:** Jul 1 – Sep 30, 2015  

The time split ensures **no data leakage** between features and labels.

---

## Pipeline

The full pipeline:

### 1. Preprocessing
- Clean raw events (deduplication, filtering)
- Split into observation and future windows

### 2. Graph Construction
- Bipartite graph (developers ↔ repositories)
- Contributor projection graph (developer collaboration network)

### 3. Feature Engineering

Repository-level features include:

#### Contribution Features
- Number of contributors
- Total contributions
- Top-1 / Top-3 contribution share
- Gini coefficient

#### Temporal Features
- Contributor growth/decline (slope)
- Activity volume trend

#### Structural Features
- Degree centralization
- k-core number
- Largest connected component after removal
- Graph density change

---

## Modeling

We train multiple models:

- Dummy baseline (majority class)
- Logistic Regression (L2 regularized)
- Random Forest (shallow)

### Feature Selection
- SelectKBest (mutual information)

### Validation
- Leave-One-Out Cross Validation (LOO)

---

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

Evaluation is done using **out-of-fold predictions**.

---

## Risk Scoring (Interpretability Layer)

In addition to ML predictions, we compute an interpretable risk score:

### Components
- **Activity Risk** (trend decline)
- **Contributor Risk** (concentration / bus factor)
- **Structural Risk** (network fragility)

### Final Score

Composite Risk = weighted sum of components


Each repository is assigned:
- Low / Medium / High / Critical risk tier

---

## Project Structure


src/
├── data_loading.py # Load GH Archive data
├── preprocessing.py # Clean + split data
├── graph_building.py # Construct graphs
├── feature_engineering.py # Extract features
├── labeling.py # Define risk labels
├── modeling.py # Train models
├── evaluation.py # Metrics + plots
├── risk_score.py # Interpretable scoring
├── utils.py # Helpers
└── run_preprocessing.py # Pipeline entry point

data/
└── processed/ # Processed datasets

outputs/
├── features.csv
├── labels.csv
├── cv_results.csv
├── loo_predictions.csv
├── feature_importance.csv
├── evaluation_report.txt
└── plots/
