# Hybrid GAN - Copula Data Synthesizer

**Hybrid Generative Adversarial Neural Network and Copul Data Synthesizer**



A powerful Streamlit-based application that enables you to **upload a real-world tabular dataset**, preprocess it automatically, and generate **high-quality synthetic datasets** using **Gaussian Copula** and **Generative Adversarial Network (GAN)**–based synthesis models.

---

##  Overview

This project provides a **hybrid framework** for **tabular data generation** using both statistical and deep-learning-based generative models.

It combines the strengths of:
- **Gaussian Copula models** for learning statistical dependencies,
- **Generative Adversarial Networks (CTGAN)** for capturing complex, non-linear relationships.

The interface is built with **Streamlit**, providing an intuitive and interactive web-based environment for:
- Uploading datasets,
- Configuring synthesis parameters,
- Training models,
- Evaluating generated data quality,
- Downloading synthetic datasets.

---

##  Key Features

###  1. Dual Synthesis Engines
Choose between:
- **Gaussian Copula:** Fast, statistically driven model for smaller datasets.
- **Generative Adversarial Network (GAN/CTGAN):** Deep-learning approach for complex, large, or high-dimensional data.

###  2. Automated Preprocessing
- Detects column types automatically.
- Cleans missing values, encodes categories, parses datetimes.
- Handles high-cardinality columns with configurable *Top-K mapping*.
- Auto-converts incompatible types to SDV-supported formats.

###  3. Evaluation Metrics
After generating synthetic data, the app computes:
- **Kolmogorov–Smirnov (KS) Statistic:** For numeric distribution similarity.
- **Jensen–Shannon (JS) Divergence:** For categorical distributions.
- **Correlation Matrix MSE:** To assess structure preservation.
- **ML Utility (AUC):** Tests predictive similarity between real and synthetic data.
- **Privacy Metrics (Memorization Check):** Ensures synthetic data doesn’t memorize real rows.

###  4. Synthetic Data Export
- Instantly download your generated synthetic dataset as a CSV file.
- Columns and types aligned with your original dataset.

###  5. Beautiful Modern UI
- Built with **Streamlit** and enhanced with custom CSS.
- Dark theme with glowing gradients and animated UI components.
- Real-time progress indicators and expandable sections for clarity.

---

##  Tech Stack

| Component | Technology |
|------------|-------------|
| **Frontend** | Streamlit |
| **Backend / Model Engine** | [SDV (Synthetic Data Vault)](https://github.com/sdv-dev/SDV) |
| **Models Used** | GaussianCopula, CTGAN |
| **Language** | Python 3.8+ |
| **Libraries** | pandas, numpy, sdv, scipy, scikit-learn |
| **Interface** | Streamlit UI with custom CSS |

---


