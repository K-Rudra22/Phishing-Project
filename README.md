# Phishing Email Detection with Sender Trust

A machine learning system for classifying phishing emails using **LightGBM** with a dynamic **Sender Trust** scoring mechanism. Built as a mini-project comparing Logistic Regression and LightGBM across five real-world email datasets.

Live demo → [Streamlit App](https://your-app-url.streamlit.app) *(replace with your deployed URL)*

---

## Results

| Metric | Logistic Regression | LightGBM v4 |
|---|---|---|
| ROC AUC (internal test) | 0.9951 | **0.9996** |
| Accuracy | 97% | **99%** |
| Phishing Recall | 0.96 | **1.00** |
| Phishing Precision | 0.97 | **0.98** |
| Phishing F1 | 0.96 | **0.99** |
| External ROC AUC | 0.857 | 1.00 ⚠️ |
| External Accuracy | 70% | 100% ⚠️ |
| Training Time | 325s | 408s |

> ⚠️ LightGBM's external score reflects in-sample performance — the 2,000 external validation emails were included in training to match the LR pipeline. The **internal 80/20 split (18,886 emails)** is the clean benchmark for LGB.

---

## How It Works

### Pipeline

```
Raw Emails
    ↓
Text Cleaning  →  TF-IDF (3,000 features, unigrams + bigrams)
    ↓
Structural Features (7)  →  StandardScaler
    ↓
Sender Trust Score (1)   →  Bayesian smoothing
    ↓
Feature Matrix (3,008)   →  LightGBM Classifier
    ↓
Phishing / Legitimate
```

### Structural Features
Seven hand-crafted features extracted from the **original** (uncleaned) email text:

| Feature | Description |
|---|---|
| `num_urls` | Total URLs in the email |
| `num_unique_domains` | Distinct domains |
| `has_ip_url` | URL with a raw IP address |
| `suspicious_tld` | TLD in `.ru .cn .tk .xyz .top .click` |
| `exclamation_count` | Number of `!` characters |
| `uppercase_ratio` | Fraction of uppercase characters |
| `urgent_flag` | Keywords: *urgent, immediately, action required, verify, suspended* |

### Sender Trust
A Bayesian-smoothed trust score computed per sender from historical labels:

```
trust = (legit_count + α) / (total_count + α + β)
```

Where `α = β = 2.0` and unknown senders default to `0.5`. In the Streamlit app, trust is updated in real time whenever a user marks an email as Legitimate or Phishing — correctly marked emails shift the sender's score for all future classifications.

**Sender trust is the #1 feature by information gain in LightGBM** (gain = 328,144 — 5× the next highest feature).

---

## Datasets

| Dataset | Emails | Source |
|---|---|---|
| CEAS-08 | 39,154 | Spam/phishing conference dataset |
| Enron | 29,767 | Enron email corpus |
| Phishing_Email | 18,650 | Kaggle phishing email dataset |
| SpamAssassin | 5,809 | Apache SpamAssassin public corpus |
| External Validation | 2,000 | Phishing validation set |
| **Total (after outlier removal)** | **94,426** | |

Labels: `0` = Legitimate, `1` = Phishing. Outliers removed at the 99th percentile of text length (~11,989 characters).

---

## Model Details

### LightGBM (primary model)
```python
LGBMClassifier(
    n_estimators=1500,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=50,
    feature_fraction=0.6,
    bagging_fraction=0.8,
    bagging_freq=5,
    reg_alpha=1.0,
    reg_lambda=1.0,
    scale_pos_weight=1.07,   # neg/pos ratio
    random_state=42
)
```
Early stopping: 50 rounds patience. Best iteration: 1,495/1,500.

### Logistic Regression (baseline)
```python
LogisticRegression(
    solver='saga',
    max_iter=4000,
    C=1.0,
    penalty='l2',
    class_weight='balanced'
)
```
Note: issued a `ConvergenceWarning` — the solver did not fully converge within 4,000 iterations.

---

## Streamlit App

The app provides a 7-tab interface for email analysis:

| Tab | Content |
|---|---|
| Prediction | Risk gauge (0–100%), verdict, manual ground truth buttons |
| Behavior Analysis | Live sender trust score, full history from SQLite |
| Feature Visualization | Radar chart of 8 structural/trust features |
| Feature Contribution | LightGBM gain-based importance for numeric features |
| Model Comparison | Text-only vs trust-aware probability side-by-side |
| Risk Breakdown | Trust influence delta on final score |
| Top Word Signals | Top 20 TF-IDF words by training gain, highlighted if present in email |

### Dynamic Trust Learning
When a user clicks **Mark as Legitimate** or **Mark as Phishing**, the sender's reputation is written to a local SQLite database. On the next analysis of any email from that sender, the updated trust score is used — adjusting the model's output accordingly.

---

## Repo Structure

```
phishing-project/
├── app.py                              # Streamlit application
├── requirements.txt                    # Python dependencies
├── runtime.txt                         # Python 3.11 (Streamlit Cloud)
├── lgb_phishing_model.pkl              # Trained LightGBM model
├── tfidf_vectorizer.pkl                # Fitted TF-IDF vectorizer (3,000 features)
├── structural_scaler.pkl               # Fitted StandardScaler (7 features)
├── sender_trust_dict.pkl               # Static trust dict from training
├── sender_email_count_dict.pkl         # Sender email counts from training
└── phishing_lightgbm_v4_final.ipynb   # Full training notebook
```

> The `.pkl` files are generated by running the notebook end-to-end in Google Colab with the datasets mounted in Google Drive.

---

## Setup

### Run locally
```bash
git clone https://github.com/your-username/phishing-project
cd phishing-project
pip install -r requirements.txt
streamlit run app.py
```

### Train from scratch
1. Upload the five CSV datasets to Google Drive at `MyDrive/phishing_v2/`
2. Open `phishing_lightgbm_v4_final.ipynb` in Google Colab
3. Run all cells — artifacts are downloaded automatically from Cell 23
4. Place the `.pkl` files in the repo root

---

## Dependencies

```
streamlit
joblib
numpy
pandas
scikit-learn
lightgbm
scipy
plotly
shap
```

Python 3.11 required (specified in `runtime.txt` for Streamlit Cloud).

---

## Limitations

- **False positives on security alerts**: Legitimate emails that use phishing-like language (e.g. "verify your account", "suspicious login detected") may score high. The sender trust system partially mitigates this after a few manual corrections.
- **Sender coverage**: Only CEAS-08 and SpamAssassin have sender fields — ~47% of training emails have non-neutral trust scores.
- **No truly unseen external benchmark for LGB**: A held-out third dataset would be needed for a fully clean external evaluation.
