import streamlit as st
import joblib
import numpy as np
import re
import plotly.graph_objects as go
from scipy.sparse import hstack, csr_matrix
from urllib.parse import urlparse
import sqlite3

# ============================
# LOAD ARTIFACTS
# ============================

# FIX 1: was "model_sender_trust_unified.pkl" (LR name) — notebook saves lgb_phishing_model.pkl
model      = joblib.load("lgb_phishing_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
scaler     = joblib.load("structural_scaler.pkl")

GLOBAL_TRUST_PRIOR = 0.5
ALPHA          = 2
AUTO_THRESHOLD = 0.8

# ============================
# SQLITE DATABASE (CACHED)
# ============================

@st.cache_resource
def init_db():
    conn   = sqlite3.connect("sender_reputation.db", check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sender_reputation (
            sender      TEXT PRIMARY KEY,
            legit_count INTEGER,
            phish_count INTEGER
        )
    """)
    conn.commit()
    return conn, cursor

conn, cursor = init_db()

# ============================
# FEATURE ENGINEERING
# Must exactly mirror notebook Cell 7 (clean_text) and Cell 9 (extract_features_correct)
# FIX 2: added '.cn' to suspicious_tlds (was missing vs notebook — notebook has 6, app had 5)
# FIX 3: urgent_words uses 'suspended' not 'suspend' (matches notebook training config exactly)
# ============================

SUSPICIOUS_TLDS = ['.ru', '.cn', '.tk', '.xyz', '.top', '.click']
URGENT_WORDS    = ['urgent', 'immediately', 'action required', 'verify', 'suspended']

def clean_text(text):
    """Mirrors notebook Cell 7 — used for TF-IDF input."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " URL ", text)
    text = re.sub(r"\S+@\S+", " EMAIL ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_structural_features(text):
    """Mirrors notebook Cell 9 — returns 7 features in exact training order."""
    text = str(text)

    urls            = re.findall(r'https?://\S+|www\.\S+', text)
    num_urls        = len(urls)
    domains         = []
    has_ip          = 0
    suspicious_flag = 0

    for url in urls:
        try:
            domain = urlparse(url).netloc
            domains.append(domain)
            if re.match(r'\d+\.\d+\.\d+\.\d+', domain):
                has_ip = 1
            if any(domain.endswith(t) for t in SUSPICIOUS_TLDS):
                suspicious_flag = 1
        except Exception:
            continue

    num_unique_domains = len(set(domains))
    exclamation_count  = text.count("!")
    uppercase_ratio    = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    urgent_flag        = int(any(w in text.lower() for w in URGENT_WORDS))

    return [
        num_urls,
        num_unique_domains,
        has_ip,
        suspicious_flag,
        exclamation_count,
        uppercase_ratio,
        urgent_flag,
    ]

# ============================
# TRUST SCORE (SQLite-backed)
# ============================

def get_trust_score(sender):
    if not sender:
        return GLOBAL_TRUST_PRIOR
    sender = sender.strip().lower()
    cursor.execute(
        "SELECT legit_count, phish_count FROM sender_reputation WHERE sender=?",
        (sender,)
    )
    row = cursor.fetchone()
    if row:
        legit, phish = row
        return (legit + ALPHA) / (legit + phish + 2 * ALPHA)
    return GLOBAL_TRUST_PRIOR

# ============================
# UPDATE REPUTATION
# ============================

def update_reputation(sender, true_label):
    if not sender:
        return
    sender = sender.strip().lower()
    cursor.execute(
        "SELECT legit_count, phish_count FROM sender_reputation WHERE sender=?",
        (sender,)
    )
    row = cursor.fetchone()
    legit, phish = row if row else (0, 0)
    if true_label == 0:
        legit += 1
    else:
        phish += 1
    cursor.execute(
        "INSERT OR REPLACE INTO sender_reputation (sender, legit_count, phish_count) VALUES (?, ?, ?)",
        (sender, legit, phish)
    )
    conn.commit()

# ============================
# BUILD FEATURE MATRIX
# Mirrors notebook Cell 14 exactly:
#   hstack([X_text(3000), X_struct_scaled(7), trust(1)]) → 3008 cols
# ============================

def build_feature_matrix(email_text, trust_score):
    cleaned       = clean_text(email_text)
    X_text        = vectorizer.transform([cleaned])                  # (1, 3000) sparse
    struct        = extract_structural_features(email_text)
    struct_scaled = scaler.transform(np.array(struct).reshape(1,-1)) # (1, 7) scaled
    trust_array   = np.array([[trust_score]])                         # (1, 1)
    X_numeric     = np.hstack([struct_scaled, trust_array])           # (1, 8)
    X_final       = hstack([X_text, csr_matrix(X_numeric)]).tocsr()  # (1, 3008)
    return X_final, X_text, struct

# ============================
# LGB FEATURE IMPORTANCE (cached — runs once per session)
# FIX 4: LightGBM has no coef_ — use booster_ gain importance instead
# Tabs 4 and 7 were completely blank before this fix
# ============================

@st.cache_resource
def get_feature_importance():
    gain  = model.booster_.feature_importance(importance_type='gain')
    names = model.feature_name_  # set in notebook via feature_name=all_feature_names

    n_tfidf = len(vectorizer.get_feature_names_out())  # 3000

    tfidf_importance   = {names[i]: gain[i] for i in range(n_tfidf)}
    numeric_importance = {names[i]: gain[i] for i in range(n_tfidf, len(names))}

    return tfidf_importance, numeric_importance

# ============================
# UI
# ============================

st.set_page_config(page_title="Sender-Trust Phishing Detection", layout="wide")
st.title("Sender-Trust Aware Phishing Detection")
st.caption("LightGBM · TF-IDF + Structural Features + Dynamic Sender Trust")

sender_input = st.text_input("Sender Email")
email_text   = st.text_area("Email Content", height=220)

if st.button("Analyze Email"):

    if not email_text.strip():
        st.warning("Please enter email content.")
        st.stop()

    trust_score = get_trust_score(sender_input)
    trust_pct   = trust_score * 100

    # Full matrix (with trust)
    X_final, X_text, struct_features = build_feature_matrix(email_text, trust_score)

    # Text-only matrix (trust set to 0, same shape → 3008 cols, fair comparison)
    X_text_only, _, _ = build_feature_matrix(email_text, 0.0)

    probability   = model.predict_proba(X_final)[0][1]
    prob_pct      = probability * 100
    text_prob_pct = model.predict_proba(X_text_only)[0][1] * 100

    # Auto-learning: update reputation on high-confidence predictions
    if probability > AUTO_THRESHOLD:
        update_reputation(sender_input, 1)
    elif probability < (1 - AUTO_THRESHOLD):
        update_reputation(sender_input, 0)

    # ============================
    # TABS
    # ============================

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Prediction",
        "Behavior Analysis",
        "Feature Visualization",
        "Feature Contribution",
        "Model Comparison",
        "Risk Breakdown",
        "Top Word Signals",
    ])

    # ── TAB 1 — Prediction ────────────────────────────────────────────────────

    with tab1:
        st.subheader("Risk Level")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_pct,
            number={'suffix': "%"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {
                    'color': "red"    if prob_pct > 70
                        else "orange" if prob_pct > 40
                        else "green"
                },
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

        if prob_pct > 70:
            st.error("High Risk: Phishing Likely")
        elif prob_pct > 40:
            st.warning("Medium Risk: Suspicious")
        else:
            st.success("Low Risk: Likely Legitimate")

        st.write(f"Sender Trust Score: {trust_pct:.1f}%")
        st.divider()
        st.subheader("Confirm Ground Truth")

        col1, col2 = st.columns(2)
        if col1.button("Mark as Legitimate"):
            update_reputation(sender_input, 0)
            st.success("Reputation updated")
        if col2.button("Mark as Phishing"):
            update_reputation(sender_input, 1)
            st.success("Reputation updated")

    # ── TAB 2 — Behavior Analysis ─────────────────────────────────────────────

    with tab2:
        st.subheader("Sender Trust")
        st.progress(float(trust_score))
        st.write(f"Trust Score: {trust_pct:.1f}%")

        st.divider()
        st.subheader("Sender History")
        sender_key = sender_input.strip().lower() if sender_input else None
        if sender_key:
            cursor.execute(
                "SELECT legit_count, phish_count FROM sender_reputation WHERE sender=?",
                (sender_key,)
            )
            row = cursor.fetchone()
            if row:
                legit, phish = row
                st.write(f"Emails seen: {legit + phish}  |  Legit: {legit}  |  Phishing: {phish}")
            else:
                st.write("No history for this sender yet.")
        else:
            st.write("Enter a sender email to see history.")

    # ── TAB 3 — Feature Visualization (radar) ────────────────────────────────

    with tab3:
        labels = ["URLs", "Domains", "IP", "TLD",
                  "Exclaim", "Uppercase", "Urgency", "Trust"]
        values = [
            min(struct_features[0], 5),
            min(struct_features[1], 5),
            struct_features[2] * 5,
            struct_features[3] * 5,
            min(struct_features[4], 5),
            struct_features[5] * 5,
            struct_features[6] * 5,
            trust_score * 5,
        ]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=values, theta=labels, fill='toself'))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
            height=450
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── TAB 4 — Feature Contribution ─────────────────────────────────────────
    # FIX 4: was if hasattr(model,"coef_") — always False for LGB → tab was blank
    # Now shows gain-based importance for the 8 numeric + trust features

    with tab4:
        st.subheader("Numeric & Trust Feature Importance")
        st.caption(
            "LightGBM uses gain-based importance (total information gained by each feature "
            "across all trees) rather than linear coefficients."
        )

        _, numeric_importance = get_feature_importance()

        display_names = {
            "num_urls":           "URLs",
            "num_unique_domains": "Unique Domains",
            "has_ip_url":         "IP URL",
            "suspicious_tld":     "Suspicious TLD",
            "exclamation_count":  "Exclamation Count",
            "uppercase_ratio":    "Uppercase Ratio",
            "urgent_flag":        "Urgency Flag",
            "sender_trust":       "Sender Trust",
        }

        feat_labels = [display_names.get(k, k) for k in numeric_importance]
        feat_gains  = list(numeric_importance.values())
        colors      = ["#f97316" if "Trust" in n else "#34d399" for n in feat_labels]

        fig = go.Figure(go.Bar(
            x=feat_gains,
            y=feat_labels,
            orientation='h',
            marker_color=colors,
        ))
        fig.update_layout(
            height=420,
            xaxis_title="Gain (Information)",
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── TAB 5 — Model Comparison ──────────────────────────────────────────────

    with tab5:
        col1, col2 = st.columns(2)
        col1.metric("Text-only Probability",  f"{text_prob_pct:.1f}%")
        col2.metric("Trust-Aware Probability", f"{prob_pct:.1f}%",
                    delta=f"{prob_pct - text_prob_pct:+.1f}%")

        fig = go.Figure(go.Bar(
            x=["Text Only", "Trust-Aware"],
            y=[text_prob_pct, prob_pct],
            marker_color=["#94a3b8", "#f97316"],
        ))
        fig.update_layout(
            yaxis=dict(range=[0, 100], title="Phishing Probability (%)"),
            height=300,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── TAB 6 — Risk Breakdown ────────────────────────────────────────────────

    with tab6:
        st.subheader("Risk Breakdown")
        st.metric("Text-Based Risk",        f"{text_prob_pct:.1f}%")
        st.metric("Sender Trust Influence", f"{(prob_pct - text_prob_pct):+.1f}%")

        if prob_pct > text_prob_pct:
            st.write("Sender reputation **increased** risk.")
        elif prob_pct < text_prob_pct:
            st.write("Sender reputation **reduced** risk.")
        else:
            st.write("Sender reputation had no effect.")

    # ── TAB 7 — Top Word Signals ──────────────────────────────────────────────
    # FIX 4: was if hasattr(model,"coef_") — always False for LGB → tab was blank
    # Now uses LGB gain importance to show top TF-IDF words globally,
    # and highlights which of those top words appear in this specific email

    with tab7:
        st.subheader("Top Word Signals")
        st.caption(
            "Ranked by information gain across all training trees. "
            "🔴 = this word is present in the submitted email."
        )

        tfidf_importance, _ = get_feature_importance()

        top_words   = sorted(tfidf_importance.items(), key=lambda x: x[1], reverse=True)[:20]
        email_tokens = set(clean_text(email_text).split())

        st.subheader("Top 20 Most Informative Words")
        for word, gain in top_words:
            marker = "🔴" if word in email_tokens else "⬜"
            st.write(f"{marker} **{word}** — gain: {gain:,.0f}")

        st.divider()
        st.caption("🔴 present in this email   ⬜ not present")
