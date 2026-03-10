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

model      = joblib.load("lgb_phishing_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
scaler     = joblib.load("structural_scaler.pkl")

GLOBAL_TRUST_PRIOR = 0.5
ALPHA          = 2.0
AUTO_THRESHOLD = 0.8

# ============================
# SQLITE DATABASE
# ============================

@st.cache_resource
def init_db():
    conn   = sqlite3.connect("sender_reputation.db", check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sender_reputation (
            sender      TEXT PRIMARY KEY,
            legit_count INTEGER DEFAULT 0,
            phish_count INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    return conn, cursor

conn, cursor = init_db()

# ============================
# FEATURE ENGINEERING
# Mirrors notebook Cell 7 + Cell 9 exactly
# ============================

SUSPICIOUS_TLDS = ['.ru', '.cn', '.tk', '.xyz', '.top', '.click']
URGENT_WORDS    = ['urgent', 'immediately', 'action required', 'verify', 'suspended']

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " URL ", text)
    text = re.sub(r"\S+@\S+", " EMAIL ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_structural_features(text):
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
    return [
        num_urls,
        len(set(domains)),
        has_ip,
        suspicious_flag,
        text.count("!"),
        sum(1 for c in text if c.isupper()) / max(len(text), 1),
        int(any(w in text.lower() for w in URGENT_WORDS)),
    ]

# ============================
# TRUST SCORE
# ============================

def get_trust_score(sender):
    if not sender or not sender.strip():
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
# BUG FIX: This now ONLY runs when the human explicitly clicks a button.
# Auto-learning is removed — it was firing on every Analyze click and
# overriding/diluting manual corrections before the user could act.
# ============================

def update_reputation(sender, true_label):
    """
    true_label: 0 = user says Legitimate → increases trust
                1 = user says Phishing   → decreases trust
    """
    if not sender or not sender.strip():
        return
    sender = sender.strip().lower()
    cursor.execute(
        "SELECT legit_count, phish_count FROM sender_reputation WHERE sender=?",
        (sender,)
    )
    row    = cursor.fetchone()
    legit, phish = row if row else (0, 0)
    if true_label == 0:
        legit += 1   # user says legit → trust goes UP
    else:
        phish += 1   # user says phishing → trust goes DOWN
    cursor.execute(
        "INSERT OR REPLACE INTO sender_reputation (sender, legit_count, phish_count) VALUES (?, ?, ?)",
        (sender, legit, phish)
    )
    conn.commit()

# ============================
# BUILD FEATURE MATRIX
# Mirrors notebook Cell 14: hstack([X_text(3000), struct_scaled(7), trust(1)]) = 3008
# ============================

def build_feature_matrix(email_text, trust_score):
    cleaned       = clean_text(email_text)
    X_text        = vectorizer.transform([cleaned])
    struct        = extract_structural_features(email_text)
    struct_scaled = scaler.transform(np.array(struct).reshape(1, -1))
    trust_array   = np.array([[trust_score]])
    X_numeric     = np.hstack([struct_scaled, trust_array])
    X_final       = hstack([X_text, csr_matrix(X_numeric)]).tocsr()
    return X_final, X_text, struct

# ============================
# LGB FEATURE IMPORTANCE (cached)
# ============================

@st.cache_resource
def get_feature_importance():
    gain  = model.booster_.feature_importance(importance_type='gain')
    names = model.feature_name_
    n_tfidf = len(vectorizer.get_feature_names_out())
    tfidf_importance   = {names[i]: gain[i] for i in range(n_tfidf)}
    numeric_importance = {names[i]: gain[i] for i in range(n_tfidf, len(names))}
    return tfidf_importance, numeric_importance

# ============================
# SESSION STATE
# BUG FIX: Store analysis results in session_state so they persist across
# button clicks (Streamlit reruns entire script on every interaction).
# Without this, clicking "Mark as Phishing" wipes out all the tab content.
# ============================

if "results" not in st.session_state:
    st.session_state.results = None
if "reputation_msg" not in st.session_state:
    st.session_state.reputation_msg = None

# ============================
# UI
# ============================

st.set_page_config(page_title="Sender-Trust Phishing Detection", layout="wide")
st.title("Sender-Trust Aware Phishing Detection")
st.caption("LightGBM · TF-IDF + Structural Features + Dynamic Sender Trust")

sender_input = st.text_input("Sender Email",
                              value=st.session_state.get("sender_input", ""))
email_text   = st.text_area("Email Content", height=220,
                             value=st.session_state.get("email_text", ""))

if st.button("Analyze Email"):
    if not email_text.strip():
        st.warning("Please enter email content.")
        st.stop()

    # Save inputs so they persist after button clicks
    st.session_state.sender_input = sender_input
    st.session_state.email_text   = email_text
    st.session_state.reputation_msg = None  # clear old message

    trust_score = get_trust_score(sender_input)
    trust_pct   = trust_score * 100

    X_final, X_text, struct_features = build_feature_matrix(email_text, trust_score)
    X_text_only, _, _                = build_feature_matrix(email_text, 0.0)

    probability   = model.predict_proba(X_final)[0][1]
    prob_pct      = probability * 100
    text_prob_pct = model.predict_proba(X_text_only)[0][1] * 100

    # ── Per-email word contributions via TF-IDF weight × global gain ──────────
    # SHAP on 3008 features is too slow for a live app (~30s).
    # Instead: for each word present in this email, multiply its TF-IDF weight
    # by the LGB gain for that feature. This gives a fast, meaningful per-email
    # signal: words the model finds important AND that actually appear here.
    tfidf_importance, _ = get_feature_importance()
    tfidf_names  = vectorizer.get_feature_names_out()
    tfidf_vector = X_text.toarray().flatten()   # (3000,) weights for this email

    word_scores = []
    for idx, weight in enumerate(tfidf_vector):
        if weight > 0:
            word  = tfidf_names[idx]
            gain  = tfidf_importance.get(word, 0)
            score = float(weight * gain)   # higher = more influential in this email
            word_scores.append((word, score, weight, gain))

    # Sort by score descending — top = most phishing-influential in this email
    word_scores.sort(key=lambda x: x[1], reverse=True)

    # Store all results — persists across reruns caused by button clicks
    st.session_state.results = {
        "sender_input":    sender_input,
        "trust_score":     trust_score,
        "trust_pct":       trust_pct,
        "probability":     probability,
        "prob_pct":        prob_pct,
        "text_prob_pct":   text_prob_pct,
        "struct_features": struct_features,
        "word_scores":     word_scores,
    }

# ── Show results if we have them ─────────────────────────────────────────────

if st.session_state.results:
    r               = st.session_state.results
    prob_pct        = r["prob_pct"]
    text_prob_pct   = r["text_prob_pct"]
    trust_score     = r["trust_score"]
    trust_pct       = r["trust_pct"]
    struct_features = r["struct_features"]
    stored_sender   = r["sender_input"]
    word_scores     = r.get("word_scores", [])

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

        st.write(f"**Sender Trust Score:** {trust_pct:.1f}%")
        st.divider()

        # ── Manual ground truth buttons ───────────────────────────────────────
        # BUG FIX: Trust update is 100% user-controlled.
        # Legitimate → legit_count++ → trust score rises next analysis
        # Phishing   → phish_count++ → trust score falls next analysis
        # The update writes to SQLite immediately and persists across sessions.

        st.subheader("Confirm Ground Truth")
        st.caption("Your feedback updates this sender's trust score for future emails.")

        col1, col2 = st.columns(2)

        if col1.button(" Mark as Legitimate", use_container_width=True):
            update_reputation(stored_sender, 0)   # 0 = legit → trust UP
            new_trust = get_trust_score(stored_sender)
            st.session_state.results["trust_score"] = new_trust
            st.session_state.results["trust_pct"]   = new_trust * 100
            st.session_state.reputation_msg = ("success",
                f" Marked as Legitimate. Trust updated: {new_trust*100:.1f}%")

        if col2.button(" Mark as Phishing", use_container_width=True):
            update_reputation(stored_sender, 1)   # 1 = phishing → trust DOWN
            new_trust = get_trust_score(stored_sender)
            st.session_state.results["trust_score"] = new_trust
            st.session_state.results["trust_pct"]   = new_trust * 100
            st.session_state.reputation_msg = ("error",
                f" Marked as Phishing. Trust updated: {new_trust*100:.1f}%")

        if st.session_state.reputation_msg:
            msg_type, msg_text = st.session_state.reputation_msg
            if msg_type == "success":
                st.success(msg_text)
            else:
                st.error(msg_text)

    # ── TAB 2 — Behavior Analysis ─────────────────────────────────────────────

    with tab2:
        st.subheader("Sender Trust")

        # Refresh trust from DB in case user just updated it
        live_trust = get_trust_score(stored_sender)
        st.progress(float(live_trust))
        st.write(f"**Trust Score:** {live_trust*100:.1f}%")

        st.divider()
        st.subheader("Sender History")

        if stored_sender and stored_sender.strip():
            cursor.execute(
                "SELECT legit_count, phish_count FROM sender_reputation WHERE sender=?",
                (stored_sender.strip().lower(),)
            )
            row = cursor.fetchone()
            if row:
                legit, phish = row
                total = legit + phish
                st.write(f"Emails reviewed: **{total}**")
                c1, c2, c3 = st.columns(3)
                c1.metric("Total", total)
                c2.metric("Marked Legit", legit)
                c3.metric("Marked Phishing", phish)

                # Show how trust changed
                trust_formula = (legit + ALPHA) / (legit + phish + 2 * ALPHA)
                st.caption(f"Trust = (legit + {int(ALPHA)}) / (total + {int(2*ALPHA)}) "
                           f"= ({legit} + {int(ALPHA)}) / ({total} + {int(2*ALPHA)}) "
                           f"= **{trust_formula*100:.1f}%**")
            else:
                st.info("No manual feedback recorded for this sender yet.")
        else:
            st.info("Enter a sender email to see their history.")

    # ── TAB 3 — Feature Visualization (radar) ────────────────────────────────

    with tab3:
        labels = ["URLs", "Domains", "IP", "Susp. TLD",
                  "Exclamation", "Uppercase", "Urgency", "Trust"]
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
        fig.add_trace(go.Scatterpolar(r=values, theta=labels, fill='toself',
                                      line_color="#f97316"))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
            height=450
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── TAB 4 — Feature Contribution ──────────────────────────────────────────

    with tab4:
        st.subheader("Numeric & Trust Feature Importance")
        st.caption("Information gain each feature contributed across all training trees.")

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
            x=feat_gains, y=feat_labels, orientation='h', marker_color=colors,
        ))
        fig.update_layout(height=420, xaxis_title="Gain",
                          yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    # ── TAB 5 — Model Comparison ──────────────────────────────────────────────

    with tab5:
        c1, c2 = st.columns(2)
        c1.metric("Text-only Probability",  f"{text_prob_pct:.1f}%")
        c2.metric("Trust-Aware Probability", f"{prob_pct:.1f}%",
                  delta=f"{prob_pct - text_prob_pct:+.1f}%")

        fig = go.Figure(go.Bar(
            x=["Text Only", "Trust-Aware"],
            y=[text_prob_pct, prob_pct],
            marker_color=["#94a3b8", "#f97316"],
        ))
        fig.update_layout(yaxis=dict(range=[0, 100], title="Phishing Probability (%)"),
                          height=300)
        st.plotly_chart(fig, use_container_width=True)

    # ── TAB 6 — Risk Breakdown ────────────────────────────────────────────────

    with tab6:
        st.subheader("Risk Breakdown")
        st.metric("Text-Based Risk",        f"{text_prob_pct:.1f}%")
        st.metric("Sender Trust Influence", f"{(prob_pct - text_prob_pct):+.1f}%")

        delta = prob_pct - text_prob_pct
        if delta > 0.5:
            st.write(" Sender reputation **increased** phishing risk.")
        elif delta < -0.5:
            st.write(" Sender reputation **reduced** phishing risk.")
        else:
            st.write(" Sender reputation had minimal effect.")

    # ── TAB 7 — Top Word Signals ──────────────────────────────────────────────
    # Shows which words in THIS specific email are driving the prediction.
    # Score = TF-IDF weight × LGB gain. High score = word is both prominent
    # in this email AND the model considers it highly informative.

    with tab7:
        st.subheader("Why Did This Email Score That Way?")
        st.caption(
            "Words from **this email** ranked by influence on the prediction. "
            "Score = how prominently the word appears × how informative the model finds it."
        )

        if not word_scores:
            st.info("No scoreable words found — email may be too short or contain no recognised vocabulary.")
        else:
            top_phishing = word_scores[:10]
            # Bottom of the list = words that appear but have low/zero model gain
            # Filter to only words that actually have some gain
            scored = [(w, s, wt, g) for w, s, wt, g in word_scores if g > 0]
            top_safe = scored[-8:] if len(scored) >= 8 else scored

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("####  Pushing Toward Phishing")
                st.caption("High-gain words the model associates with phishing that appear in this email.")

                if top_phishing:
                    words_p  = [w for w,s,wt,g in top_phishing]
                    scores_p = [round(s, 1) for w,s,wt,g in top_phishing]

                    fig = go.Figure(go.Bar(
                        x=scores_p,
                        y=words_p,
                        orientation='h',
                        marker_color='rgba(239,68,68,0.7)',
                        marker_line_color='#ef4444',
                        marker_line_width=1,
                    ))
                    fig.update_layout(
                        height=320,
                        xaxis_title="Influence Score",
                        yaxis=dict(autorange="reversed"),
                        margin=dict(l=0, r=10, t=10, b=30),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#94a3b8'),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No strong phishing word signals found.")

            with col2:
                st.markdown("####  Pushing Toward Legitimate")
                st.caption("Words present in this email that the model associates with legitimate emails.")

                # Legitimate words = present in email but model gain is LOW
                # (model doesn't flag them as phishing signals)
                safe_words = [(w, s, wt, g) for w, s, wt, g in word_scores
                              if wt > 0 and g < 5000]  # low-gain = not a phishing signal
                safe_words.sort(key=lambda x: x[2], reverse=True)  # sort by TF-IDF weight
                safe_top = safe_words[:10]

                if safe_top:
                    words_s  = [w for w,s,wt,g in safe_top]
                    weights_s = [round(wt * 100, 2) for w,s,wt,g in safe_top]

                    fig2 = go.Figure(go.Bar(
                        x=weights_s,
                        y=words_s,
                        orientation='h',
                        marker_color='rgba(34,197,94,0.7)',
                        marker_line_color='#22c55e',
                        marker_line_width=1,
                    ))
                    fig2.update_layout(
                        height=320,
                        xaxis_title="TF-IDF Weight (×100)",
                        yaxis=dict(autorange="reversed"),
                        margin=dict(l=0, r=10, t=10, b=30),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#94a3b8'),
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.info("No strong legitimate word signals found.")

            st.divider()
            st.caption(
                f"**{len(word_scores)}** words from this email matched the model vocabulary. "
                "Phishing score = TF-IDF weight × training gain. "
                "Legitimate score = words present but not flagged as phishing signals by the model."
            )
