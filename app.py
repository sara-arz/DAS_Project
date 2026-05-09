import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from matplotlib.patches import Rectangle
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings("ignore")

DATA_RAW = "cybersecurity.csv"
DATA_PREP = "cybersecurity_preprocessed.csv"

st.set_page_config(
    page_title="Cybersecurity Traffic Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0d1117;
    color: #e6edf3;
    font-family: 'Segoe UI', system-ui, sans-serif;
}
[data-testid="stSidebar"] { background-color: #161b22; }
#MainMenu, footer, header { visibility: hidden; }
h1, h2, h3 { color: #e6edf3 !important; }
</style>
""",
    unsafe_allow_html=True,
)

# ── palette & helpers ─────────────────────────────────────────────────────────

P = {
    "green": "#238636",
    "blue": "#388bfd",
    "red": "#f85149",
    "orange": "#e3b341",
    "purple": "#bc8cff",
    "muted": "#8b949e",
    "bg": "#0d1117",
    "card": "#161b22",
    "border": "#30363d",
}


def fig1(w=9, h=4):
    """Single-axes dark figure."""
    fig, ax = plt.subplots(figsize=(w, h))
    _style_fig(fig, [ax])
    return fig, ax


def figs(rows, cols, w=12, h=4):
    """Multi-axes dark figure; always returns a flat ndarray of axes."""
    fig, axes = plt.subplots(rows, cols, figsize=(w, h))
    axlist = np.array(axes).flatten()
    _style_fig(fig, axlist)
    return fig, axlist


def _style_fig(fig, axlist):
    fig.patch.set_facecolor(P["card"])
    for ax in axlist:
        ax.set_facecolor(P["bg"])
        ax.tick_params(colors=P["muted"])
        ax.xaxis.label.set_color(P["muted"])
        ax.yaxis.label.set_color(P["muted"])
        ax.title.set_color(P["green"])
        for sp in ax.spines.values():
            sp.set_color(P["card"])
        ax.grid(color="#21262d", linewidth=0.6)


def show(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ── data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_raw():
    try:
        return pd.read_csv(DATA_RAW)
    except FileNotFoundError:
        return None


@st.cache_data
def load_prep():
    try:
        return pd.read_csv(DATA_PREP)
    except FileNotFoundError:
        return None


@st.cache_data
def run_models():
    df = load_prep()
    if df is None:
        return None
    FEATURES = [
        "bytes_sent",
        "bytes_received",
        "total_bytes",
        "bytes_ratio",
        "port_diff",
        "src_port",
        "dst_port",
        "dst_port_common",
        "src_port_common",
        "is_internal_traffic",
        "has_url",
    ]
    X, y = df[FEATURES], df["label"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr)
    X_te_sc = sc.transform(X_te)
    models = {
        "OLS": LinearRegression(),
        "Ridge (α=1.0)": Ridge(alpha=1.0),
        "Lasso (α=0.001)": Lasso(alpha=0.001, max_iter=5000),
    }
    results, coefs, preds = {}, {}, {}
    for name, m in models.items():
        m.fit(X_tr_sc, y_tr)
        yp = m.predict(X_te_sc)
        preds[name] = yp
        results[name] = {
            "MAE": round(mean_absolute_error(y_te, yp), 4),
            "RMSE": round(np.sqrt(mean_squared_error(y_te, yp)), 4),
            "R²": round(r2_score(y_te, yp), 4),
        }
        coefs[name] = dict(zip(FEATURES, m.coef_.round(4)))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv = cross_val_score(LinearRegression(), X_tr_sc, y_tr, cv=skf, scoring="r2")
    train_r2 = r2_score(y_tr, models["OLS"].predict(X_tr_sc))
    return {
        "results": results,
        "coefs": coefs,
        "preds": preds,
        "y_test": y_te.values,
        "features": FEATURES,
        "cv": cv,
        "ols_train_r2": train_r2,
    }


df_raw = load_raw()
df_prep = load_prep()
model_data = run_models()

# ── sidebar nav ───────────────────────────────────────────────────────────────

st.sidebar.title("CyberTraffic")
st.sidebar.caption("Network Analysis Dashboard")
st.sidebar.markdown("---")

pages = {
    "Overview": "overview",
    "Data Collection": "collection",
    "Preprocessing": "preprocessing",
    "EDA": "eda",
    "Modeling": "modeling",
    "Conclusion": "conclusion",
}
page_id = pages[st.sidebar.radio("", list(pages.keys()))]

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Dataset**  
10,000 network records  
13 raw → 23 engineered features  
4% attack rate
 
**Models**  
OLS · Ridge · Lasso  
Target: `label` (0/1)
""")

# ═════════════════════════════════════════════════════════════════════════════
# OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
if page_id == "overview":
    st.title("Cybersecurity Network Traffic Analysis")
    st.caption("Collection → Preprocessing → EDA → Modeling → Deployment")
    st.markdown("---")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("Pipeline")
        steps = [
            (
                "1 - Data Collection",
                "10K simulated records, 13 raw features: ports, bytes, protocol, IPs, labels.",
            ),
            (
                "2 - Preprocessing",
                "Missing values flagged, one-hot encoding, datetime parsing, column drops.",
            ),
            (
                "3 - EDA",
                "Distributions, class separation, binary attack rates, temporal patterns, correlation heatmap.",
            ),
            (
                "4 - Modeling",
                "OLS, Ridge, Lasso. 80/20 split, StandardScaler, 5-fold CV.",
            ),
        ]
        for title, desc in steps:
            st.markdown(f"**{title}**  \n{desc}")
            st.markdown("")

    with col2:
        st.subheader("Class Distribution")
        fig, ax = fig1(4, 4)
        result = ax.pie(
            [9600, 400],
            labels=["Benign (96%)", "Attack (4%)"],
            colors=[P["blue"], P["red"]],
            autopct="%1.1f%%",
            startangle=140,
            wedgeprops=dict(width=0.55, edgecolor=P["card"]),
        )
        wedges, texts = result[0], result[1]
        autotexts = result[2] if len(result) > 2 else []
        for t in texts:
            t.set_color(P["muted"])
        for at in autotexts:
            at.set_color("white")
            at.set_fontsize(11)
        ax.set_title("Label Distribution")
        show(fig)

# ═════════════════════════════════════════════════════════════════════════════
# DATA COLLECTION
# ═════════════════════════════════════════════════════════════════════════════
elif page_id == "collection":
    st.title("Data Collection")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records", "10,000")
    c2.metric("Raw Features", "13")
    c3.metric("Attack Types", "5")
    c4.metric("Protocols", "3")

    st.markdown("---")
    st.subheader("Dataset Preview")
    if df_raw is not None:
        st.dataframe(
            df_raw.head(10).style.map(
                lambda v: "color: #f85149; font-weight:bold" if v == 1 else "",
                subset=["label"],
            ),
            use_container_width=True,
        )
    else:
        st.warning(f"`{DATA_RAW}` not found — place it alongside app.py")

    st.markdown("---")
    st.subheader("Feature Schema")
    schema = pd.DataFrame(
        {
            "Feature": [
                "timestamp",
                "src_ip",
                "dst_ip",
                "src_port",
                "dst_port",
                "protocol",
                "bytes_sent",
                "bytes_received",
                "user_agent",
                "url",
                "is_internal_traffic",
                "label",
                "attack_type",
            ],
            "Type": [
                "datetime",
                "string",
                "string",
                "int",
                "int",
                "categorical",
                "int",
                "int",
                "string",
                "string",
                "bool",
                "binary",
                "categorical",
            ],
            "Notes": [
                "Connection start",
                "Source IP (dropped)",
                "Dest IP (dropped)",
                "0–65535",
                "0–65535",
                "TCP / UDP / ICMP",
                "Bytes sent",
                "Bytes received",
                "HTTP user agent (dropped)",
                "URL if HTTP (sparse)",
                "Internal network flag",
                "0=Benign  1=Attack  ← TARGET",
                "benign/ddos/injection/intrusion/ransomware",
            ],
        }
    )
    st.dataframe(schema, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Attack Type Breakdown")
    if df_raw is not None:
        fig, ax = fig1(5, 4)
        vc = df_raw["attack_type"].value_counts()
        colors = [P["green"] if v == "benign" else P["red"] for v in vc.index]
        bars = ax.barh(
            vc.index, np.asarray(vc.values), color=colors, edgecolor=P["card"]
        )
        ax.set_xlabel("Count")
        ax.set_title("Records per Attack Type")
        for bar, val in zip(bars, vc.values):
            ax.text(
                val + 50,
                bar.get_y() + bar.get_height() / 2,
                f"{val:,}",
                va="center",
                color=P["muted"],
                fontsize=9,
            )
        plt.tight_layout()
        show(fig)
    st.markdown("---")
    st.subheader("Protocol Distribution")
    if df_raw is not None:
        fig, ax = fig1(5, 3)
        proto = df_raw["protocol"].str.upper().value_counts()
        ax.bar(
            proto.index,
            np.asarray(proto.values),
            color=[P["blue"], P["purple"], P["orange"]],
            edgecolor=P["card"],
        )
        ax.set_title("Protocol Counts")
        plt.tight_layout()
        show(fig)

# ═════════════════════════════════════════════════════════════════════════════
# PREPROCESSING
# ═════════════════════════════════════════════════════════════════════════════
elif page_id == "preprocessing":
    st.title(" Preprocessing & Feature Engineering")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Missing values (after)", "0")
    c2.metric("Duplicate rows", "0")
    c3.metric("Inconsistencies", "0")
    c4.metric("Columns after engineering", "23")

    st.markdown("---")
    tab1, tab2 = st.tabs(["Preprocessing Steps", "Feature Engineering"])

    with tab1:
        steps = [
            (
                "Missing Values",
                "The `url` column had NaN where no HTTP traffic was present. A binary `has_url` flag preserves the signal without dropping rows or meaningless imputation.",
            ),
            (
                "String Normalization",
                "`protocol` had mixed casing (tcp / TCP / ' TCP'). All values were stripped and uppercased to prevent duplicated categories.",
            ),
            (
                "Type Fixing",
                "`timestamp` → datetime for feature extraction. `is_internal_traffic` (bool) → int. `protocol` one-hot encoded into 3 binary columns.",
            ),
            (
                "Consistency Checks",
                "Verified label=0 ↔ attack_type='benign' with 0 mismatches. No negative byte values or port anomalies found.",
            ),
            (
                "Column Removal",
                "Dropped: `timestamp` (decomposed), `src_ip`, `dst_ip` (high cardinality), `user_agent`, `url` (captured via has_url), `attack_type` (target leakage).",
            ),
        ]
        for i, (title, body) in enumerate(steps, 1):
            st.markdown(f"**{i}. {title}**")
            st.markdown(body)
            st.markdown("")

        st.subheader("Preprocessed Dataset Preview")
        if df_prep is not None:
            st.dataframe(df_prep.head(8), use_container_width=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Rows", f"{df_prep.shape[0]:,}")
            c2.metric("Columns", df_prep.shape[1])
            c3.metric("Missing values", df_prep.isnull().sum().sum())

    with tab2:
        st.subheader("Temporal Features")
        st.markdown("""
- `hour` — hour of day (0–23)  
- `day_of_week` — 0=Mon … 6=Sun  
- `is_weekend` — 1 if Sat or Sun  
- `is_night` — 1 if hour in [22..5]

*Attack patterns vary by time of day — attackers exploit off-hours.*
            """)
        st.subheader("Byte Features")
        st.markdown("""
- `total_bytes` = bytes_sent + bytes_received  
- `bytes_ratio` = bytes_sent / (bytes_received + 1)

*DDoS and exfiltration create abnormally large or asymmetric byte volumes.*
            """)
        st.subheader("Port Features")
        st.markdown("""
- `port_diff` = |src_port − dst_port|  
- `src/dst_port_is_well_known` — port < 1024  
- `src/dst_port_is_ephemeral` — port ≥ 49152  
- `dst_port_common` — targets SSH/RDP/HTTP/SMB…  
- `src_port_common` — source is a known service port

*Port numbers carry structural meaning — attackers favor specific ranges.*
            """)
        st.markdown("---")
        st.subheader("Before vs After")
        ba = pd.DataFrame(
            {
                "": ["Rows", "Columns", "Missing", "Numeric cols", "Text cols"],
                "Before": ["10,000", "13", "~6,800 (url)", "7", "6"],
                "After": ["10,000", "23", "0", "23", "0"],
            }
        )
        st.dataframe(ba, use_container_width=True, hide_index=True)

# ═════════════════════════════════════════════════════════════════════════════
# EDA
# ═════════════════════════════════════════════════════════════════════════════
elif page_id == "eda":
    st.title(" Exploratory Data Analysis")
    st.markdown("---")

    if df_prep is None:
        st.error(f"`{DATA_PREP}` not found.")
        st.stop()

    df = df_prep.copy()
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Distributions", "Class Separation", "Temporal", "Correlation"]
    )

    with tab1:
        st.subheader("Continuous Feature Distributions")
        st.info(
            "Byte features are heavily right-skewed. Log-transform reveals bell-shaped structure."
        )

        cont_feats = [
            "bytes_sent",
            "bytes_received",
            "total_bytes",
            "bytes_ratio",
            "port_diff",
            "src_port",
            "dst_port",
        ]
        sel = st.selectbox("Feature", cont_feats)

        fig, axes = figs(1, 2, w=12, h=4)
        axes[0].hist(
            np.asarray(df[sel]),
            bins=50,
            color=P["blue"],
            edgecolor=P["card"],
            alpha=0.85,
        )
        axes[0].set_title(f"{sel} — raw")
        axes[0].set_xlabel("Value")
        axes[0].set_ylabel("Count")
        axes[1].hist(
            np.log1p(df[sel].values),
            bins=50,
            color=P["green"],
            edgecolor=P["card"],
            alpha=0.85,
        )
        axes[1].set_title(f"{sel} — log1p")
        axes[1].set_xlabel("log1p(Value)")
        plt.tight_layout()
        show(fig)

        c1, c2 = st.columns(2)
        c1.metric("Skewness (raw)", f"{df[sel].skew():.2f}")
        c2.metric("Skewness (log)", f"{pd.Series(np.log1p(df[sel])).skew():.2f}")

    with tab2:
        st.subheader("Feature Distribution by Class")
        st.warning(
            "Attack traffic produces more extreme byte outliers — consistent with flooding or exfiltration."
        )

        cont_feats2 = [
            "bytes_sent",
            "bytes_received",
            "total_bytes",
            "bytes_ratio",
            "port_diff",
            "src_port",
            "dst_port",
        ]
        sel2 = st.selectbox("Feature", cont_feats2, key="cls_feat")

        fig, ax = fig1(8, 4)
        benign = df[df["label"] == 0][sel2]
        attack = df[df["label"] == 1][sel2]
        ax.boxplot(
            [np.asarray(benign), np.asarray(attack)],
            tick_labels=["Benign (0)", "Attack (1)"],
            patch_artist=True,
            boxprops=dict(facecolor=P["card"], color=P["blue"]),
            medianprops=dict(color=P["green"], linewidth=2),
            whiskerprops=dict(color=P["muted"]),
            flierprops=dict(marker=".", color=P["red"], alpha=0.3, markersize=3),
        )
        ax.set_title(f"{sel2} — by class")
        ax.set_ylabel(sel2)
        plt.tight_layout()
        show(fig)

        c1, c2 = st.columns(2)
        c1.metric("Benign median", f"{benign.median():,.0f}")
        c2.metric("Attack median", f"{attack.median():,.0f}")

        st.subheader("Binary Feature Attack Rates")
        bin_feats = [
            "has_url",
            "is_internal_traffic",
            "is_weekend",
            "is_night",
            "src_port_is_well_known",
            "dst_port_is_well_known",
            "dst_port_common",
            "src_port_common",
            "protocol_TCP",
            "protocol_UDP",
            "protocol_ICMP",
        ]
        bin_feats = [f for f in bin_feats if f in df.columns]

        rates_0 = [df[df[f] == 0]["label"].mean() for f in bin_feats]
        rates_1 = [df[df[f] == 1]["label"].mean() for f in bin_feats]
        x = np.arange(len(bin_feats))
        w = 0.35

        fig, ax = fig1(12, 4)
        ax.bar(x - w / 2, rates_0, w, label="Value=0", color=P["blue"], alpha=0.85)
        ax.bar(x + w / 2, rates_1, w, label="Value=1", color=P["orange"], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(bin_feats, rotation=35, ha="right", fontsize=9)
        ax.set_ylabel("Attack rate")
        ax.set_title("Attack Rate by Binary Feature Value")
        ax.legend(facecolor=P["card"], edgecolor=P["border"], labelcolor=P["muted"])
        plt.tight_layout()
        show(fig)

        st.error(
            "🔑 **dst_port_common** shows the largest gap — uncommon destination port → higher attack rate."
        )

    with tab3:
        st.subheader("Attack Rate by Hour of Day")
        hourly = df.groupby("hour")["label"].mean()
        fig, ax = fig1(10, 4)
        ax.plot(
            hourly.index,
            np.asarray(hourly.values),
            color=P["green"],
            linewidth=2,
            marker="o",
            markersize=5,
        )
        ax.fill_between(
            hourly.index, np.asarray(hourly.values), alpha=0.12, color=P["green"]
        )
        ax.axhline(
            df["label"].mean(),
            color=P["red"],
            linestyle="--",
            linewidth=1.2,
            label="Overall avg",
        )
        ax.set_xlabel("Hour")
        ax.set_ylabel("Attack rate")
        ax.set_title("Attack Rate by Hour of Day")
        ax.set_xticks(range(24))
        ax.legend(facecolor=P["card"], edgecolor=P["border"])
        plt.tight_layout()
        show(fig)

        st.subheader("Attack Rate by Day of Week")
        daily = df.groupby("day_of_week")["label"].mean()
        day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][: len(daily)]
        fig, ax = fig1(8, 3)
        ax.bar(
            day_labels,
            np.asarray(daily.values),
            color=[P["red"] if d >= 5 else P["blue"] for d in daily.index],
            edgecolor=P["card"],
        )
        ax.axhline(
            df["label"].mean(),
            color=P["orange"],
            linestyle="--",
            linewidth=1.2,
            label="Overall avg",
        )
        ax.set_title("Attack Rate by Day of Week")
        ax.legend(facecolor=P["card"], edgecolor=P["border"])
        plt.tight_layout()
        show(fig)

        st.warning(
            "No strong temporal pattern — attack rate fluctuates 2.5%–5.5% with no consistent trend. Temporal features add little predictive value."
        )

    with tab4:
        st.subheader("Correlation with Target")
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        corr_target = (
            df[num_cols]
            .corr()["label"]
            .drop("label")
            .sort_values(key=abs, ascending=False)
        )

        fig, ax = fig1(10, 5)
        colors = [P["red"] if v < 0 else P["green"] for v in corr_target.values]
        ax.barh(
            corr_target.index,
            np.asarray(corr_target.values),
            color=colors,
            edgecolor=P["card"],
        )
        ax.axvline(0, color=P["muted"], linewidth=0.8, linestyle="--")
        ax.set_title("Feature Correlation with Label")
        ax.set_xlabel("Pearson r")
        plt.tight_layout()
        show(fig)

        st.error(
            "Max |r| ≈ 0.17 — all correlations are weak. The attack boundary is non-linear; linear regression explains only ~17% of variance."
        )

        st.subheader("Correlation Heatmap")
        feat_cols = [
            "bytes_sent",
            "bytes_received",
            "total_bytes",
            "bytes_ratio",
            "port_diff",
            "src_port",
            "dst_port",
            "dst_port_common",
            "src_port_common",
            "is_internal_traffic",
            "has_url",
            "label",
        ]
        feat_cols = [f for f in feat_cols if f in df.columns]
        corr = df[feat_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 7))
        fig.patch.set_facecolor(P["card"])
        ax.set_facecolor(P["card"])
        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="RdYlGn",
            center=0,
            linewidths=0.5,
            ax=ax,
            annot_kws={"size": 7},
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title("Correlation Heatmap", color=P["green"], fontsize=11)
        ax.tick_params(colors=P["muted"])
        plt.xticks(rotation=35, ha="right", fontsize=8, color=P["muted"])
        plt.yticks(fontsize=8, color=P["muted"])
        plt.tight_layout()
        show(fig)

# ═════════════════════════════════════════════════════════════════════════════
# MODELING
# ═════════════════════════════════════════════════════════════════════════════
elif page_id == "modeling":
    st.title(" Linear Regression Modeling")
    st.markdown("---")

    if model_data is None:
        st.error(f"`{DATA_PREP}` not found.")
        st.stop()

    res = model_data["results"]
    coefs = model_data["coefs"]
    preds = model_data["preds"]
    y_te = model_data["y_test"]
    feats = model_data["features"]
    cv = model_data["cv"]
    train_r2 = model_data["ols_train_r2"]

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Setup", "Coefficients", "Performance", "Diagnostics"]
    )

    with tab1:
        st.subheader("Model Equation")
        st.code(
            "ŷ = β₀ + β₁(bytes_sent) + β₂(bytes_received) + … + β₁₁(has_url)",
            language=None,
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Train / Test Split")
            st.markdown(
                "80/20 split · random_state=42  \n8,000 training | 2,000 test  \nAttack rate preserved in both sets."
            )
        with col2:
            st.subheader("Feature Scaling")
            st.markdown(
                "StandardScaler fit on training set only to prevent data leakage.  \nAll features → mean=0, std=1."
            )
        with col3:
            st.subheader("Features (11)")
            st.markdown(
                "`bytes_sent` `bytes_received` `total_bytes` `bytes_ratio` `port_diff` `src_port` `dst_port` `dst_port_common` `src_port_common` `is_internal_traffic` `has_url`"
            )

        st.markdown("---")
        st.subheader("Models")
        model_info = [
            (
                "OLS",
                "LinearRegression()",
                "Ordinary Least Squares — minimises sum of squared residuals.",
            ),
            (
                "Ridge",
                "Ridge(alpha=1.0)",
                "L2 regularisation — shrinks all coefficients. Handles multicollinearity, keeps all features.",
            ),
            (
                "Lasso",
                "Lasso(alpha=0.001, max_iter=5000)",
                "L1 regularisation — can zero out coefficients, performing automatic feature selection.",
            ),
        ]
        for name, code, desc in model_info:
            st.markdown(f"**{name}** — `{code}`  \n{desc}")
            st.markdown("")

    with tab2:
        st.subheader("Feature Coefficients")
        model_sel = st.selectbox("Model", list(coefs.keys()))
        c = coefs[model_sel]

        coef_s = pd.Series(c).sort_values()
        fig, ax = fig1(10, 5)
        colors = [P["red"] if v < 0 else P["green"] for v in coef_s.values]
        bars = ax.barh(
            coef_s.index, np.asarray(coef_s.values), color=colors, edgecolor=P["card"]
        )
        ax.axvline(0, color=P["muted"], linewidth=0.8, linestyle="--")
        ax.set_title(f"{model_sel} — Coefficients (standardised features)")
        ax.set_xlabel("Coefficient value (Δ attack score per σ of feature)")
        for bar, val in zip(bars, coef_s.values):
            if val != 0:
                ax.text(
                    val + (0.001 if val >= 0 else -0.001),
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}",
                    va="center",
                    ha="left" if val >= 0 else "right",
                    fontsize=8.5,
                    color=P["muted"],
                )
        plt.tight_layout()
        show(fig)

        zeroed = [f for f, v in c.items() if v == 0]
        if zeroed:
            st.warning(f"Lasso zeroed out: {', '.join(zeroed)}")

        st.subheader("Coefficient Table")
        interp = {
            "dst_port_common": (
                "negative",
                "Uncommon ports → higher attack probability.",
            ),
            "bytes_sent": (
                "positive",
                "Higher bytes sent → exfiltration/flood signal.",
            ),
            "src_port_common": (
                "positive",
                "Connections from known service ports slightly attack-like.",
            ),
            "is_internal_traffic": (
                "negative",
                "Internal traffic less likely to be an attack.",
            ),
            "bytes_ratio": (
                "near zero",
                "Sent/received ratio carries little linear signal.",
            ),
        }
        coef_table = pd.DataFrame(
            [
                {
                    "Feature": f,
                    "Coefficient": f"{c.get(f, 0):.4f}",
                    "Direction": interp[f][0] if f in interp else "—",
                    "Interpretation": interp[f][1] if f in interp else "—",
                }
                for f in feats
            ]
        )
        st.dataframe(coef_table, use_container_width=True, hide_index=True)

    with tab3:
        st.subheader("Metrics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("OLS R²", f"{res['OLS']['R²']:.4f}")
        c2.metric("Ridge R²", f"{res['Ridge (α=1.0)']['R²']:.4f}")
        c3.metric("Lasso R²", f"{res['Lasso (α=0.001)']['R²']:.4f}")
        c4.metric("5-fold CV R²", f"{cv.mean():.3f} ± {cv.std():.3f}")

        st.markdown("---")
        results_df = (
            pd.DataFrame(res).T.reset_index().rename(columns={"index": "Model"})
        )
        st.dataframe(
            results_df.style.highlight_max(
                subset=["R²"], color="#1f3d1f"
            ).highlight_min(subset=["MAE", "RMSE"], color="#1f3d1f"),
            use_container_width=True,
            hide_index=True,
        )

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = fig1(5, 4)
            models = list(res.keys())
            x = np.arange(len(models))
            w = 0.35
            ax.bar(
                x - w / 2,
                [res[m]["MAE"] for m in models],
                w,
                label="MAE",
                color=P["red"],
                edgecolor=P["card"],
            )
            ax.bar(
                x + w / 2,
                [res[m]["RMSE"] for m in models],
                w,
                label="RMSE",
                color=P["blue"],
                edgecolor=P["card"],
            )
            ax.set_xticks(x)
            ax.set_xticklabels(models, fontsize=8)
            ax.set_title("MAE & RMSE (lower = better)")
            ax.legend(facecolor=P["card"], edgecolor=P["border"])
            plt.tight_layout()
            show(fig)

        with col2:
            fig, ax = fig1(5, 4)
            r2s = [res[m]["R²"] for m in models]
            ax.bar(models, r2s, color=P["green"], edgecolor=P["card"])
            ax.set_title("R² Score (higher = better)")
            for patch, val in zip(ax.patches, r2s):
                if isinstance(patch, Rectangle):
                    ax.text(
                        patch.get_x() + patch.get_width() / 2,
                        patch.get_height() + 0.001,
                        f"{val:.4f}",
                        ha="center",
                        fontsize=9,
                        color=P["muted"],
                    )
            plt.tight_layout()
            show(fig)

        st.markdown("---")
        st.subheader("Overfitting Check")
        c1, c2, c3 = st.columns(3)
        c1.metric("Train R² (OLS)", f"{train_r2:.4f}")
        c2.metric("Test R² (OLS)", f"{res['OLS']['R²']:.4f}")
        c3.metric("Gap", f"{abs(train_r2 - res['OLS']['R²']):.4f}")
        st.success(
            "No overfitting — train and test R² are close. The model underfits due to weak linear relationships."
        )

        st.subheader("5-Fold Cross-Validation R²")
        fig, ax = fig1(8, 3)
        ax.bar(
            [f"Fold {i+1}" for i in range(len(cv))],
            cv,
            color=P["blue"],
            edgecolor=P["card"],
        )
        ax.axhline(
            cv.mean(),
            color=P["green"],
            linestyle="--",
            linewidth=1.5,
            label=f"Mean = {cv.mean():.3f}",
        )
        ax.set_title("Stratified 5-Fold CV R²")
        ax.legend(facecolor=P["card"], edgecolor=P["border"])
        plt.tight_layout()
        show(fig)

    with tab4:
        st.subheader("Diagnostic Plots")
        model_sel2 = st.selectbox("Model", list(preds.keys()), key="diag_model")
        y_pred = preds[model_sel2]
        residuals = y_te - y_pred

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = fig1(5, 4)
            ax.scatter(y_te, y_pred, alpha=0.3, color=P["blue"], s=15)
            ax.plot([0, 1], [0, 1], "r--", linewidth=2, label="Perfect prediction")
            ax.set_xlabel("Actual label")
            ax.set_ylabel("Predicted score")
            ax.set_title("Actual vs Predicted")
            ax.legend(facecolor=P["card"], edgecolor=P["border"])
            plt.tight_layout()
            show(fig)

        with col2:
            fig, ax = fig1(5, 4)
            ax.scatter(y_pred, residuals, alpha=0.3, color=P["orange"], s=15)
            ax.axhline(0, color=P["red"], linewidth=2, linestyle="--")
            ax.set_xlabel("Predicted score")
            ax.set_ylabel("Residual")
            ax.set_title("Residual Plot")
            plt.tight_layout()
            show(fig)

        st.error("""
**Actual vs Predicted:** Benign predictions cluster near 0. Attack predictions scatter widely (0–0.5), 
far below the perfect-prediction line — the model severely under-predicts attack scores.

**Residual Plot:** Two distinct clusters with a diagonal pattern — not randomly scattered around zero. 
This violates OLS assumptions and confirms systematic under-prediction of attacks.
        """)

        st.markdown("---")
        st.subheader("Conclusions")
        st.error(
            "R² ≈ 0.17 — only 17% of label variance explained. Linear regression is insufficient for this non-linear, imbalanced problem."
        )
        st.warning(
            "MAE ≈ 0.07 is misleading — dominated by the 96% majority class. A model predicting zero everywhere achieves similar MAE."
        )
        st.success(
            "Next steps: Random Forest / XGBoost, SMOTE for class imbalance, or logistic regression with class weighting."
        )
# ═════════════════════════════════════════════════════════════════════════════
# CONCLUSION
# ═════════════════════════════════════════════════════════════════════════════
elif page_id == "conclusion":
    st.title("📋 Conclusion")
    st.caption("A full-pipeline summary of findings, limitations, and recommendations.")
    st.markdown("---")

    # ── Pipeline recap ────────────────────────────────────────────────────────
    st.subheader("Pipeline Recap")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**1 · Data Collection**")
        st.markdown("""
The dataset consists of **10,000 simulated network connection records** with 13 raw features
covering ports, byte volumes, protocol, IP addresses, and a binary attack label.
Five attack categories are present (DDoS, injection, intrusion, ransomware, benign),
but the dataset is heavily imbalanced — **96% benign, 4% attack**.
        """)

        st.markdown("**2 · Preprocessing**")
        st.markdown("""
- The sparse `url` column was converted to a binary `has_url` flag instead of being dropped or imputed.  
- `protocol` casing was normalised; the column was then one-hot encoded into three binary columns.  
- `timestamp` was parsed and decomposed into temporal features (`hour`, `day_of_week`, `is_weekend`, `is_night`).  
- High-cardinality identifiers (`src_ip`, `dst_ip`) and leakage columns (`attack_type`) were removed.  
- Result: **0 missing values, 0 duplicates, 23 clean numeric features**.
        """)

        st.markdown("**3 · Feature Engineering**")
        st.markdown("""
Ten new features were derived from the raw columns:
 
| Group | Features |
|---|---|
| Temporal | `hour`, `day_of_week`, `is_weekend`, `is_night` |
| Byte-based | `total_bytes`, `bytes_ratio` |
| Port-based | `port_diff`, `src/dst_port_is_well_known`, `src/dst_port_is_ephemeral`, `dst/src_port_common` |
 
Port features proved most informative; temporal features showed little predictive value.
        """)

    with col2:
        st.markdown("**4 · Exploratory Data Analysis**")
        st.markdown("""
- Byte features (`bytes_sent`, `bytes_received`, `total_bytes`) are **extremely right-skewed**;
  log-transform reveals bell-shaped structure.
- Attack traffic produces **more extreme byte outliers**, consistent with flooding or exfiltration.
- **`dst_port_common`** shows the largest attack-rate gap between its two values —
  traffic targeting uncommon ports is disproportionately malicious.
- All Pearson correlations with the target are **weak (max |r| ≈ 0.17)**,
  indicating the attack/benign boundary is inherently non-linear.
- No strong temporal pattern: attack rate fluctuates between 2.5% and 5.5% with no consistent hour or day trend.
        """)

        st.markdown("**5 · Modeling**")
        st.markdown("""
Three linear regression models were trained on 11 selected features after StandardScaler normalisation:
 
| Model | MAE | RMSE | R² |
|---|---|---|---|
| OLS | 0.0694 | 0.1786 | 0.1697 |
| Ridge (α=1.0) | 0.0694 | 0.1786 | 0.1697 |
| Lasso (α=0.001) | 0.0694 | 0.1787 | 0.1695 |
 
5-fold stratified cross-validation confirmed stable generalisation (CV R² ≈ 0.17 ± 0.01).
Train and test R² are nearly identical — **no overfitting**, only underfitting.
        """)

    st.markdown("---")

    # ── Key findings ─────────────────────────────────────────────────────────
    st.subheader("Key Findings")

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Best R²",
        "0.170",
        help="OLS and Ridge tie — regularisation has no effect at this scale",
    )
    c2.metric(
        "Strongest feature",
        "dst_port_common",
        help="Largest coefficient magnitude across all models",
    )
    c3.metric("CV stability", "±0.01", help="Standard deviation of 5-fold R² scores")

    st.markdown("")
    col1, col2 = st.columns(2)

    with col1:
        st.success("""
**What worked**  
- Feature engineering surfaced meaningful structure (port flags, byte aggregates).  
- Preprocessing was clean: zero leakage, zero missing values after engineering.  
- All three models are consistent and generalise well — no overfitting detected.  
- `dst_port_common` reliably separates the two classes better than any other feature.
        """)

    with col2:
        st.error("""
**What did not work**  
- R² ≈ 0.17 means 83% of label variance is unexplained by any linear combination of features.  
- MAE ≈ 0.07 is deceptively low — a trivial zero-predictor achieves similar MAE due to class imbalance.  
- The residual plot shows two diagonal clusters, violating OLS homoscedasticity assumptions.  
- Temporal features (`hour`, `day_of_week`) contributed negligible predictive signal.
        """)

    st.markdown("---")

    # ── Limitations ───────────────────────────────────────────────────────────
    st.subheader("Limitations")
    st.warning("""
**Class imbalance (96/4)**  
Standard accuracy and MAE metrics are dominated by the majority class.
No resampling (SMOTE, undersampling) or class-weight adjustment was applied in this pipeline.
 
**Linear assumption**  
The attack boundary is non-linear. Linear regression produces continuous scores rather than
crisp 0/1 labels, making threshold selection arbitrary and evaluation metrics misleading.
 
**Simulated data**  
The dataset is synthetically generated. Real-world network traffic exhibits stronger temporal
autocorrelation, IP-level clustering, and protocol-specific byte patterns not present here.
    """)

    # ── Final verdict ─────────────────────────────────────────────────────────
    st.subheader("Final Verdict")
    st.info("""
Linear regression establishes a clear **baseline**: the engineered features carry genuine
signal (particularly port-based flags), but a linear model cannot exploit the non-linear
structure of the attack boundary. The pipeline — from raw CSV to scaled, cross-validated
models — is sound and reproducible. The logical next step is to swap the estimator for a
tree-based classifier and adopt imbalance-aware evaluation metrics, keeping the rest of
the pipeline unchanged.
    """)
