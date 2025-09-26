import os
import io
from typing import Optional

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================
# App Meta
# ==========================
st.set_page_config(page_title="Node-2 RNA-seq • Quick Viz (v1.3.0)", layout="wide")
st.title("Node-2 RNA-seq • Quick Viz (v1.3.0)")
st.caption("Andrea × GC × Strategist v1 | R computation × Python visualization — Unified dashboard")

# --- Session init ---
if "reset_id" not in st.session_state:
    st.session_state["reset_id"] = 0
if "do_run" not in st.session_state:
    st.session_state["do_run"] = False

# ==========================
# Helpers
# ==========================
def normalize_columns(df: pd.DataFrame):
    mapping = {}
    for c in df.columns:
        norm = (
            str(c).strip()
            .replace("\u00A0", " ")
            .replace(" ", "_").replace("-", "_").replace("/", "_")
            .replace("(", "").replace(")", "")
        )
        norm = "".join(ch for ch in norm if ch.isalnum() or ch == "_").lower()
        mapping[c] = norm
    return df.rename(columns=mapping)

def read_table(file, sep=None):
    if file is None:
        return None
    name = file.name.lower()
    if sep is None:
        if name.endswith(".csv"): sep = ","
        elif name.endswith(".tsv") or name.endswith(".txt"): sep = "\t"
        else: sep = ","
    file.seek(0)
    return pd.read_csv(file, sep=sep)

def safe_float(s):
    try:
        return float(s)
    except Exception:
        return np.nan

def prepare_deg(df_deg_raw: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df_deg_raw.copy())
    col_map = {"log2foldchange": "log2fc", "logfc": "log2fc", "pvalue": "pval", "p_adj": "padj", "padjust": "padj"}
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    if "log2fc" not in df.columns:
        raise ValueError("DEG table must contain log2FoldChange / log2FC column")
    if ("padj" not in df.columns) and ("pval" not in df.columns):
        raise ValueError("DEG table must contain pvalue / padj column")
    if "padj" not in df.columns:
        df["padj"] = df["pval"]
    if "symbol" not in df.columns and "gene" not in df.columns:
        df["symbol"] = [f"gene_{i}" for i in range(len(df))]
    if "gene" in df.columns and "symbol" not in df.columns:
        df = df.rename(columns={"gene": "symbol"})
    if "basemean" in df.columns and "aveexpr" not in df.columns:
        df["abundance"] = df["basemean"]
    elif "aveexpr" in df.columns:
        df["abundance"] = df["aveexpr"]
    else:
        df["abundance"] = np.log10(np.abs(df["log2fc"]).replace(0, np.nan)).fillna(0)
    return df

def prepare_fgsea(df_fgsea_raw: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df_fgsea_raw.copy())
    ren = {"pathway": "pathway", "padj": "padj", "pval": "pval", "nes": "nes", "es": "es"}
    df = df.rename(columns={k: v for k, v in ren.items() if k in df.columns})
    if "padj" not in df.columns and "pval" in df.columns:
        df["padj"] = df["pval"]
    if "pathway" not in df.columns:
        for c in df.columns:
            if df[c].dtype == object:
                df = df.rename(columns={c: "pathway"})
                break
    return df

def prepare_rank(df_rank_raw: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df_rank_raw.copy())
    cand_cols = ["score", "log2fc", "logfc", "stat", "t", "wald_statistic", "rank"]
    score_col = next((c for c in cand_cols if c in df.columns), None)
    if score_col is None:
        if df.shape[1] == 2:
            score_col = df.columns[-1]
        else:
            raise ValueError("rank.csv must contain a ranking score column (e.g., score/log2FC/stat)")
    gene_col = next((g for g in ["symbol", "gene", "genes", "id"] if g in df.columns), df.columns[0])
    df = df.rename(columns={gene_col: "gene", score_col: "score"})
    return df[["gene", "score"]]

def prepare_le(df_le_raw: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df_le_raw.copy())
    if "gene" not in df.columns and "symbol" in df.columns:
        df = df.rename(columns={"symbol": "gene"})
    return df

def prepare_escurve(df_es_raw: pd.DataFrame) -> pd.DataFrame:
    return normalize_columns(df_es_raw.copy())

# ==========================
# Sidebar Controls
# ==========================
st.sidebar.header("Controls")

# Run / Reset controls (top-level)
_run_col = st.sidebar.container()
if _run_col.button("▶️ Run Analysis", type="primary"):
    st.session_state["do_run"] = True
if _run_col.button("Reset"):
    st.session_state["do_run"] = False
    st.session_state["reset_id"] += 1       # refresh uploader keys
    st.cache_data.clear()                    # purge cached reads

do_run = st.session_state.get("do_run", False)

# Anchors
with st.sidebar.expander("Anchor Inputs", expanded=False):
    salt = st.session_state["reset_id"]
    up_deg  = st.file_uploader("1) DEG table", type=["csv","tsv","txt"], key=f"up_deg_{salt}")
    up_fgsea= st.file_uploader("2) fgsea_results.csv", type=["csv"],      key=f"up_fgsea_{salt}")
    up_rank = st.file_uploader("3) rank.csv", type=["csv"],               key=f"up_rank_{salt}")
    up_le   = st.file_uploader("4) leading_edge.csv (optional)", type=["csv"], key=f"up_le_{salt}")
    up_es   = st.file_uploader("5) es_curve.csv (from R)", type=["csv"],  key=f"up_es_{salt}")

# Thresholds
with st.sidebar.expander("Thresholds", expanded=False):
    padj_cutoff     = st.number_input("Volcano padj cutoff", value=0.05, step=0.01, format="%.3f")
    logfc_cutoff    = st.number_input("|log2FC| cutoff", value=1.0, step=0.1, format="%.1f")
    ma_padj_cutoff  = st.number_input("MA plot padj cutoff", value=0.05, step=0.01, format="%.2f")
    lowexpr_dim     = st.number_input("Low-expression dim (MA plot)", value=0.2, step=0.05, min_value=0.0, max_value=1.0)

with st.sidebar.expander("Display", expanded=False):
    show_table_sample = st.checkbox("Show head() of inputs", value=True)
    theme_dark = st.checkbox("Dark-ish background for charts", value=False)

# ==========================
# Data Loading (cached)
# ==========================
@st.cache_data
def _load_df(file, sep=None):
    return read_table(file, sep=sep)

df_deg  = _load_df(up_deg)
df_fgsea= _load_df(up_fgsea)
df_rank = _load_df(up_rank)
df_le   = _load_df(up_le)
df_es   = _load_df(up_es)

# ==========================
# Layout
# ==========================
tab1, tab2, tab3, tab4 = st.tabs(["Volcano", "MA plot", "GSEA (table + ES)", "Lead Edge"])

# Optional: show input snapshots (允許在未按 Run 前僅看頭部資料，避免眼花)
if show_table_sample:
    with st.expander("Input snapshots", expanded=False):
        if df_deg is not None:  st.write("DEG table:", df_deg.head())
        if df_fgsea is not None:st.write("fgsea_results:", df_fgsea.head())
        if df_rank is not None: st.write("rank.csv:", df_rank.head())
        if df_le is not None:   st.write("leading_edge.csv:", df_le.head())
        if df_es is not None:   st.write("es_curve.csv:", df_es.head())

# ==========================
# Volcano
# ==========================
with tab1:
    st.subheader("Volcano plot")
    if not do_run:
        st.info("Press **Run Analysis** to render volcano.")
    elif df_deg is None:
        st.warning("Upload DEG table to render volcano.")
    else:
        try:
            ddeg = prepare_deg(df_deg)
            x = ddeg["log2fc"].astype(float)
            y = -np.log10(ddeg["padj"].astype(float))
            sig = (ddeg["padj"].astype(float) <= padj_cutoff) & (np.abs(ddeg["log2fc"].astype(float)) >= logfc_cutoff)
            fig, ax = plt.subplots()
            if theme_dark:
                ax.set_facecolor("#0f1116"); fig.patch.set_facecolor("#0f1116")
            ax.scatter(x, y, s=8, alpha=0.6, linewidths=0)
            ax.scatter(x[sig], y[sig], s=12, alpha=0.9)
            ax.axvline(+logfc_cutoff, ls="--", lw=1); ax.axvline(-logfc_cutoff, ls="--", lw=1)
            ax.axhline(-np.log10(padj_cutoff), ls="--", lw=1)
            ax.set_xlabel("log2FC"); ax.set_ylabel("-log10(padj)")
            st.pyplot(fig, use_container_width=True)
        except Exception as e:
            st.error(f"[Volcano] {e}")

# ==========================
# MA plot
# ==========================
with tab2:
    st.subheader("MA plot")
    if not do_run:
        st.info("Press **Run Analysis** to render MA plot.")
    elif df_deg is None:
        st.warning("Upload DEG table to render MA plot.")
    else:
        try:
            ddeg = prepare_deg(df_deg)
            x = ddeg["abundance"].astype(float)
            y = ddeg["log2fc"].astype(float)
            sig = (ddeg["padj"].astype(float) <= ma_padj_cutoff)
            fig, ax = plt.subplots()
            if theme_dark:
                ax.set_facecolor("#0f1116"); fig.patch.set_facecolor("#0f1116")
            ax.scatter(x, y, s=8, alpha=0.6, linewidths=0)
            ax.scatter(x[sig], y[sig], s=12, alpha=0.9)
            ax.axhline(0.0, ls="-", lw=1)
            ax.set_xlabel("Abundance (baseMean / aveExpr)"); ax.set_ylabel("log2FC")
            st.pyplot(fig, use_container_width=True)
        except Exception as e:
            st.error(f"[MA] {e}")

# ==========================
# GSEA
# ==========================
with tab3:
    st.subheader("GSEA results")
    col_g1, col_g2 = st.columns([0.45, 0.55], gap="large")

    with col_g1:
        if not do_run:
            st.info("Press **Run Analysis** to show GSEA table.")
        elif df_fgsea is None:
            st.warning("Upload fgsea_results.csv to show pathway table.")
        else:
            try:
                dfg = prepare_fgsea(df_fgsea)
                st.dataframe(dfg, use_container_width=True)
            except Exception as e:
                st.error(f"[GSEA table] {e}")

    with col_g2:
        st.subheader("Enrichment curve (from R output)")
        if not do_run:
            st.info("Press **Run Analysis** to render ES curve.")
        elif df_es is None:
            st.warning("Upload es_curve.csv to render ES curve.")
        else:
            try:
                des = prepare_escurve(df_es)
                pos_col = next((c for c in ["position","pos","i","rank_index"] if c in des.columns), None)
                es_col  = next((c for c in ["running_es","running_score","es","score"] if c in des.columns), None)
                if pos_col is None or es_col is None:
                    raise ValueError("es_curve.csv must contain running ES over ranked positions")
                fig, ax = plt.subplots()
                if theme_dark:
                    ax.set_facecolor("#0f1116"); fig.patch.set_facecolor("#0f1116")
                ax.plot(des[pos_col].astype(float), des[es_col].astype(float), lw=2)
                ax.axhline(0.0, ls="--", lw=1)
                ax.set_xlabel("Ranked position"); ax.set_ylabel("Running ES")
                st.pyplot(fig, use_container_width=True)
            except Exception as e:
                st.error(f"[ES curve] {e}")

# ==========================
# Leading Edge
# ==========================
with tab4:
    st.subheader("Leading edge genes")
    if not do_run:
        st.info("Press **Run Analysis** to list leading edge genes.")
    elif df_le is None:
        st.warning("Upload leading_edge.csv to list genes.")
    else:
        try:
            dle = prepare_le(df_le)
            st.dataframe(dle, use_container_width=True)
        except Exception as e:
            st.error(f"[Leading edge] {e}")

# ==========================
# Export pack
# ==========================
st.markdown("---")
st.subheader("Export deliverables")
with st.container():
    fn = st.text_input("ZIP filename", value="node2_rnaseq_deliverables.zip")
    if st.button("Build ZIP"):
        bufs, names = [], []
        for file_obj, nm in [(up_deg,"DEG.csv"), (up_fgsea,"fgsea_results.csv"),
                             (up_rank,"rank.csv"), (up_le,"leading_edge.csv"), (up_es,"es_curve.csv")]:
            if file_obj is not None:
                file_obj.seek(0); bufs.append(file_obj.read()); names.append(nm)
        import zipfile
        bio = io.BytesIO()
        with zipfile.ZipFile(bio, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for data, nm in zip(bufs, names):
                zf.writestr(nm, data)
        st.download_button("Download ZIP", bio.getvalue(), file_name=fn, mime="application/zip")

# Footer
st.markdown("---")
st.caption("Charts render only after pressing **Run Analysis**. Reset clears cache and uploaded anchors via reset_id-salted keys.")
