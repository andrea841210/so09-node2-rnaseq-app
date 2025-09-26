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
st.set_page_config(page_title="Node-2 RNA-seq • Quick Viz (v1.2.4)", layout="wide")
st.title("Node-2 RNA-seq • Quick Viz (v1.2.4)")
st.caption("Andrea × GC × Strategist v1 | R computation × Python visualization — Unified dashboard")

# ==========================
# Helpers
# ==========================

@st.cache_data(show_spinner=False)
def read_csv(uploaded, **kwargs) -> Optional[pd.DataFrame]:
    if uploaded is None:
        return None
    try:
        return pd.read_csv(uploaded, **kwargs)
    except Exception as e:
        st.warning(f"Read failed: {e}")
        return None

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {}
    for c in df.columns:
        norm = (
            str(c).strip().replace("\u00A0", " ")
            .replace(" ", "_").replace("-", "_")
            .replace("/", "_").replace("(", "").replace(")", "")
        )
        norm = "".join(ch for ch in norm if ch.isalnum() or ch == "_").lower()
        mapping[c] = norm
    return df.rename(columns=mapping)

# ---- preprocessors

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
    elif "symbol" not in df.columns and "gene" in df.columns:
        df = df.rename(columns={"gene": "symbol"})
    return df

def prepare_fgsea(df_fgsea_raw: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df_fgsea_raw.copy())
    alias = {"gs": "pathway", "leading_edge": "leadingedge"}
    df = df.rename(columns={k: v for k, v in alias.items() if k in df.columns})
    req = ["pathway", "es", "nes", "padj"]
    for r in req:
        if r not in df.columns:
            raise ValueError(f"FGSEA results missing column: {r}")
    return df

def prepare_rank(df_rank_raw: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df_rank_raw.copy())
    if "rank" not in df.columns:
        score_col = None
        for cand in ["score", "stat", "t", "z", "log2fc"]:
            if cand in df.columns:
                score_col = cand; break
        if score_col is None:
            raise ValueError("rank.csv must contain either rank or score column")
        df = df.sort_values(score_col, ascending=False).reset_index(drop=True)
        df["rank"] = np.arange(1, len(df) + 1)
    if "score" not in df.columns:
        for cand in ["score", "stat", "t", "z", "log2fc"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "score"}); break
    if "gene" not in df.columns and "symbol" in df.columns:
        df = df.rename(columns={"symbol": "gene"})
    return df

def prepare_es_curve(df_es_raw: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df_es_raw.copy())
    if not set(["rank", "es", "hit"]).issubset(df.columns):
        raise ValueError("es_curve.csv must contain columns: rank, ES, hit")
    df["rank"], df["es"], df["hit"] = df["rank"].astype(int), df["es"].astype(float), df["hit"].astype(int)
    return df.sort_values("rank")

# ==========================
# Sidebar — Run + Collapsibles
# ==========================

# Run / Reset controls (top-level)
_run_col = st.sidebar.container()
if _run_col.button("▶️ Run Analysis", type="primary"):
    st.session_state["do_run"] = True
if _run_col.button("Reset"):
    st.session_state["do_run"] = False
    for k in ["up_deg", "up_fgsea", "up_rank", "up_le", "up_es"]:
        if k in st.session_state:
            del st.session_state[k]

do_run = st.session_state.get("do_run", False)

# Anchors (collapsible)
with st.sidebar.expander("Anchor Inputs", expanded=False):
    up_deg = st.file_uploader("1) DEG table", type=["csv", "tsv", "txt"], key="up_deg")
    up_fgsea = st.file_uploader("2) fgsea_results.csv", type=["csv"], key="up_fgsea")
    up_rank = st.file_uploader("3) rank.csv", type=["csv"], key="up_rank")
    up_le = st.file_uploader("4) leading_edge.csv (optional)", type=["csv"], key="up_le")
    up_es = st.file_uploader("5) es_curve.csv (from R)", type=["csv"], key="up_es")

# Thresholds (collapsible)
with st.sidebar.expander("Thresholds", expanded=False):
    padj_cutoff = st.number_input("Volcano padj cutoff", value=0.05, step=0.01, format="%.3f")
    logfc_cutoff = st.number_input("|log2FC| cutoff", value=1.0, step=0.1, format="%.1f")
    ma_padj_cutoff = st.number_input("MA plot padj cutoff", value=0.05, step=0.01, format="%.2f")
    lowexpr_dim = st.checkbox("MA: dim low expression", value=False, help="If checked: FPKM mode uses AveExpr<1; counts mode uses baseMean<1")
    topn = st.number_input("Top-N pathways", value=15, step=1, min_value=1, max_value=50)

# Enrichment title (collapsible)
with st.sidebar.expander("Enrichment curve title", expanded=False):
    pathway_title = st.text_input("Pathway name (display in title)", value="")

# Global style controls (collapsible)
st.sidebar.header("Global style controls")
with st.sidebar.expander("Style (shared)", expanded=False):
    cfg_color = {
        "sig": st.color_picker("Significant color", "#D62728"),
        "bg": st.color_picker("Background color", "#8C8C8C"),
        "low": st.color_picker("Low expression color", "#B0B0B0"),
        "line": st.color_picker("Enrichment line color", "#00C853"),
        "tick": st.color_picker("Hit tick color", "#000000"),
        "bar": st.color_picker("Top-N bar color", "#1f77b4"),
    }
    cfg_size = {
        "point": st.slider("Point size", 3, 15, 6),
        "tick": st.slider("Hit tick length factor (%)", 2, 10, 6) / 100.0,
    }
    cfg_line = {
        "main": st.slider("Enrichment line width", 1, 5, 2),
        "cutoff": st.slider("NES cutoff line width", 0, 3, 1),
    }

# Debug (collapsible)
with st.sidebar.expander("Debug view", expanded=False):
    show_heads = st.checkbox("Show first 10 rows of inputs", value=False)

# ==========================
# Load + Prepare (only after Run)
# ==========================

df_deg = df_fgsea = df_rank = df_le = df_es = None

if do_run:
    df_deg = read_csv(up_deg)
    df_fgsea = read_csv(up_fgsea)
    df_rank = read_csv(up_rank)
    df_le = read_csv(up_le)
    df_es = read_csv(up_es)

    try:
        if df_deg is not None:
            df_deg = prepare_deg(df_deg)
        if df_fgsea is not None:
            df_fgsea = prepare_fgsea(df_fgsea)
        if df_rank is not None:
            df_rank = prepare_rank(df_rank)
        if df_le is not None:
            df_le = normalize_columns(df_le)
        if df_es is not None:
            df_es = prepare_es_curve(df_es)
    except Exception as e:
        st.error(f"Preprocessing error: {e}")

# ==========================
# Debug heads
# ==========================

if do_run and show_heads:
    c1, c2 = st.columns(2)
    with c1:
        if df_deg is not None:
            st.subheader("DEG head"); st.dataframe(df_deg.head(10))
        if df_rank is not None:
            st.subheader("Rank head"); st.dataframe(df_rank.head(10))
        if df_es is not None:
            st.subheader("ES curve head"); st.dataframe(df_es.head(10))
    with c2:
        if df_fgsea is not None:
            st.subheader("FGSEA head"); st.dataframe(df_fgsea.head(10))
        if df_le is not None:
            st.subheader("Leading-edge head"); st.dataframe(df_le.head(10))

# ==========================
# A. Volcano plot
# ==========================

st.markdown("---")
st.subheader("A. Volcano plot")
if not do_run:
    st.info("Press ▶️ Run Analysis to generate plots.")
elif df_deg is not None:
    fig = plt.figure()
    x = df_deg["log2fc"]; y = -np.log10(df_deg["padj"])
    sig = (df_deg["padj"] < padj_cutoff) & (x.abs() > logfc_cutoff)
    plt.scatter(x[~sig], y[~sig], s=cfg_size["point"], alpha=0.6, c=cfg_color["bg"])
    plt.scatter(x[sig], y[sig], s=cfg_size["point"], alpha=0.9, c=cfg_color["sig"])
    plt.axvline(-logfc_cutoff, linestyle=":"); plt.axvline(logfc_cutoff, linestyle=":")
    plt.axhline(-np.log10(max(padj_cutoff, 1e-300)), linestyle=":")
    plt.xlabel("log2FC"); plt.ylabel("-log10(padj)")
    st.pyplot(fig)
else:
    st.info("DEG table not provided, Volcano plot skipped.")

# ==========================
# B. MA plot (dual-mode: FPKM vs counts)
# ==========================

st.markdown("---")
st.subheader("B. MA plot")
if not do_run:
    st.info("Press ▶️ Run Analysis to generate plots.")
elif df_deg is not None:
    try:
        cols = set(df_deg.columns)
        if "basemean" in cols:  # DESeq2 counts mode — use log10(baseMean+1)
            dfm = df_deg[["basemean", "log2fc", "padj"]].copy()
            dfm["basemean"] = pd.to_numeric(dfm["basemean"], errors="coerce")
            dfm["log2fc"] = pd.to_numeric(dfm["log2fc"], errors="coerce")
            dfm["padj"] = pd.to_numeric(dfm["padj"], errors="coerce")
            dfm = dfm.dropna(subset=["basemean", "log2fc", "padj"])  # +1 guards zeros for log10
            A_raw = dfm["basemean"].values
            A = np.log10(A_raw + 1.0)
            M = dfm["log2fc"].values
            sig_mask = dfm["padj"].values < ma_padj_cutoff
            low_mask = A_raw < 1.0
            fig = plt.figure()
            idx_bg = ~sig_mask
            if lowexpr_dim:
                plt.scatter(A[idx_bg & low_mask], M[idx_bg & low_mask], s=cfg_size["point"], alpha=0.3, c=cfg_color["low"]) 
                plt.scatter(A[idx_bg & ~low_mask], M[idx_bg & ~low_mask], s=cfg_size["point"], alpha=0.65, c=cfg_color["bg"]) 
            else:
                plt.scatter(A[idx_bg], M[idx_bg], s=cfg_size["point"], alpha=0.65, c=cfg_color["bg"]) 
            plt.scatter(A[sig_mask], M[sig_mask], s=cfg_size["point"], alpha=0.65, c=cfg_color["sig"]) 
            plt.axhline(0.0, linestyle=":", linewidth=1.0)
            plt.xlabel("log10(baseMean)")
            plt.ylabel("M = log2FC")
            st.pyplot(fig)
            st.caption(f"MA (counts) — points: {int(len(dfm))}, significant (padj<{ma_padj_cutoff:.2f}): {int(sig_mask.sum())}")
        elif "aveexpr" in cols:  # FPKM/TPM limma-trend mode — already on log2 scale
            dfm = df_deg[["aveexpr", "log2fc", "padj"]].copy()
            dfm["aveexpr"] = pd.to_numeric(dfm["aveexpr"], errors="coerce")
            dfm["log2fc"] = pd.to_numeric(dfm["log2fc"], errors="coerce")
            dfm["padj"] = pd.to_numeric(dfm["padj"], errors="coerce")
            dfm = dfm.dropna(subset=["aveexpr", "log2fc", "padj"])  # no extra transforms
            A = dfm["aveexpr"].values
            M = dfm["log2fc"].values
            sig_mask = dfm["padj"].values < ma_padj_cutoff
            low_mask = A < 1.0
            fig = plt.figure()
            idx_bg = ~sig_mask
            if lowexpr_dim:
                plt.scatter(A[idx_bg & low_mask], M[idx_bg & low_mask], s=cfg_size["point"], alpha=0.3, c=cfg_color["low"]) 
                plt.scatter(A[idx_bg & ~low_mask], M[idx_bg & ~low_mask], s=cfg_size["point"], alpha=0.65, c=cfg_color["bg"]) 
            else:
                plt.scatter(A[idx_bg], M[idx_bg], s=cfg_size["point"], alpha=0.65, c=cfg_color["bg"]) 
            plt.scatter(A[sig_mask], M[sig_mask], s=cfg_size["point"], alpha=0.65, c=cfg_color["sig"]) 
            plt.axhline(0.0, linestyle=":", linewidth=1.0)
            plt.xlabel("Average expression (log2 FPKM)")
            plt.ylabel("M = log2FC")
            st.pyplot(fig)
            st.caption(f"MA (FPKM) — points: {int(len(dfm))}, significant (padj<{ma_padj_cutoff:.2f}): {int(sig_mask.sum())}")
        else:
            st.error("Missing abundance column (baseMean or AveExpr).")
    except Exception as e:
        st.warning(f"MA plot failed: {e}")
else:
    st.info("DEG table not provided, MA plot skipped.")

# ==========================
# C. Top-N GSEA pathways (by NES)
# ==========================

st.markdown("---")
st.subheader("C. Top-N GSEA pathways (by NES)")
if not do_run:
    st.info("Press ▶️ Run Analysis to generate plots.")
elif df_fgsea is not None:
    df_plot = df_fgsea.sort_values("nes", ascending=False).head(int(topn))
    fig = plt.figure()
    plt.barh(df_plot["pathway"], df_plot["nes"], color=cfg_color["bar"])
    plt.xlabel("NES"); plt.ylabel("Pathway"); plt.gca().invert_yaxis()
    st.pyplot(fig)
else:
    st.info("fgsea_results.csv not provided, Top-N skipped.")

# ==========================
# D. Enrichment curve (R-style)
# ==========================

st.markdown("---")
st.subheader("D. Enrichment curve (R-style)")
step_where = st.selectbox("Enrichment step alignment", ["pre", "post"], index=0, help="If first step misaligned with R, switch here.")
st.session_state["where_mode"] = step_where
if not do_run:
    st.info("Press ▶️ Run Analysis to generate plots.")
elif (df_es is not None) and (df_rank is not None):
    fig = plt.figure()
    xs = df_es["rank"].values; es = df_es["es"].values
    es_max, es_min = float(np.max(es)), float(np.min(es)); es_range = es_max - es_min
    plt.step(xs, es, where=step_where, linewidth=cfg_line["main"], color=cfg_color["line"])
    plt.axhline(0.0, color="black", linestyle="-", linewidth=1.0)
    plt.axhline(es_max, color="red", linestyle="--", linewidth=cfg_line["cutoff"])
    plt.axhline(es_min, color="red", linestyle="--", linewidth=cfg_line["cutoff"])
    hits = df_es.loc[df_es["hit"] == 1, "rank"].values
    if (len(hits) > 0) and (es_range > 0):
        tick_len = cfg_size["tick"] * es_range
        nes_val = None; pt = pathway_title.strip()
        if (df_fgsea is not None) and (len(pt) > 0) and ("pathway" in df_fgsea.columns):
            row = df_fgsea[df_fgsea["pathway"].str.lower() == pt.lower()]
            if not row.empty and "nes" in row.columns:
                nes_val = float(row.iloc[0]["nes"])
        if nes_val is None:
            nes_val = es_max if abs(es_max) >= abs(es_min) else es_min
        y0, y1 = (0.0, -tick_len) if nes_val >= 0 else (0.0, tick_len)
        for r in hits:
            plt.vlines(r, y0, y1, linewidth=0.5, color=cfg_color["tick"])
    buf = 1.5 * (cfg_size["tick"] * es_range if es_range > 0 else max(abs(es_max), abs(es_min))*0.05)
    plt.ylim(es_min - buf, es_max + buf)
    title_txt = pathway_title.strip()
    if (df_fgsea is not None) and (len(title_txt) > 0) and ("pathway" in df_fgsea.columns):
        row = df_fgsea[df_fgsea["pathway"].str.lower() == title_txt.lower()]
        if not row.empty:
            nes = row.iloc[0]["nes"] if "nes" in row.columns else None
            fdr = row.iloc[0]["padj"] if "padj" in row.columns else None
            if nes is not None and fdr is not None:
                title_txt = f"{title_txt} (NES={nes:.2f}, FDR={fdr:.2g})"
    if len(title_txt) > 0:
        plt.title(title_txt)
    plt.xlabel("rank (descending score)"); plt.ylabel("enrichment score")
    st.pyplot(fig)
else:
    st.info("es_curve.csv or rank.csv not provided, Enrichment curve skipped.")

# ==========================
# E. Summary (rule-based quick stats)
# ==========================

st.markdown("---")
st.subheader("E. Summary")
if do_run and (df_fgsea is not None):
    up_df = df_fgsea.sort_values("nes", ascending=False).head(1)
    down_df = df_fgsea.sort_values("nes", ascending=True).head(1)
    up_item = up_df.iloc[0]; down_item = down_df.iloc[0]
    st.markdown(
        f"**Top NES**: {up_item['pathway']} (NES={up_item['nes']:.3f}, ES={up_item['es']:.3f}, padj={up_item['padj']:.2e})\n\n"
        f"**Bottom NES**: {down_item['pathway']} (NES={down_item['nes']:.3f}, ES={down_item['es']:.3f}, padj={down_item['padj']:.2e})"
    )
elif do_run:
    st.info("FGSEA results not provided, summary skipped.")
else:
    st.info("Press ▶️ Run Analysis to generate summary.")

# ==========================
# F. Export plots (ZIP)
# ==========================

st.markdown("---")
st.subheader("F. Export plots (ZIP)")
colx1, colx2 = st.columns(2)
with colx1:
    want_export = st.checkbox("Package current plots into ZIP (PNG)", value=False)
with colx2:
    zip_name = st.text_input("Output filename", value="node2_quickviz_export.zip")

if want_export:
    bufs, names = [], []

    # Volcano
    if do_run and (df_deg is not None):
        fig = plt.figure()
        x = df_deg["log2fc"]; y = -np.log10(df_deg["padj"])
        sig = (df_deg["padj"] < padj_cutoff) & (x.abs() > logfc_cutoff)
        plt.scatter(x[~sig], y[~sig], s=cfg_size["point"], alpha=0.6, c=cfg_color["bg"])
        plt.scatter(x[sig], y[sig], s=cfg_size["point"], alpha=0.9, c=cfg_color["sig"])
        plt.axvline(-logfc_cutoff, linestyle=":"); plt.axvline(logfc_cutoff, linestyle=":")
        plt.axhline(-np.log10(max(padj_cutoff, 1e-300)), linestyle=":")
        plt.xlabel("log2FC"); plt.ylabel("-log10(padj)")
        png = io.BytesIO(); plt.savefig(png, format="png", bbox_inches="tight", dpi=200)
        png.seek(0); bufs.append(png.read()); names.append("Volcano_plot.png")

    # MA plot (export; same dual-mode logic)
    if do_run and (df_deg is not None):
        try:
            cols = set(df_deg.columns)
            if "basemean" in cols:
                dfm = df_deg[["basemean", "log2fc", "padj"]].copy()
                dfm["basemean"] = pd.to_numeric(dfm["basemean"], errors="coerce")
                dfm["log2fc"], dfm["padj"] = pd.to_numeric(dfm["log2fc"], errors="coerce"), pd.to_numeric(dfm["padj"], errors="coerce")
                dfm = dfm.dropna(subset=["basemean", "log2fc", "padj"]) 
                A_raw = dfm["basemean"].values
                A = np.log10(A_raw + 1.0)
                M = dfm["log2fc"].values
                sig_mask = dfm["padj"].values < ma_padj_cutoff; low_mask = A_raw < 1.0
                fig = plt.figure(); idx_bg = ~sig_mask
                if lowexpr_dim:
                    plt.scatter(A[idx_bg & low_mask], M[idx_bg & low_mask], s=cfg_size["point"], alpha=0.3, c=cfg_color["low"]) 
                    plt.scatter(A[idx_bg & ~low_mask], M[idx_bg & ~low_mask], s=cfg_size["point"], alpha=0.65, c=cfg_color["bg"]) 
                else:
                    plt.scatter(A[idx_bg], M[idx_bg], s=cfg_size["point"], alpha=0.65, c=cfg_color["bg"]) 
                plt.scatter(A[sig_mask], M[sig_mask], s=cfg_size["point"], alpha=0.65, c=cfg_color["sig"]) 
                plt.axhline(0.0, linestyle=":", linewidth=1.0)
                plt.xlabel("log10(baseMean)"); plt.ylabel("M = log2FC")
                png = io.BytesIO(); plt.savefig(png, format="png", bbox_inches="tight", dpi=200)
                png.seek(0); bufs.append(png.read()); names.append("MA_plot.png")
            elif "aveexpr" in cols:
                dfm = df_deg[["aveexpr", "log2fc", "padj"]].copy()
                dfm["aveexpr"] = pd.to_numeric(dfm["aveexpr"], errors="coerce")
                dfm["log2fc"], dfm["padj"] = pd.to_numeric(dfm["log2fc"], errors="coerce"), pd.to_numeric(dfm["padj"], errors="coerce")
                dfm = dfm.dropna(subset=["aveexpr", "log2fc", "padj"]) 
                A = dfm["aveexpr"].values; M = dfm["log2fc"].values
                sig_mask = dfm["padj"].values < ma_padj_cutoff; low_mask = A < 1.0
                fig = plt.figure(); idx_bg = ~sig_mask
                if lowexpr_dim:
                    plt.scatter(A[idx_bg & low_mask], M[idx_bg & low_mask], s=cfg_size["point"], alpha=0.3, c=cfg_color["low"]) 
                    plt.scatter(A[idx_bg & ~low_mask], M[idx_bg & ~low_mask], s=cfg_size["point"], alpha=0.65, c=cfg_color["bg"]) 
                else:
                    plt.scatter(A[idx_bg], M[idx_bg], s=cfg_size["point"], alpha=0.65, c=cfg_color["bg"]) 
                plt.scatter(A[sig_mask], M[sig_mask], s=cfg_size["point"], alpha=0.65, c=cfg_color["sig"]) 
                plt.axhline(0.0, linestyle=":", linewidth=1.0)
                plt.xlabel("Average expression (log2 FPKM)"); plt.ylabel("M = log2FC")
                png = io.BytesIO(); plt.savefig(png, format="png", bbox_inches="tight", dpi=200)
                png.seek(0); bufs.append(png.read()); names.append("MA_plot.png")
        except Exception:
            pass

    # Enrichment curve
    if do_run and (df_es is not None) and (df_rank is not None):
        fig = plt.figure()
        xs = df_es["rank"].values; es = df_es["es"].values
        es_max, es_min = float(np.max(es)), float(np.min(es)); es_range = es_max - es_min
        where_mode = st.session_state.get("where_mode", "pre") if isinstance(st.session_state.get("where_mode", "pre"), str) else "pre"
        plt.step(xs, es, where=where_mode, linewidth=cfg_line["main"], color=cfg_color["line"])
        plt.axhline(0.0, color="black", linestyle="-", linewidth=1.0)
        plt.axhline(es_max, color="red", linestyle="--", linewidth=cfg_line["cutoff"]) 
        plt.axhline(es_min, color="red", linestyle="--", linewidth=cfg_line["cutoff"]) 
        hits = df_es.loc[df_es["hit"] == 1, "rank"].values
        if (len(hits) > 0) and (es_range > 0):
            tick_len = cfg_size["tick"] * es_range
            nes_val = None; pt = (pathway_title or "").strip()
            if (df_fgsea is not None) and (len(pt) > 0) and ("pathway" in df_fgsea.columns):
                row = df_fgsea[df_fgsea["pathway"].str.lower() == pt.lower()]
                if not row.empty and "nes" in row.columns:
                    nes_val = float(row.iloc[0]["nes"])
            if nes_val is None:
                nes_val = es_max if abs(es_max) >= abs(es_min) else es_min
            y0, y1 = (0.0, -tick_len) if nes_val >= 0 else (0.0, tick_len)
            for r in hits:
                plt.vlines(r, y0, y1, linewidth=0.5, color=cfg_color["tick"])        
        plt.xlabel("rank (descending score)"); plt.ylabel("enrichment score")
        png = io.BytesIO(); plt.savefig(png, format="png", bbox_inches="tight", dpi=200)
        png.seek(0); bufs.append(png.read()); names.append("Enrichment_curve.png")

    if len(bufs) == 0:
        st.warning("No plots to export.")
    else:
        import zipfile
        bio = io.BytesIO()
        with zipfile.ZipFile(bio, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for data, nm in zip(bufs, names):
                zf.writestr(nm, data)
        st.download_button(label="Download ZIP", data=bio.getvalue(), file_name=zip_name, mime="application/zip")

# Footer
st.markdown("---")
st.caption("R = statistics core (DEG/FGSEA/ES running), Python = visualization (Volcano/MA/Top-N/ES curve) — bridged by es_curve.csv.")
