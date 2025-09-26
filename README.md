# Node-2 RNA-seq Deliverable Pack (R → Python Anchors)

## 🔑 Required Anchors (5 files)

1. **DEG_table.csv**

   * Columns: `SYMBOL, log2FoldChange, pvalue, padj, AveExpr`
   * Notes:

     * If generated from **DESeq2 with raw counts** → include `baseMean` instead of `AveExpr`.
     * If generated from **FPKM / limma** → include `AveExpr`.
     * 👉 Column name should remain as-is (`baseMean` or `AveExpr`). Python side must interpret accordingly.

2. **rank.csv**

   * Columns: `SYMBOL, rank_metric`
   * Recommended metric: `sign(log2FC) * -log10(pvalue)`
   * Purpose: used for gene ranking in GSEA.

3. **fgsea_results.csv**

   * Columns: `pathway, NES, pval, padj, size, leadingEdge, ES`
   * ⚠️ Must contain the **ES column** (for curve reconstruction consistency).

4. **leading_edge.csv**

   * Gene hits for a given pathway.
   * Columns: `SYMBOL, is_hit (1/0)`

5. **es_curve.csv**

   * Running enrichment score curve (for one pathway only).
   * Columns: `rank, ES, hit`
   * ⚠️ One file per pathway (e.g., Apoptosis / PI3K–AKT–mTOR / mTORC1).

---

## ⚙️ Default Rules (Plan A)

* **R output logic**:

  * If starting from raw counts → export `baseMean`.
  * If starting from FPKM/limma → export `AveExpr`.

* **Python receiving logic**:

  * If `baseMean` → apply **log10** for abundance axis.
  * If `AveExpr` → apply **log2** for abundance axis.
  * This ensures consistent MA-plot X-axis interpretation.

---

## 📌 General Notes

* All CSVs must be saved as **UTF-8**, no row names.
* `SYMBOL` must always represent gene names (to avoid ID mismatch).
* Enrichment curves should be exported **one pathway per file**.
* This pack is designed as the handoff between **R (statistical analysis)** and **Python (visualization app / “Painter”)**.

---
