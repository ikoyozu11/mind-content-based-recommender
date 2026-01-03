import pandas as pd
import streamlit as st


def render_metrics_cards(metrics_df: pd.DataFrame):
    if metrics_df is None or metrics_df.empty:
        st.warning("metrics.csv kosong atau tidak ditemukan.")
        return

    m = metrics_df.iloc[0].to_dict()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Threshold", f"{m.get('threshold', 0):.3f}")
    c2.metric("Accuracy", f"{m.get('accuracy', 0):.4f}")
    c3.metric("AUC", f"{m.get('auc', 0):.4f}")
    c4.metric("F1 (class 1)", f"{m.get('f1_score', 0):.4f}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precision (class 1)", f"{m.get('precision', 0):.4f}")
    c2.metric("Recall (class 1)", f"{m.get('recall', 0):.4f}")

    if "pr_auc" in m:
        c3.metric("PR-AUC", f"{m.get('pr_auc', 0):.4f}")
    c4.metric("Rows eval", f"{int(m.get('rows', 0)):,}")


def render_recs_table(recs_rows):
    if not recs_rows:
        st.info("Belum ada rekomendasi.")
        return

    df = pd.DataFrame(recs_rows)
    cols = [c for c in ["rank", "news_id", "score", "pred_label", "category", "subcategory", "title"] if c in df.columns]
    st.dataframe(df[cols], use_container_width=True, hide_index=True)