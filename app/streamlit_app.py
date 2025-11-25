import sys
from pathlib import Path

import streamlit as st
import pandas as pd
from PIL import Image
import torch
import clip

# Make project root importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# ---- Import pipeline versions ----
import models.validate_full_pipeline_v5 as v5
import legacy_pipelines.validate_full_pipeline_v2 as v2
import legacy_pipelines.validate_full_pipeline_v4_clip018 as v4

from groundingdino.util.inference import Model


PIPELINES = {
    "v2 – BOX=0.18, CLIP=0.20 (baseline)": {
        "module": v2,
        "box": 0.18,
        "clip": 0.20,
        "desc": "Baseline pipeline with CLIP filtering.",
    },
    "v4 – BOX=0.18, CLIP=0.18 (relaxed CLIP)": {
        "module": v4,
        "box": 0.18,
        "clip": 0.18,
        "desc": "More permissive CLIP threshold, relaxed status logic.",
    },
    "v5 – BOX=0.17, CLIP=0.20 (count-aware)": {
        "module": v5,
        "box": 0.17,
        "clip": 0.20,
        "desc": "Midpoint DINO threshold + count-aware cap per SKU.",
    },
}


# ---------- CACHED HELPERS ----------

@st.cache_resource
def load_models(version_label: str):
    cfg = PIPELINES[version_label]
    mod = cfg["module"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.write(f"Using device: {device}")

    gdino = Model(mod.CONFIG_PATH, mod.CKPT_PATH, device=str(device))
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    return device, gdino, clip_model, preprocess


@st.cache_data
def list_bins(images_dir: Path):
    return sorted(p.stem for p in images_dir.glob("*.jpg"))


def run_inference_for_bin(version_label: str, image_id: str):
    cfg = PIPELINES[version_label]
    mod = cfg["module"]

    device, gdino, clip_model, preprocess = load_models(version_label)

    rows, boxes, labels = mod.process_single_bin(
        image_id=image_id,
        gdino=gdino,
        clip_model=clip_model,
        preprocess=preprocess,
        device=device,
    )
    df = pd.DataFrame(rows)

    vis_path = None
    if boxes:
        suffix = version_label.split(" ")[0]  # "v2", "v4", "v5"
        vis_path = mod.OUT_DIR / f"{image_id}_detected_{suffix}.jpg"
        mod.draw_boxes(mod.IMAGES_DIR / f"{image_id}.jpg", boxes, labels, vis_path)

    return df, vis_path


def build_order_from_bin(df_bin: pd.DataFrame):
    """
    Build an 'invoice' from SKUs in this bin and validate against model predictions.
    User chooses which SKUs to order and quantities.
    """
    st.subheader("2. Create an order (invoice) for this bin")

    skus = (
        df_bin[["asin", "sku_name", "expected", "visible"]]
        .drop_duplicates(subset=["asin"])
        .reset_index(drop=True)
    )

    order_rows = []
    for _, row in skus.iterrows():
        asin = row["asin"]
        name = row["sku_name"]
        expected = int(row["expected"])
        visible = int(row["visible"])

        cols = st.columns([4, 1, 1, 1])
        with cols[0]:
            st.markdown(f"**{name}**  \n_ASIN: {asin}_")
        with cols[1]:
            # User chooses how many they want to order
            qty = st.number_input(
                "Order qty",
                min_value=0,
                max_value=max(expected * 2, 10),
                value=0,
                key=f"qty_{asin}",
            )
        with cols[2]:
            st.text(f"Invoice qty: {expected}")
        with cols[3]:
            st.text(f"Model sees: {visible}")

        if qty > 0:
            order_rows.append(
                {
                    "asin": asin,
                    "sku_name": name,
                    "order_qty": int(qty),
                    "expected": expected,
                    "visible": visible,
                }
            )

    if not order_rows:
        st.info("Set order quantities > 0 to build an order for this bin.")
        return None

    df_order = pd.DataFrame(order_rows)

    def check_line(row):
        vis = int(row["visible"])
        if vis >= row["order_qty"]:
            return "PASS"
        elif vis > 0:
            return "FAIL (too few in bin)"
        else:
            return "FAIL (item not detected)"

    df_order["order_status"] = df_order.apply(check_line, axis=1)
    return df_order


# ---------- MAIN APP ----------

def main():
    st.set_page_config(page_title="Smart Bin Order Validator", layout="wide")
    st.title("📦 Smart Bin Order Validator")
    st.caption(
        "Prototype UI: verify whether items and quantities in an invoice "
        "match what is visible in a bin image."
    )

    st.sidebar.header("Controls")

    # Pick pipeline version
    version_label = st.sidebar.selectbox(
        "Choose pipeline version",
        list(PIPELINES.keys()),
        index=2,  # default to v5
    )
    cfg = PIPELINES[version_label]
    mod = cfg["module"]

    st.sidebar.markdown(
        f"**Selected:** {version_label}\n\n"
        f"- DINO BOX_THRESHOLD = `{cfg['box']}`\n"
        f"- CLIP_THRESHOLD     = `{cfg['clip']}`\n\n"
        f"_{cfg['desc']}_"
    )

    # List bin IDs from that module's IMAGES_DIR
    bin_ids = list_bins(mod.IMAGES_DIR)
    if not bin_ids:
        st.error("No images found in bin-images/.")
        return

    image_id = st.sidebar.selectbox("Choose bin ID", bin_ids, index=0)
    run_btn = st.sidebar.button("Run model on this bin")

    if (
        run_btn
        or "last_bin" not in st.session_state
        or st.session_state.get("last_bin") != image_id
        or st.session_state.get("last_version") != version_label
    ):
        with st.spinner(f"Running {version_label} for bin {image_id}..."):
            df_bin, vis_path = run_inference_for_bin(version_label, image_id)
        st.session_state["df_bin"] = df_bin
        st.session_state["vis_path"] = vis_path
        st.session_state["last_bin"] = image_id
        st.session_state["last_version"] = version_label
    else:
        df_bin = st.session_state.get("df_bin", None)
        vis_path = st.session_state.get("vis_path", None)

    if df_bin is None or df_bin.empty:
        st.warning("No predictions yet. Click 'Run model on this bin' in the sidebar.")
        return

    # Show bin image + detections
    col_img, col_meta = st.columns([2, 3])

    with col_img:
        st.subheader(f"1. Bin image {image_id}")
        img_path = mod.IMAGES_DIR / f"{image_id}.jpg"
        st.image(
            str(img_path),
            caption=f"Bin {image_id} – raw image",
            use_container_width=True,
        )
        if vis_path is not None and vis_path.exists():
            st.image(
                str(vis_path),
                caption=f"Detections ({version_label})",
                use_container_width=True,
            )

    with col_meta:
        st.subheader("Bin-level SKU summary (dataset metadata vs model)")
        st.dataframe(
            df_bin[["asin", "sku_name", "expected", "visible", "status"]],
            use_container_width=True,
            hide_index=True,
        )

    # Build an order from this bin and validate it
    df_order = build_order_from_bin(df_bin)
    if df_order is not None:
        st.subheader("3. Order validation result for this bin")
        st.dataframe(
            df_order[
                ["asin", "sku_name", "order_qty", "visible", "expected", "order_status"]
            ],
            use_container_width=True,
            hide_index=True,
        )

        num_pass = (df_order["order_status"] == "PASS").sum()
        num_total = len(df_order)
        st.markdown(
            f"**Order-level summary:** {num_pass}/{num_total} line items are `PASS`."
        )


if __name__ == "__main__":
    main()
