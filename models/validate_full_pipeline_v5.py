import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torchvision.ops import nms

import pandas as pd

from groundingdino.util.inference import Model, load_image, predict
import clip


# ---------------- CONFIG ----------------

# Your repo has these in the weights/ folder
CONFIG_PATH = "weights/GroundingDINO_SwinT_OGC.py"
CKPT_PATH = "weights/groundingdino_swint_ogc.pth"

IMAGES_DIR = Path("bin-images")
META_DIR = Path("metadata")
OUT_DIR = Path("out")

# GroundingDINO thresholds
# v2/v4: 0.18, v3: ~0.16, v5: midpoint
BOX_THRESHOLD = 0.17
TEXT_THRESHOLD = 0.18

# CLIP validation
# v2: 0.20, v4: 0.18, v5: 0.20 (stricter again)
CLIP_THRESHOLD = 0.20

# Non-max suppression
IOU_THRESHOLD_PER_SKU = 0.55

# Count post-processing: cap accepted boxes per SKU to expected + this tolerance
COUNT_TOLERANCE = 1

# Output summary for this version
SUMMARY_CSV = OUT_DIR / "summary_v5_clip020_box017.csv"


# ---------------- HELPERS ----------------

def load_metadata(path: Path):
    """Load metadata from JSON and return list of SKUs."""
    with open(path, "r") as f:
        meta = json.load(f)

    return [
        {
            "asin": asin,
            "name": data["name"],
            "expected": int(data["quantity"]),
        }
        for asin, data in meta["BIN_FCSKU_DATA"].items()
    ]


def normalize_box(box: torch.Tensor) -> torch.Tensor:
    """Ensure boxes have proper ordering x1<x2, y1<y2."""
    x1, y1, x2, y2 = box.tolist()
    return torch.tensor(
        [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)],
        dtype=box.dtype,
        device=box.device,
    )


def scale_boxes(boxes, img_w, img_h):
    """Convert normalized bbox -> absolute pixel bbox."""
    if isinstance(boxes, list):
        boxes = torch.stack(boxes)

    scale = torch.tensor(
        [img_w, img_h, img_w, img_h], dtype=boxes.dtype, device=boxes.device
    )
    return boxes * scale


def crop_image(img: Image.Image, box: torch.Tensor) -> Image.Image:
    """Crop image safely from bounding box."""
    box = normalize_box(box).cpu()
    x1, y1, x2, y2 = map(int, box.tolist())
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img.width, x2)
    y2 = min(img.height, y2)
    if x2 <= x1 or y2 <= y1:
        # degenerate crop; caller should handle
        return Image.new("RGB", (0, 0))
    return img.crop((x1, y1, x2, y2))


def draw_boxes(image_path: Path, boxes, labels, out_path: Path):
    """Draw boxes + labels and save image."""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    for box, label in zip(boxes, labels):
        box = normalize_box(box).cpu()
        x1, y1, x2, y2 = map(int, box.tolist())
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, max(0, y1 - 14)), label[:40], fill="red")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    print(f"\n🖼 Results saved → {out_path}")


def make_clip_short_name(full_name: str, max_words: int = 16) -> str:
    """
    Shorten long product titles so CLIP tokenize doesn't overflow context.
    Rough heuristic: keep first N words, strip brackets.
    """
    cleaned = full_name.replace("(", " ").replace(")", " ")
    tokens = cleaned.split()
    short = " ".join(tokens[:max_words])
    if not short:
        short = full_name[:60]
    return short


def status_from_counts(expected: int, visible: int):
    """
    Relaxed v2/v4 logic: ±1 is treated as FULL_MATCH.
    """
    if visible == 0:
        return "NOT_FOUND", "❌ NOT FOUND"
    elif abs(visible - expected) <= 1:
        return "FULL_MATCH", "✅ FULL MATCH (±1)"
    elif visible < expected:
        return "PARTIAL_OCCLUDED", "⚠ PARTIAL — LIKELY OCCLUDED"
    else:
        return "OVERCOUNT", "⚠ OVERCOUNT (Possible false positives)"


# ---------------- CORE BIN PROCESSING ----------------

def process_single_bin(
    image_id: str,
    gdino: Model,
    clip_model,
    preprocess,
    device: torch.device,
):
    """
    Process one bin (one image + one metadata JSON).
    Returns:
        rows: list of dicts (for CSV summary)
        final_boxes, final_labels: for visualisation
    """
    img_path = IMAGES_DIR / f"{image_id}.jpg"
    meta_path = META_DIR / f"{image_id}.json"

    print("========================================")
    print(f"🖼 Processing bin: {image_id}")
    print(f"   Image:    {img_path}")
    print(f"   Metadata: {meta_path}\n")

    if not img_path.exists() or not meta_path.exists():
        print("❌ Missing image or metadata, skipping.")
        return [], [], []

    items = load_metadata(meta_path)
    pil_img = Image.open(img_path).convert("RGB")
    img_w, img_h = pil_img.size

    _, img_tensor = load_image(str(img_path))

    visible_counts = {item["asin"]: 0 for item in items}
    final_boxes = []
    final_labels = []

    rows = []

    for item in items:
        asin = item["asin"]
        sku_name = item["name"]
        expected = int(item["expected"])

        prompt = f"{sku_name} retail product box"
        short_name = make_clip_short_name(sku_name)

        print("------------------------------")
        print(f"🔍 Detecting: {sku_name}")
        print(f"📝 Prompt: {prompt}")

        # GroundingDINO prediction (force CPU device on Mac)
        boxes_norm, logits, _ = predict(
            model=gdino.model,
            image=img_tensor,
            caption=prompt,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            device=str(device),  # <- IMPORTANT: 'cpu' on your Mac
        )

        if isinstance(boxes_norm, list):
            num_raw = len(boxes_norm)
        else:
            num_raw = boxes_norm.size(0)

        print(f"📦 Raw detections: {num_raw}")

        if num_raw == 0:
            # no candidates at all
            rows.append(
                {
                    "image_id": image_id,
                    "asin": asin,
                    "sku_name": sku_name,
                    "expected": expected,
                    "visible": 0,
                    "status": "NOT_FOUND",
                    "pretty_status": "❌ NOT FOUND",
                }
            )
            continue

        # scale + NMS
        boxes_xyxy = scale_boxes(boxes_norm, img_w, img_h)
        logits_t = torch.tensor(
            [float(v) for v in logits], dtype=torch.float32, device=boxes_xyxy.device
        )

        keep = nms(boxes_xyxy, logits_t, IOU_THRESHOLD_PER_SKU)
        boxes_xyxy = boxes_xyxy[keep]
        logits_t = logits_t[keep]

        print(f"✨ After NMS: {boxes_xyxy.size(0)}")

        # CLIP text features (once per SKU) using short_name to avoid overflow
        with torch.no_grad():
            text_tokens = clip.tokenize([short_name]).to(device)
            text_feat = clip_model.encode_text(text_tokens)
            text_feat /= text_feat.norm(dim=-1, keepdim=True)

        # ----- CLIP: collect candidates for this SKU (v5) -----
        candidates = []  # list of (clip_sim, box)

        for i, (box, score) in enumerate(zip(boxes_xyxy, logits_t)):
            crop = crop_image(pil_img, box)
            if crop.width <= 0 or crop.height <= 0:
                print(f"   ⚠ Skipping degenerate crop with size {crop.size}")
                continue

            image_input = preprocess(crop).unsqueeze(0).to(device)

            with torch.no_grad():
                img_feat = clip_model.encode_image(image_input)
                img_feat /= img_feat.norm(dim=-1, keepdim=True)
                clip_sim = float((img_feat @ text_feat.T).item())

            print(f"   📍 Box {i+1}: CLIP={clip_sim:.3f} | DINO={float(score):.3f}")

            if clip_sim >= CLIP_THRESHOLD:
                candidates.append((clip_sim, box))

        # ----- Count-aware post-filtering (new in v5) -----
        if candidates:
            # sort by CLIP similarity, best first
            candidates.sort(key=lambda x: x[0], reverse=True)

            # cap number of accepted boxes to expected + tolerance
            max_keep = min(len(candidates), expected + COUNT_TOLERANCE)
            chosen = candidates[:max_keep]

            for clip_sim, box in chosen:
                visible_counts[asin] += 1
                final_boxes.append(normalize_box(box).cpu())
                final_labels.append(short_name)
                print(f"      ✔ Accepted (kept for counting)  CLIP={clip_sim:.3f}")
        else:
            print("   (no CLIP candidates passed threshold)")

    # ---------- SUMMARY WITH OCCLUSION / COUNT LOGIC ----------
    print("\n📊 SKU VERIFICATION SUMMARY:\n")

    for item in items:
        asin = item["asin"]
        sku_name = item["name"]
        expected = int(item["expected"])
        visible = int(visible_counts[asin])

        status, pretty = status_from_counts(expected, visible)

        print(f"- {sku_name}: visible={visible} / expected={expected} → {pretty}")

        rows.append(
            {
                "image_id": image_id,
                "asin": asin,
                "sku_name": sku_name,
                "expected": expected,
                "visible": visible,
                "status": status,
                "pretty_status": pretty,
            }
        )

    return rows, final_boxes, final_labels


# ---------------- MAIN ----------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="v5: GroundingDINO + CLIP SKU verification with count-aware cap."
    )
    parser.add_argument(
        "--image-id",
        type=str,
        help="Process a single bin id, e.g. 00015",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all bins under bin-images/ and metadata/.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit when using --all (e.g. first 300 bins).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # MacBook Pro: CPU-only
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🧠 Using device: {device}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # load models on chosen device (cpu for you)
    gdino = Model(CONFIG_PATH, CKPT_PATH, device=str(device))
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    all_rows = []

    # ---------- SINGLE BIN MODE ----------
    if args.image_id and not args.all:
        image_id = args.image_id
        print(f"📂 Running on single bin: {image_id}\n")

        rows, boxes, labels = process_single_bin(
            image_id=image_id,
            gdino=gdino,
            clip_model=clip_model,
            preprocess=preprocess,
            device=device,
        )
        all_rows.extend(rows)

        # draw visualisation for single bin
        if boxes:
            vis_out = OUT_DIR / f"{image_id}_detected_v5.jpg"
            draw_boxes(IMAGES_DIR / f"{image_id}.jpg", boxes, labels, vis_out)
        else:
            print("\n❌ No valid detections passed filtering → no visualisation image.")

    # ---------- ALL BINS MODE ----------
    elif args.all:
        im_paths = sorted(IMAGES_DIR.glob("*.jpg"))
        image_ids = [p.stem for p in im_paths]

        if args.limit is not None:
            image_ids = image_ids[: args.limit]
            print(f"📂 Running on first {len(image_ids)} bins (limit={args.limit})\n")
        else:
            print(f"📂 Running on all {len(image_ids)} bins\n")

        for image_id in image_ids:
            rows, boxes, labels = process_single_bin(
                image_id=image_id,
                gdino=gdino,
                clip_model=clip_model,
                preprocess=preprocess,
                device=device,
            )
            all_rows.extend(rows)

            # optional: save per-bin visualisation only if there are boxes
            if boxes:
                vis_out = OUT_DIR / f"{image_id}_detected_v5.jpg"
                draw_boxes(IMAGES_DIR / f"{image_id}.jpg", boxes, labels, vis_out)

    else:
        print("❌ You must specify either --image-id or --all")
        return

    # ---------- WRITE SUMMARY CSV ----------
    if all_rows:
        df = pd.DataFrame(all_rows)
        SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(SUMMARY_CSV, index=False)
        print(f"\n✅ Done.")
        print(f"🧾 Wrote {len(df)} summary rows to: {SUMMARY_CSV}")
    else:
        print("\n❌ No rows collected; nothing to write.")


if __name__ == "__main__":
    main()
