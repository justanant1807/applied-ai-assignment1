import argparse
import csv
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torchvision.ops import nms

from groundingdino.util.inference import Model, load_image, predict
import clip


# ---------------- CONFIG (V3: DINO-LOOSE) ----------------

CONFIG_PATH = "weights/GroundingDINO_SwinT_OGC.py"
CKPT_PATH = "weights/groundingdino_swint_ogc.pth"

BIN_DIR = Path("bin-images")
META_DIR = Path("metadata")
OUT_DIR = Path("out")

# GroundingDINO thresholds (looser than v2: 0.18 -> 0.15)
BOX_THRESHOLD = 0.15
TEXT_THRESHOLD = 0.15

# CLIP validation (same as v2: slightly relaxed)
CLIP_THRESHOLD = 0.20

# Non-max suppression
IOU_THRESHOLD_PER_SKU = 0.55


# ---------------- HELPERS ----------------


def load_metadata(path: Path):
    """Load metadata from JSON file for a given bin."""
    with open(path, "r") as f:
        meta = json.load(f)

    return [
        {
            "asin": asin,
            "name": data["name"],
            "expected": data["quantity"],
        }
        for asin, data in meta["BIN_FCSKU_DATA"].items()
    ]


def normalize_box(box: torch.Tensor) -> torch.Tensor:
    """Ensure boxes have proper ordering x1<x2, y1<y2."""
    x1, y1, x2, y2 = box.tolist()
    return torch.tensor(
        [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)],
        dtype=box.dtype,
    )


def scale_boxes(boxes, img_w, img_h):
    """Convert normalized bbox -> absolute pixel bbox."""
    if isinstance(boxes, list):
        boxes = torch.stack(boxes)

    scale = torch.tensor([img_w, img_h, img_w, img_h], dtype=boxes.dtype)
    return boxes * scale


def crop_image(img: Image.Image, box: torch.Tensor) -> Image.Image:
    """Crop image safely from bounding box."""
    x1, y1, x2, y2 = map(int, normalize_box(box).tolist())

    # Clamp to image bounds
    x1 = max(0, min(x1, img.width - 1))
    y1 = max(0, min(y1, img.height - 1))
    x2 = max(1, min(x2, img.width))
    y2 = max(1, min(y2, img.height))

    return img.crop((x1, y1, x2, y2))


def draw_boxes(image_path: Path, boxes, labels, out_path: Path):
    """Draw final boxes and save to out/."""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    for box, label in zip(boxes, labels):
        box = normalize_box(box)
        x1, y1, x2, y2 = map(int, box.tolist())
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, max(0, y1 - 14)), label[:30], fill="red")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    print(f"\n🖼 Results saved → {out_path}")


def shorten_for_clip(text: str, max_words: int = 25, max_chars: int = 120) -> str:
    """
    Aggressively shorten a long product title so CLIP's tokenizer never overflows.
    We keep the first N words and then clip by characters.
    """
    words = text.split()
    if len(words) > max_words:
        words = words[:max_words]
    short = " ".join(words)
    if len(short) > max_chars:
        short = short[:max_chars]
    return short


def parse_args():
    p = argparse.ArgumentParser(
        description="Validate SKU detection on bin images (V3: DINO-looser)"
    )
    group = p.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--image-id",
        type=str,
        help="Single image id like 00015 (expects bin-images/<id>.jpg and metadata/<id>.json)",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Process all images in bin-images/",
    )

    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="If used with --all, only process the first N images (e.g. --all --limit 300)",
    )

    # default: quick test on 00015
    p.set_defaults(image_id="00015", all=False)
    return p.parse_args()


# ---------------- CORE PROCESSOR FOR ONE BIN ----------------


def process_single_bin(
    image_id: str,
    gdino,
    clip_model,
    preprocess,
    device: str,
):
    image_path = BIN_DIR / f"{image_id}.jpg"
    meta_path = META_DIR / f"{image_id}.json"
    out_image_path = OUT_DIR / f"{image_id}_detected_v3_dino_loose.jpg"

    if not image_path.exists():
        print(f"[WARN] Image not found, skipping: {image_path}")
        return []

    if not meta_path.exists():
        print(f"[WARN] Metadata not found, skipping: {meta_path}")
        return []

    print("\n========================================")
    print(f"🖼 Processing bin: {image_id}")
    print(f"   Image:    {image_path}")
    print(f"   Metadata: {meta_path}")

    items = load_metadata(meta_path)

    pil_img = Image.open(image_path).convert("RGB")
    img_w, img_h = pil_img.size
    _, img_tensor = load_image(str(image_path))

    final_boxes, final_labels = [], []
    visible_counts = {item["asin"]: 0 for item in items}

    for item in items:
        full_name = item["name"]
        short_name = shorten_for_clip(full_name)
        prompt = f"{short_name} retail product box"

        print("\n------------------------------")
        print(f"🔍 Detecting: {full_name}")
        print(f"📝 Prompt (shortened): {prompt}")

        boxes_norm, logits, _ = predict(
            model=gdino.model,
            image=img_tensor,
            caption=prompt,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            device=device,
        )

        if isinstance(boxes_norm, list):
            count = len(boxes_norm)
        elif isinstance(boxes_norm, torch.Tensor):
            count = boxes_norm.size(0)
        else:
            count = 0

        print(f"📦 Raw detections: {count}")

        if count == 0:
            continue

        boxes_xyxy = scale_boxes(boxes_norm, img_w, img_h)
        logits = torch.tensor([float(v) for v in logits])

        keep = nms(boxes_xyxy, logits, IOU_THRESHOLD_PER_SKU)
        boxes_xyxy = boxes_xyxy[keep]
        logits = logits[keep]

        print(f"✨ After NMS: {len(boxes_xyxy)}")

        with torch.no_grad():
            text_tokens = clip.tokenize([short_name]).to(device)
            text_feat = clip_model.encode_text(text_tokens)
            text_feat /= text_feat.norm(dim=-1, keepdim=True)

        best_score = 0.0

        for i, (box, score) in enumerate(zip(boxes_xyxy, logits)):
            crop = crop_image(pil_img, box)

            if crop.width == 0 or crop.height == 0:
                print(f"   ⚠ Skipping degenerate crop with size {crop.size}")
                continue

            image_input = preprocess(crop).unsqueeze(0).to(device)

            with torch.no_grad():
                img_feat = clip_model.encode_image(image_input)
                img_feat /= img_feat.norm(dim=-1, keepdim=True)
                clip_sim = float((img_feat @ text_feat.T).item())

            print(f"   📍 Box {i+1}: CLIP={clip_sim:.3f} | DINO={score:.3f}")

            if clip_sim >= CLIP_THRESHOLD and clip_sim >= best_score:
                best_score = clip_sim
                visible_counts[item["asin"]] += 1
                final_boxes.append(normalize_box(box))
                final_labels.append(full_name)
                print("      ✔ Accepted")

    if final_boxes:
        draw_boxes(image_path, final_boxes, final_labels, out_image_path)
    else:
        print("\n❌ No valid detections passed filtering.")

    print("\n📊 SKU VERIFICATION SUMMARY (V3 DINO-loose):\n")
    rows = []
    for item in items:
        asin = item["asin"]
        expected = item["expected"]
        visible = visible_counts[asin]

        # same relaxed logic as v2 (±1)
        if visible == 0:
            status = "NOT_FOUND"
            pretty = "❌ NOT FOUND"
        elif abs(visible - expected) <= 1:
            status = "FULL_MATCH"
            pretty = "✅ FULL MATCH (±1)"
        elif visible < expected:
            status = "PARTIAL_OCCLUDED"
            pretty = "⚠ PARTIAL — LIKELY OCCLUDED"
        else:
            status = "OVERCOUNT"
            pretty = "⚠ OVERCOUNT (Possible false positives)"

        print(
            f"- {item['name']}: visible={visible} / expected={expected} → {pretty}"
        )

        rows.append(
            {
                "image_id": image_id,
                "asin": asin,
                "sku_name": item["name"],
                "expected": expected,
                "visible": visible,
                "status": status,
            }
        )

    return rows


# ---------------- MAIN ----------------


def main():
    args = parse_args()

    OUT_DIR.mkdir(exist_ok=True, parents=True)

    if args.all:
        image_ids = sorted(p.stem for p in BIN_DIR.glob("*.jpg"))
        if not image_ids:
            print("[ERROR] No images found in bin-images/")
            return

        if args.limit is not None:
            image_ids = image_ids[:args.limit]
            print(f"📂 Running on first {len(image_ids)} bins (limit={args.limit})\n")
        else:
            print(f"📂 Running on ALL bins: {len(image_ids)} images\n")
    else:
        image_ids = [args.image_id]
        print(f"📂 Running on single bin: {image_ids[0]}\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🧠 Using device: {device}")

    gdino = Model(CONFIG_PATH, CKPT_PATH, device=device)
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    summary_path = OUT_DIR / "summary_v3_dino_loose.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_id",
                "asin",
                "sku_name",
                "expected",
                "visible",
                "status",
            ],
        )
        writer.writeheader()

        total_rows = 0
        for image_id in image_ids:
            rows = process_single_bin(
                image_id=image_id,
                gdino=gdino,
                clip_model=clip_model,
                preprocess=preprocess,
                device=device,
            )
            for r in rows:
                writer.writerow(r)
            total_rows += len(rows)

    print("\n✅ Done (V3 DINO-loose).")
    print(f"🧾 Wrote {total_rows} summary rows to: {summary_path}")


if __name__ == "__main__":
    main()
