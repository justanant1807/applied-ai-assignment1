# Applied AI for Industry – Assignment 1  
## Smart Bin Item & Quantity Verifier

This repository contains a working prototype of a **computer vision system** that:

- Takes a **bin image** and its **metadata / invoice**.
- Uses **GroundingDINO + CLIP** to detect SKUs and count visible units.
- Verifies whether the **items and quantities** in an order match what is visible in the bin.
- Exposes a simple **Streamlit web UI** for interactive validation and demo.
- Includes evaluation scripts and plots to quantify performance.

The system is designed to run on **CPU** (no GPU required), both locally and—partially—on Streamlit Cloud.

---

## 1. Project structure

```text
Assignment_1/
  app/
    __init__.py
    streamlit_app.py              # Streamlit UI (main app entrypoint)

  bin-images/                     # Bin photos (dataset – usually local only)
    00001.jpg
    00002.jpg
    ...

  metadata/                       # Per-bin JSON with ASIN, SKU name, quantity
    00001.json
    00002.json
    ...

  models/
    __init__.py
    validate_full_pipeline_v5.py  # FINAL CV pipeline (v5: BOX=0.17, CLIP=0.20)

  legacy_pipelines/
    __init__.py
    validate_full_pipeline_v2.py          # Older strict baseline
    validate_full_pipeline_v4_clip018.py  # Relaxed CLIP baseline

  mlops/
    __init__.py
    evaluate_model_v3_counts.py   # Evaluation + count-style confusion metrics
    plot_results.py               # CLI plots for metrics

  notebooks/
    01_eda_dataset.ipynb          # (optional) EDA notebook
    02_threshold_sweeps.ipynb     # (optional) threshold tuning
    03_eval_plots.ipynb           # (optional) pretty figures for report

  out/
    summary_v2_relaxed.csv
    summary_v3_*.csv
    summary_v4_*.csv
    summary_v5_clip020_box017.csv # Final metrics (from v5)
    *_detected_*.jpg              # Visualisations with bounding boxes
    inference_log.csv             # (optional) inference logging

  weights/
    GroundingDINO_SwinT_OGC.py    # GroundingDINO config (committed)
    groundingdino_swint_ogc.pth   # GroundingDINO checkpoint (local; NOT in repo if too big)

  requirements.txt
  README.md
  .gitignore
  "Assignment #1 [Applied AI for Industry] (1).pdf"

> **Important:** For size reasons, the full dataset and large model weights are usually **not** committed to the public GitHub repo.\
> They must be placed locally under `bin-images/`, `metadata/`, and `weights/` as described below.

* * * * *

2\. Prerequisites
-----------------

-   **OS:** Windows / macOS / Linux

-   **Python:** 3.10 or 3.11

-   **Git** installed (for cloning)

-   **GPU:** *Not required* -- everything runs on **CPU** (slower but fine for this assignment).

* * * * *

3\. Local setup (step-by-step)
------------------------------

### 3.1. Clone the repo

`git clone https://github.com/justanant1807/applied-ai-assignment1.git
cd applied-ai-assignment1`

(If you already have the `Assignment_1` folder, just `cd` into it.)

### 3.2. Create and activate a virtual environment

**Windows (PowerShell):**

`python -m venv .venv
.\.venv\Scripts\activate`

**macOS / Linux:**

`python -m venv .venv
source .venv/bin/activate`

You should now see `(.venv)` at the start of your terminal prompt.

### 3.3. Install dependencies

`pip install -r requirements.txt`

Key packages:

-   `torch`, `torchvision`, `timm` -- deep learning + vision backbones

-   `groundingdino-py` -- GroundingDINO model wrapper

-   `git+https://github.com/openai/CLIP.git` -- OpenAI CLIP

-   `opencv-python-headless`, `Pillow`, `numpy`, `pandas`, `matplotlib`

-   `streamlit`, `tqdm`, `scikit-image`, etc.

* * * * *

4\. Data and model weights
--------------------------

### 4.1. Bin images & metadata (local)

Obtain the dataset from the course / assignment link (bin images + metadata JSONs), then:

-   Place **bin images** in:

    `bin-images/
        00001.jpg
        00002.jpg
        ...`

-   Place **metadata JSONs** in:

    `metadata/
        00001.json
        00002.json
        ...`

The file names must match: `00015.jpg` ↔ `00015.json`.

> **Note:** For size reasons, the full dataset may **not** be included in the public GitHub repo.\
> Only a small subset of images/JSONs (if any) may be committed.\
> For full experiments, download the dataset locally and populate `bin-images/` and `metadata/` as above.

### 4.2. GroundingDINO weights (local)

The project uses the GroundingDINO Swin-T model.

In the repo structure you should have (on your machine):

`weights/
    GroundingDINO_SwinT_OGC.py     # model config (committed to GitHub)
    groundingdino_swint_ogc.pth    # model checkpoint (NOT committed if >100MB)`

Because the `.pth` checkpoint is large, it is typically **not tracked** in the public repo.

To run the model **locally**:

1.  Download `groundingdino_swint_ogc.pth` from the course resources or from the original GroundingDINO release.

2.  Place it exactly at:

    `weights/groundingdino_swint_ogc.pth`

If this file is missing, any script that constructs the model will raise a `FileNotFoundError`.

* * * * *

5\. How to run the model locally
--------------------------------

### 5.1. Final CV pipeline v5 -- single bin

With your venv active and data + weights in place:

`python models/validate_full_pipeline_v5.py --image-id 00015`

This will:

-   Load `bin-images/00015.jpg` and `metadata/00015.json`.

-   For each SKU in the JSON:

    -   Run GroundingDINO with text prompt:

        -   `BOX_THRESHOLD = 0.17`

        -   `TEXT_THRESHOLD = 0.18`

    -   Apply NMS and CLIP validation (`CLIP_THRESHOLD = 0.20`).

    -   Count `visible` units that pass both DINO + CLIP filters.

-   Print a **SKU verification summary**, for example:

    `- Some SKU: visible=3 / expected=3 → ✅ FULL MATCH
    - Another : visible=1 / expected=4 → ⚠ PARTIAL --- LIKELY OCCLUDED`

-   Save a visualisation image with bounding boxes:

    `out/00015_detected_v5.jpg`

-   Append a row for each SKU to:

    `out/summary_v5_clip020_box017.csv`

### 5.2. Run on all bins (or first N bins)

`# All bins with metadata
python models/validate_full_pipeline_v5.py --all

# Or limit to first N bins for quicker experiments
python models/validate_full_pipeline_v5.py --all --limit 300`

This builds a full summary CSV used for evaluation.

* * * * *

6\. Evaluation & plots (local)
------------------------------

Once you have `out/summary_v5_clip020_box017.csv`, you can compute metrics.

### 6.1. Evaluate model performance

`python mlops/evaluate_model_v3_counts.py --summary out/summary_v5_clip020_box017.csv`

This prints:

#### a) Presence metrics (SKU present vs. not)

-   Ground truth presence: `expected > 0`

-   Predicted presence: `visible > 0`

You get:

-   TP, FP, FN, TN

-   Accuracy, Precision, Recall, F1-score

Note: Because metadata describes only SKUs that are actually in the bin, FP/TN for presence are often 0.

#### b) Count metrics

-   SKU-level MAE: `mean(|visible - expected|)`

-   SKU-level RMSE

-   Total **over-count mass**: `sum max(visible - expected, 0)`

-   Total **under-count mass**: `sum max(expected - visible, 0)`

-   Bin-level exact accuracy: fraction of bins where *all* SKUs match exactly.

#### c) Count-style "confusion"

Treat counts like this:

-   `TP_exact` -- SKUs where `visible == expected`

-   `FP_over` -- "count false positives": `visible > expected`

-   `FN_under` -- "count false negatives": `visible < expected`

This tells you how often the model **overestimates** vs **underestimates** counts.

#### d) Status label distribution

Using relaxed status logic:

-   `NOT_FOUND` -- visible == 0

-   `FULL_MATCH` -- `abs(visible - expected) <= 1` (count within ±1)

-   `PARTIAL_OCCLUDED` -- visible < expected - 1 (under by 2+)

-   `OVERCOUNT` -- visible > expected + 1 (over by 2+)

The script prints the number and percentage of rows per status.

### 6.2. Plots for the report

`python mlops/plot_results.py --summary out/summary_v5_clip020_box017.csv`

This generates matplotlib plots such as:

-   Histogram of absolute count errors.

-   Bar plot of status distribution.

-   Bar plot for count-style confusion (`TP_exact`, `FP_over`, `FN_under`).

-   (Optionally) comparisons across multiple pipeline versions if you pass multiple summary files.

These figures can be used directly in your assignment report.

* * * * *

7\. Streamlit app -- Smart Bin Order Validator (local)
-----------------------------------------------------

The Streamlit app provides a user-friendly UI to:

-   Select a **pipeline version** (v2 / v4 / v5).

-   Select a **bin ID**.

-   Run the model and visualise detections.

-   Build a **mock order** and validate whether requested items/quantities are available.

### 7.1. Running the app locally

With venv active:

`streamlit run app/streamlit_app.py`

Then open:

`http://localhost:8501`

(Usually opens automatically.)

### 7.2. UI behaviour

**Sidebar:**

-   **Pipeline version selector**, e.g.:

    -   `v2 -- BOX=0.18, CLIP=0.20 (baseline)`

    -   `v4 -- BOX=0.18, CLIP=0.18 (more recall)`

    -   `v5 -- BOX=0.17, CLIP=0.20 (count-aware final)`

-   Shows the DINO `BOX_THRESHOLD` and `CLIP_THRESHOLD` for each version.

-   **Bin ID selector**: choose a bin like `00001`, `00015`, etc.

-   **"Run model on this bin"** button.

**Main content:**

1.  Shows selected **version**, **bin**, and **device** (`cpu`).

2.  Displays:

    -   Raw bin image.

    -   Detection overlay (`*_detected_*.jpg`) with bounding boxes + SKU labels.

3.  Shows a **SKU table**:

    -   `asin`, `sku_name`, `expected`, `visible`, `status`.

4.  **Order Builder**:

    -   For each SKU, you can enter an **order quantity**.

    -   The app checks if `order_qty <= visible` and shows:

        -   `PASS`

        -   `FAIL (too few in bin)`

        -   `FAIL (item not detected)` if visible == 0.

5.  **Order-level summary**:

    -   e.g. "3/5 line items PASS".

This matches the assignment's requirement:

> Allow the user to select an item and quantity, select the relevant bin image, and validate whether items and quantities in the order exist in the bin image using the CV model.

* * * * *

8\. Running on the cloud (Streamlit Cloud)
------------------------------------------

Because the GroundingDINO checkpoint file (`weights/groundingdino_swint_ogc.pth`) is **large**, it is normally **not** stored in the public GitHub repo. This has consequences for how the app behaves on Streamlit Cloud.

### 8.1. Basic cloud deployment (UI demo, no heavy model)

This is the simplest and most realistic setup:

1.  Ensure your code + `GroundingDINO_SwinT_OGC.py` (config) are pushed to GitHub:

    `git add .
    git commit -m "Update code and README"
    git push`

2.  Go to **<https://share.streamlit.io>** (Streamlit Community Cloud).

3.  Click **"Create app"** → "From existing repo".

4.  Choose:

    -   Repo: `justanant1807/applied-ai-assignment1`

    -   Branch: `main`

    -   Main file: `app/streamlit_app.py`

    -   Python version: **3.10**

5.  Deploy.

Since `groundingdino_swint_ogc.pth` is not present in the cloud filesystem:

-   The app's `load_models()` function will detect that the checkpoint is missing.

-   It shows a **clear error message** in the UI:

    > GroundingDINO weights not found.\
    > This Streamlit Cloud demo does **not** include the large `weights/groundingdino_swint_ogc.pth` file.\
    > The full model runs locally with the weights and full dataset installed, as described in the README.

This satisfies:

-   **"Working prototype"** → fully demonstrated **locally** (with screen recording).

-   **Cloud deployment** → UI reachable online, even if heavy model cannot be loaded.

> All **quantitative results** and full-scale inference are based on **local runs** with the weights and full dataset. The cloud app is a visual/demo front-end.

### 8.2. (Optional, advanced) Cloud with full model

If and only if you want to attempt running the **full model on the cloud** (not necessary for the assignment), you have two options:

#### Option A -- Commit checkpoint if it's small enough

If `weights/groundingdino_swint_ogc.pth` is **< ~90--95 MB**:

1.  Remove `weights/` from `.gitignore` (if present) or force-add:

    `git add weights/groundingdino_swint_ogc.pth
    git commit -m "Add GroundingDINO checkpoint"
    git push`

2.  Redeploy / restart the app on Streamlit Cloud.

If GitHub rejects the push due to file size, this option isn't viable.

#### Option B -- Download checkpoint at runtime

If you host `groundingdino_swint_ogc.pth` somewhere else (e.g. private Google Drive, S3):

-   You can modify `validate_full_pipeline_v5.py` (or a small helper) to:

    `from pathlib import Path
    import urllib.request

    CKPT_PATH = ROOT / "weights" / "groundingdino_swint_ogc.pth"

    def ensure_weights():
        if not CKPT_PATH.exists():
            CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
            url = "https://your-hosted-url/groundingdino_swint_ogc.pth"
            urllib.request.urlretrieve(url, CKPT_PATH)`

-   Call `ensure_weights()` before creating the `Model`.

However, this will:

-   Increase app startup time,

-   Use more memory and bandwidth,

-   Still be CPU-only and relatively slow.

For this coursework, this complexity is **not required** and is documented here only as a possible extension.

* * * * *

9\. How to reproduce results (summary)
--------------------------------------

**Locally (recommended, full system):**

1.  Clone repo and create venv.

2.  Install dependencies: `pip install -r requirements.txt`.

3.  Download dataset into `bin-images/` and `metadata/`.

4.  Download `groundingdino_swint_ogc.pth` and place in `weights/`.

5.  Run the final pipeline:

    `python models/validate_full_pipeline_v5.py --all`

6.  Evaluate:

    `python mlops/evaluate_model_v3_counts.py --summary out/summary_v5_clip020_box017.csv`

7.  Generate plots:

    `python mlops/plot_results.py --summary out/summary_v5_clip020_box017.csv`

8.  Launch the Streamlit UI:

    `streamlit run app/streamlit_app.py`

**On Streamlit Cloud (UI demo):**

-   Deploy repo as described in **Section 8**.

-   The app will show a friendly message if the heavy model weights are not present.

-   Use the local run + screen recording to demonstrate the **full CV system** for grading.

This setup gives you:

-   A complete **local prototype** (model + evaluation + UI).

-   A clean **cloud-hosted UI/demo** with clear limitations due to weight size.

 `Use this as your final `README.md`, then:

```bash
git add README.md
git commit -m "Update README with local and cloud run instructions"
git push`
