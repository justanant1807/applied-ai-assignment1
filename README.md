# Applied AI for Industry – Assignment 1  
## Smart Bin Item & Quantity Verifier

This repository contains a working prototype of a **computer vision system** that:
- Takes a **bin image** and its **metadata / invoice**.
- Uses **GroundingDINO + CLIP** to detect SKUs and count visible units.
- Verifies whether the **items and quantities** in an order match what is visible in the bin.
- Exposes a simple **Streamlit web UI** for interactive validation and demo.

the following is the streamlit link : https://applied-ai-assignment1-yksxtsbn2qrtwazp9zxbym.streamlit.app/

---

## 1. Project structure

```text
Assignment_1/
  app/
    __init__.py
    streamlit_app.py          # Streamlit UI (main app entrypoint)

  models/
    __init__.py
    validate_full_pipeline_v5.py   # Final CV pipeline (v5: BOX=0.17, CLIP=0.20)

  legacy_pipelines/
    __init__.py
    validate_full_pipeline_v2.py       # Older baselines
    validate_full_pipeline_v4_clip018.py

  mlops/
    __init__.py
    evaluate_model_v3_counts.py    # Evaluation + count-style confusion
    plot_results.py                # CLI plots for metrics

  bin-images/                      # Bin photos (not all committed to GitHub)
  metadata/                        # Per-bin JSON with ASIN, SKU name, quantity
  weights/
    groundingdino_swint_ogc.pth    # GroundingDINO weights (not in repo if >100MB)
    GroundingDINO_SwinT_OGC.py     # Model config

  out/
    summary_*.csv                  # Evaluation summaries
    *_detected_*.jpg               # Visualisations with bounding boxes
    inference_log.csv (optional)   # Inference logging

  requirements.txt
  README.md
  "Assignment #1 [Applied AI for Industry] (1).pdf"


2\. Prerequisites
-----------------

-   **OS:** Windows / macOS / Linux

-   **Python:** 3.10 or 3.11 (tested on CPU)

-   **Git** installed (for cloning)

-   GPU is *not* required; the code runs on **CPU** (slower but works).

* * * * *

3\. Setup (local)
-----------------

### 3.1. Clone the repo

`git clone https://github.com/justanant1807/applied-ai-assignment1.git
cd applied-ai-assignment1`

If you are already working inside `Assignment_1`, you can skip the clone step; just make sure you are in that folder.

### 3.2. Create and activate virtual environment

**Windows (PowerShell):**

`python -m venv .venv
.\.venv\Scripts\activate`

**macOS / Linux:**

`python -m venv .venv
source .venv/bin/activate`

### 3.3. Install dependencies

`pip install -r requirements.txt`

This installs:

-   `torch`, `torchvision` (CPU)

-   `groundingdino-py`

-   `git+https://github.com/openai/CLIP.git`

-   `streamlit`, `pandas`, `matplotlib`, etc.

* * * * *

4\. Data and model weights
--------------------------

### 4.1. Bin images & metadata

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

### 4.2. GroundingDINO weights

Download `groundingdino_swint_ogc.pth` from the course resources (or the original GroundingDINO release) and place it here:

`weights/groundingdino_swint_ogc.pth
weights/GroundingDINO_SwinT_OGC.py`

If the `weights/` folder does not exist, create it and copy both files.

> The model will **not** run without this `.pth` file.

* * * * *

5\. Running the CV pipeline from CLI
------------------------------------

### 5.1. Single-bin run (final pipeline v5)

From the project root, with venv activated:

`python models/validate_full_pipeline_v5.py --image-id 00015`

This will:

-   Load `bin-images/00015.jpg` and `metadata/00015.json`.

-   Run GroundingDINO + CLIP for each SKU in the metadata.

-   Apply NMS and CLIP similarity to select boxes.

-   Count `visible` units per SKU.

-   Print a **SKU verification summary** like:

    `- Some SKU name: visible=3 / expected=3 → ✅ FULL MATCH
    - Another SKU : visible=1 / expected=4 → ⚠ PARTIAL --- LIKELY OCCLUDED`

-   Save a visualisation image with boxes:

    `out/00015_detected_v5.jpg`

-   Append summary rows to a CSV:

    `out/summary_v5_clip020_box017.csv`

### 5.2. Run over all bins

`python models/validate_full_pipeline_v5.py --all
# or limit to first N bins
python models/validate_full_pipeline_v5.py --all --limit 300`

* * * * *

6\. Evaluation & plots (MLOps)
------------------------------

Once you have a summary CSV (e.g. `out/summary_v5_clip020_box017.csv`) you can compute metrics.

### 6.1. Evaluate model performance

`python mlops/evaluate_model_v3_counts.py --summary out/summary_v5_clip020_box017.csv`

This prints:

-   **Presence metrics** (SKU present vs not):\
    Accuracy, Precision, Recall, F1, TP / FN etc.

-   **Count metrics**:\
    MAE, RMSE, total over-count and under-count mass, bin-level accuracy.

-   **Count-style confusion**:

    -   `TP_exact` -- exact matches (`visible == expected`)

    -   `FP_over` -- count false positives (`visible > expected`)

    -   `FN_under` -- count false negatives (`visible < expected`)

-   Distribution of status labels (`FULL_MATCH`, `PARTIAL_OCCLUDED`, `OVERCOUNT`, `NOT_FOUND`).

### 6.2. Plotting results

`python mlops/plot_results.py --summary out/summary_v5_clip020_box017.csv`

This generates matplotlib figures (e.g. error histograms, confusion-style bar charts) for use in the report.

* * * * *

7\. Running the Streamlit app locally
-------------------------------------

The app provides a simple web UI called **"Smart Bin Order Validator"**.

From the project root with venv active:

`streamlit run app/streamlit_app.py`

Then open `http://localhost:8501` in your browser (Streamlit usually opens it automatically).

### 7.1. How to use the UI

1.  **Select pipeline version** (sidebar):

    -   `v2 -- BOX=0.18, CLIP=0.20 (baseline)`

    -   `v4 -- BOX=0.18, CLIP=0.18 (relaxed CLIP)`

    -   `v5 -- BOX=0.17, CLIP=0.20 (count-aware)` ← final version

2.  **Choose bin ID** (sidebar):

    -   Pick one of the available IDs, e.g. `00015`.

3.  Click **"Run model on this bin"**.

4.  The main page shows:

    -   Raw bin image.

    -   Detection overlay image (bounding boxes and SKU labels).

    -   SKU-level table: `asin`, `sku_name`, `expected`, `visible`, `status`.

5.  **Order builder section**:

    -   For each SKU in that bin, you can set an **order quantity**.

    -   For each order line, the app compares:

        -   `order_qty` (user),

        -   `visible` (model),

        -   `expected` (metadata / invoice),

    -   And reports:

        -   `PASS`

        -   `FAIL (too few in bin)`

        -   `FAIL (item not detected)`

6.  At the bottom you get an **order-level summary**:

    -   e.g. "3/5 line items are PASS".

This matches the assignment requirement: *"Allow the user to select an item and its quantity... select the appropriate bin image... use the CV model to validate if items and quantities exist in the bin image."*

* * * * *

8\. (Optional) Deployment on Streamlit Cloud
--------------------------------------------

You can deploy this Streamlit app online using **Streamlit Community Cloud**:

1.  Push the code to GitHub:\
    `https://github.com/justanant1807/applied-ai-assignment1`

2.  Make sure `requirements.txt` contains `streamlit`, `groundingdino-py`, `git+https://github.com/openai/CLIP.git`, etc.

3.  Go to **<https://share.streamlit.io>** → "New app".

4.  Choose repo `justanant1807/applied-ai-assignment1`, branch `main`.

5.  Set main file to:

    `app/streamlit_app.py`

6.  Deploy.

> ⚠️ If the full dataset or `groundingdino_swint_ogc.pth` are not committed (due to size limits), the **cloud app may only support a small demo subset**.\
> The full-scale model and evaluation can always be run locally following the steps above.

* * * * *

9\. How to reproduce results
----------------------------

1.  Set up env and install dependencies (Section 3).

2.  Download dataset and weights (Section 4).

3.  Run v5 pipeline on all bins:

    `python models/validate_full_pipeline_v5.py --all`

4.  Evaluate:

    `python mlops/evaluate_model_v3_counts.py --summary out/summary_v5_clip020_box017.csv`

5.  Generate plots for the report:

    `python mlops/plot_results.py --summary out/summary_v5_clip020_box017.csv`

6.  Launch UI:

    `streamlit run app/streamlit_app.py`
