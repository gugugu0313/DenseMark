# DenseMark — inference demo

`inference_demo.ipynb` loads the **DenseMark** model, runs a **Fig.1** two-image composite (encode → mask blend → decode → plots), and **Part 3** robustness metrics on COCO val (mean bit accuracy by class: `geo` / `value` / `iden`).

## Environment

From the repository root:

```bash
pip install -r requirements.txt
```

Python 3.10+ recommended.

## Checkpoints

Download **`bce_bg_crl_llr.pth`** from Google Drive:

**[DenseMark weights](https://drive.google.com/drive/folders/1_jTOcX-QBPkA5KPkCFeic7aqxQcC6u7c?usp=sharing)**

Place the file at:

```text
checkpoints/bce_bg_crl_llr.pth
```

(relative to the repo root, same folder as `densemark_tools.py` expects).

## COCO paths (Part 3)

Set paths in **Part 1** of `inference_demo.ipynb` inside `args = SimpleNamespace(...)`.

Expected layout:

```text
<data_dir>/
  val2017/
  annotations/
    instances_val2017.json
```

| Field | Example | Role |
|-------|---------|------|
| `data_dir` | `"/path/to/COCO2017"` | Root containing `val2017` and `annotations` |
| `val_subdir` | `"val2017"` | Image folder under `data_dir` |
| `ann_file` | `"instances_val2017.json"` | Under `data_dir/annotations/` |

## Running the notebook

1. Open `inference_demo.ipynb` in Jupyter / VS Code.
2. **Part 1 — Setup:** edit `args` (paths above), then run: imports, seeds, `DenseMark` construction.
3. **Part 2 — Fig.1:** composite + decode + plots (requires Fig.1 files under `fig1_dir`).
4. **Part 3 — Robustness metrics:** runs the augmentation loop on COCO val (requires correct `data_dir` / `val_subdir` / `ann_file`).

Metrics are printed with three decimal places; re-run Part 1 if you change seed or paths.

---

**Upstream.** This implementation builds on the [**Watermark Anything**](https://arxiv.org/abs/2411.07231) codebase.
