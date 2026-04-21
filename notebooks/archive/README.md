# Archive — Exploratory Notebooks

Earlier iterations leading up to the final models. Included for transparency — the **numbered notebooks one level up** (`01_…`, `02_…`, `03_…`) are the ones referenced in the final report.

| Notebook | What it explored |
|---|---|
| `vgg-bases-face-recognition.ipynb` | Very first VGG attempt, pre-tuning. |
| `cnn-classifier-based-face-recognition.ipynb` | Plain-CNN baseline before moving to VGG depth. |
| `script-6-reconstruction-recognition.ipynb` | First script hooking reconstruction output into recognition — motivated the AE track. |
| `ae-improv1.ipynb` – `ae-improv3.ipynb` | Progressive AE architecture iterations (depth, bottleneck size, activation choice). |
| `ae-improv4.ipynb` | Centre-loss experiment — the **negative result** in §4.4 of the report (accuracy collapsed to 33.4%). |
| `ae-improv-20x10-identities.ipynb` | First 20-identity centroid variant (v1 baseline). |
| `ae-improv-20x10-identities-v3.ipynb` | 12-layer v3 variant; marginally more accurate than v2 but ~4× the parameters, so v2 was chosen for the demo. |

Outputs are preserved where useful. Some cells reference Kaggle-specific paths and won't re-run outside Kaggle without edits.
