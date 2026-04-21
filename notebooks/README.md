# Notebooks

The three notebooks that produced the final results in the report.

| # | Notebook | What it does | Reference in report |
|---|---|---|---|
| 01 | [`01_vgg_classifier.ipynb`](01_vgg_classifier.ipynb) | VGG-style classifier on 158 LFW identities — the baseline. | §3.4, §4.1 |
| 02 | [`02_autoencoder_centroid.ipynb`](02_autoencoder_centroid.ipynb) | Pure-reconstruction autoencoder v2 (SSIM+MSE loss) + centroid matching on 20 identities. **This is the model that powers the demo.** | §3.6.2, §4.5.2 |
| 03 | [`03_loss_weight_ablation.ipynb`](03_loss_weight_ablation.ipynb) | Four-way loss-weight ablation sweep that validates the joint reconstruction-classification training strategy. | §4.3 |

All three were authored and trained on Kaggle (NVIDIA T4 / P100). They fetch LFW via `sklearn.datasets.fetch_lfw_people`, so you can re-run them on any GPU-equipped environment without manually downloading the dataset.

Output cells are preserved so you can read the results (training curves, confusion matrices, t-SNE plots) directly on GitHub without running anything.

## Archive

[`archive/`](archive/) contains the ~9 earlier exploratory notebooks that led up to the final models — centre-loss experiments, shallower/deeper variants, and the initial VGG and CNN iterations. They are included for transparency on the development process; the three numbered notebooks above supersede them.
