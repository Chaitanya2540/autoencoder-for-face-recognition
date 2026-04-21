# Autoencoder for Face Recognition

> **Final Year Project — B.Eng. Electrical & Electronic Engineering, Nanyang Technological University (2026)**
> Supervisor: Assoc. Prof. Anamitra Makur · Project No. A3010-251

Investigating whether **deep autoencoders** can serve as lightweight, parameter-efficient alternatives to conventional CNN classifiers for face recognition — and, crucially, whether they can enrol *new identities without retraining*.

**[▶ Try the live demo](https://huggingface.co/spaces/chaitanyagupta/autoencoder-for-face-recognition)** · **[📄 Read the full report (PDF)](report/ChaitanyaGupta_Final_Report_FYP.pdf)**

---

## Highlights

| | VGG Classifier (baseline) | AE + SoftMax (joint) | AE + Centroid (v2) |
|---|---|---|---|
| Top-1 accuracy | 68.71% (158 IDs) | **74.84%** (158 IDs) | 45.00% (20 IDs) |
| Top-5 accuracy | 86.11% | 92.99% | 78.33% |
| Parameters | 15.17 M | ~15 M | **2.97 M** |
| Batch inference latency | 12.89 ms/image | 0.82 ms/image | **1.71 ms/image** |
| Enrol a new person? | ❌ Full retraining | ❌ Full retraining | ✅ **~1 second, zero-shot** |

Three headline findings:

1. **Joint reconstruction + classification training beats pure classification** — the reconstruction objective regularises the encoder, pushing top-1 accuracy from 63.6% (pure CE) to 74.8% (joint). A four-point loss-weight ablation confirms this isn't an artefact of a single hyperparameter choice.
2. **Centroid matching over a pure-reconstruction encoder enables zero-shot enrolment.** Adding a 21st identity to the gallery at inference time takes under a second and does not degrade accuracy on the original 20.
3. **Centroid embeddings favour Euclidean over cosine distance** — the reconstruction objective encodes useful information in embedding *magnitude* (brightness, contrast, facial structure complexity). Normalising it away hurts accuracy by 5 pp.

## What's in this repo

```
autoencoder-for-face-recognition/
├── demo/                          ← Gradio web app
│   ├── app.py                     ← Three-tab UI: reconstruct, identify, enrol
│   ├── requirements.txt
│   ├── gallery_images/            ← 20 LFW faces used by the demo
│   └── models/                    ← Small weights only (encoder, centroids)
├── notebooks/
│   ├── 01_vgg_classifier.ipynb           ← VGG baseline on 158 identities
│   ├── 02_autoencoder_centroid.ipynb     ← AE + centroid recognition (the demo model)
│   ├── 03_loss_weight_ablation.ipynb     ← Joint-loss weight sweep
│   └── archive/                          ← Earlier exploratory iterations
├── report/
│   └── ChaitanyaGupta_Final_Report_FYP.pdf   ← Full thesis with all figures & tables
└── README.md
```

Large model weights (VGG 174 MB, AE full 35 MB) are hosted on Hugging Face and downloaded on first launch — see [Running locally](#running-locally).

## Method in one paragraph

The project evaluates two complementary paradigms on the **Labeled Faces in the Wild (LFW)** dataset:

- **Paradigm A — AE + SoftMax (158 identities).** A convolutional autoencoder is trained jointly with a softmax classification head, using a weighted combination of SSIM+MSE reconstruction loss and cross-entropy. A three-phase training pipeline (encoder warm-up → joint training → fine-tuning) and a loss-weight ablation study validate the joint objective.
- **Paradigm B — AE + Centroid matching (20 identities, open-set).** A pure reconstruction autoencoder is trained without identity labels; each enrolled person is represented by the mean embedding of their gallery images. Recognition is nearest-centroid in embedding space. Three progressively deeper encoders (1.24 M → 2.97 M → 10.89 M params) are compared, and both Euclidean and cosine metrics are benchmarked.

A VGG classifier trained end-to-end on the same 158 identities provides the baseline. Full architectural, training, and evaluation detail is in the [PDF report](report/ChaitanyaGupta_Final_Report_FYP.pdf).

## Running locally

```bash
git clone https://github.com/<your-username>/autoencoder-for-face-recognition.git
cd autoencoder-for-face-recognition/demo

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python app.py      # opens at http://127.0.0.1:7860
```

On first launch the app downloads `ae_v2_full.keras` and `vgg_classifier_final.keras` (~210 MB combined) from the Hugging Face model repo cached under `~/.cache/huggingface/`. To run fully offline, drop the `.keras` files into `demo/models/` and the loader will prefer the local copy.

To reproduce training, open the notebooks under [`notebooks/`](notebooks/) in Kaggle or Google Colab — they fetch LFW via `sklearn.datasets.fetch_lfw_people` and expect a GPU. Key hyperparameters are documented at the top of each notebook and in Appendix A of the [report](report/ChaitanyaGupta_Final_Report_FYP.pdf).

## Stack

`Python 3.10` · `TensorFlow / Keras 2.16+` · `NumPy` · `scikit-learn` (for LFW fetching & baseline classifiers) · `Gradio 4.x` (demo) · `Hugging Face Hub` (weight hosting) · `Kaggle` (training, NVIDIA T4 / P100)

## Citation

If you reference this work, please cite the accompanying report:

> Chaitanya Gupta. *Autoencoder for Face Recognition*. Final Year Project Report, Nanyang Technological University, 2026.

## Licence

The code in this repository is released under the MIT License. The LFW dataset is used under its own terms — see [vis-www.cs.umass.edu/lfw](http://vis-www.cs.umass.edu/lfw/).

## Acknowledgements

Thanks to Assoc. Prof. Anamitra Makur for supervision, and to NTU EEE for compute access.
