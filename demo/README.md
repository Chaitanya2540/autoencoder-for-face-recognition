# Demo — Autoencoder vs VGG Face Recognition

A three-tab Gradio app that lets you:

1. **See** how the autoencoder compresses and reconstructs any face you upload.
2. **Identify** a face and compare the autoencoder's top-5 nearest centroids against the VGG classifier's single best guess.
3. **Enrol a brand-new person** (upload 3–5 photos, name them) and confirm that the autoencoder recognises them immediately — while the VGG classifier cannot.

## Run it

```bash
pip install -r requirements.txt
python app.py
```

Opens at `http://127.0.0.1:7860`.

## Weights

Small weights ship in [`models/`](models/):

- `ae_v2_encoder.keras` — 2.5 MB encoder used for embeddings
- `centroids_v2.npy`, `centroid_names_v2.json` — 20 gallery centroids
- `class_names.json` — VGG label map

Large weights are downloaded on first launch from Hugging Face (`chaitanyagupta/lfw-face-recognition-models`):

- `ae_v2_full.keras` — 35 MB encoder+decoder (for the reconstruction tab)
- `vgg_classifier_final.keras` — 174 MB VGG baseline (for the comparison tab)

Override the repo with the `MODEL_REPO` environment variable if you fork. To run fully offline, copy both files into `models/` — the loader checks locally first.

## Deploying to Hugging Face Spaces

The contents of this `demo/` folder are what you push to a Gradio Space — no edits needed. See the root `README.md` for step-by-step instructions.
