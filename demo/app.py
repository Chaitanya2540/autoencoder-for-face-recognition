"""
FYP Demo — Autoencoder vs VGG Face Recognition
-----------------------------------------------
Runs locally:
    pip install -r requirements.txt
    python app.py

Weight loading:
    Small weights (AE encoder, centroids, metadata) ship with this repo in ./models/.
    Large weights (AE full decoder, VGG classifier) are downloaded on first launch
    from a Hugging Face model repo — see MODEL_REPO below.

    If you prefer to run fully offline, drop the .keras files into ./models/ and
    the loader will use the local copy automatically.
"""

from __future__ import annotations

import os, json, base64
from io import BytesIO
from pathlib import Path

import numpy as np
import gradio as gr
from PIL import Image
import tensorflow as tf

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
GALLERY_DIR = BASE_DIR / "gallery_images"
IMG_SIZE = 128

# Hugging Face model repo holding the large .keras weights.
# Replace with your own repo id if you fork this project.
MODEL_REPO = os.environ.get("MODEL_REPO", "chaitanya2540/lfw-face-recognition-models")


def resolve_weight(filename: str) -> str:
    """Return a local filesystem path to `filename`, downloading from HF Hub if needed."""
    local = MODELS_DIR / filename
    if local.exists():
        return str(local)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise RuntimeError(
            f"'{filename}' not found locally and huggingface-hub is not installed. "
            "Install it with `pip install huggingface-hub` or place the file in ./models/."
        ) from e
    print(f"Downloading {filename} from {MODEL_REPO} …")
    return hf_hub_download(repo_id=MODEL_REPO, filename=filename)


# ── Load models ───────────────────────────────────────────────────────────────
print("Loading models …")
ae_encoder = tf.keras.models.load_model(resolve_weight("ae_v2_encoder.keras"), compile=False)
ae_full    = tf.keras.models.load_model(resolve_weight("ae_v2_full.keras"), compile=False)
vgg_model  = tf.keras.models.load_model(resolve_weight("vgg_classifier_final.keras"), compile=False)
print("Models loaded.")

# ── Load metadata ─────────────────────────────────────────────────────────────
centroids = np.load(MODELS_DIR / "centroids_v2.npy")
with open(MODELS_DIR / "centroid_names_v2.json") as f:
    centroid_names: list[str] = json.load(f)
with open(MODELS_DIR / "class_names.json") as f:
    vgg_class_names: list[str] = json.load(f)

# ── Mutable enrollment state ──────────────────────────────────────────────────
enrolled_centroids: np.ndarray = centroids.copy()
enrolled_names: list[str] = list(centroid_names)
enrolled_gallery: dict[str, str | Image.Image] = {}  # name → path or PIL image

for _name in centroid_names:
    _path = GALLERY_DIR / f"{_name}.jpg"
    if _path.exists():
        enrolled_gallery[_name] = str(_path)

# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _fmt(name: str) -> str:
    """'Colin_Powell' → 'Colin Powell'"""
    return name.replace("_", " ")


def _preprocess(image) -> np.ndarray:
    """Any input → (128, 128, 3) float32 in [0, 1]."""
    if isinstance(image, np.ndarray):
        img = Image.fromarray(image.astype("uint8"))
    elif isinstance(image, str):
        img = Image.open(image)
    else:
        img = image
    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    return np.array(img, dtype=np.float32) / 255.0


def _embed(image) -> np.ndarray:
    """Return AE embedding."""
    arr = _preprocess(image)
    return ae_encoder.predict(arr[np.newaxis], verbose=0)[0]


def _reconstruct(image) -> np.ndarray:
    """Return AE reconstruction as (128,128,3) float array."""
    arr = _preprocess(image)
    return ae_full.predict(arr[np.newaxis], verbose=0)[0]


def _vgg_probs(image) -> np.ndarray:
    """Return softmax vector from the VGG classifier."""
    arr = _preprocess(image)
    return vgg_model.predict(arr[np.newaxis], verbose=0)[0]


def _top_k(embedding: np.ndarray, k: int = 5):
    """Return [(name, distance), …] for k nearest enrolled centroids."""
    dists = np.linalg.norm(enrolled_centroids - embedding, axis=1)
    idxs = np.argsort(dists)[:k]
    return [(enrolled_names[i], float(dists[i])) for i in idxs]


def _img_b64(src) -> str:
    """Convert a file path or PIL Image to a base64 data-URI."""
    if isinstance(src, str):
        with open(src, "rb") as f:
            data = f.read()
    else:
        buf = BytesIO()
        src.save(buf, format="JPEG")
        data = buf.getvalue()
    return "data:image/jpeg;base64," + base64.b64encode(data).decode()


# ══════════════════════════════════════════════════════════════════════════════
#  Tab 1 — How the Autoencoder Sees Faces
# ══════════════════════════════════════════════════════════════════════════════

def tab1_run(image):
    if image is None:
        return None, None
    orig = (_preprocess(image) * 255).astype(np.uint8)
    recon = (np.clip(_reconstruct(image), 0, 1) * 255).astype(np.uint8)
    return orig, recon


# ══════════════════════════════════════════════════════════════════════════════
#  Tab 2 — Face Recognition
# ══════════════════════════════════════════════════════════════════════════════

def _gallery_html() -> str:
    """HTML grid of all currently enrolled people."""
    cards = []
    for name in enrolled_names:
        src = enrolled_gallery.get(name)
        if src:
            tag = (
                f'<img src="{_img_b64(src)}" '
                f'style="width:100%;aspect-ratio:1;object-fit:cover;border-radius:8px;">'
            )
        else:
            tag = (
                '<div style="width:100%;aspect-ratio:1;background:#ddd;'
                'border-radius:8px;display:flex;align-items:center;'
                'justify-content:center;font-size:11px;color:#999;">No photo</div>'
            )
        cards.append(
            f'<div style="text-align:center">{tag}'
            f'<div style="font-size:11px;margin-top:4px;font-weight:500;">'
            f'{_fmt(name)}</div></div>'
        )
    return (
        '<div style="display:grid;grid-template-columns:repeat(5,1fr);'
        f'gap:10px;padding:8px;">{"".join(cards)}</div>'
    )


def tab2_identify(image):
    if image is None:
        return "<p style='color:#888;'>Please upload an image first.</p>"

    # ── Autoencoder side ──────────────────────────────────────────────────
    emb = _embed(image)
    matches = _top_k(emb, k=5)

    # ── VGG side ──────────────────────────────────────────────────────────
    probs = _vgg_probs(image)
    top_idx = int(np.argmax(probs))
    top_conf = float(probs[top_idx])
    top_name = vgg_class_names[top_idx]
    enrolled_set = set(enrolled_names)

    if top_conf < 0.3 or top_name not in enrolled_set:
        vgg_html = (
            '<div style="padding:20px;text-align:center;color:#888;">'
            '<div style="font-size:32px;margin-bottom:8px;">&#x2753;</div>'
            '<div style="font-size:14px;font-weight:500;">Low confidence / Not in enrolled set</div>'
            f'<div style="font-size:12px;margin-top:6px;color:#aaa;">'
            f"VGG's best guess: {_fmt(top_name)} ({top_conf:.1%})</div></div>"
        )
    else:
        src = enrolled_gallery.get(top_name)
        thumb = (
            f'<img src="{_img_b64(src)}" '
            f'style="width:60px;height:60px;border-radius:50%;object-fit:cover;margin-bottom:6px;">'
            if src else ""
        )
        vgg_html = (
            f'<div style="text-align:center;padding:20px;">{thumb}'
            f'<div style="font-size:16px;font-weight:600;">{_fmt(top_name)}</div>'
            f'<div style="font-size:13px;color:#666;margin-top:4px;">'
            f'Confidence: {top_conf:.1%}</div></div>'
        )

    return (
        '<div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;">'
        '<div>'
        '<h3 style="margin:0 0 10px;font-size:16px;color:#333;">Autoencoder — Top 5 Matches</h3>'
        f'{_ae_top5_html(matches)}'
        '</div>'
        '<div style="border-left:1px solid #e0e0e0;padding-left:20px;">'
        '<h3 style="margin:0 0 10px;font-size:16px;color:#333;">VGG Classifier</h3>'
        f'{vgg_html}'
        '</div></div>'
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Tab 3 — Enroll a New Person (No Retraining)
# ══════════════════════════════════════════════════════════════════════════════

def tab3_enroll(name: str, photos: list | None):
    global enrolled_centroids, enrolled_names, enrolled_gallery

    if not name or not name.strip():
        return "&#x26A0;&#xFE0F; Please enter a name."
    if not photos:
        return "&#x26A0;&#xFE0F; Please upload at least one photo."

    name_key = name.strip().replace(" ", "_")
    embeddings = []
    first_img = None

    for photo in photos:
        path = photo if isinstance(photo, str) else getattr(photo, "name", photo)
        img = Image.open(path)
        if first_img is None:
            first_img = img.copy()
        embeddings.append(_embed(img))

    new_centroid = np.mean(embeddings, axis=0)

    if name_key in enrolled_names:
        idx = enrolled_names.index(name_key)
        enrolled_centroids[idx] = new_centroid
    else:
        enrolled_centroids = np.vstack([enrolled_centroids, new_centroid[np.newaxis]])
        enrolled_names.append(name_key)

    enrolled_gallery[name_key] = first_img
    return (
        f"&#x2705; **{_fmt(name_key)}** enrolled with {len(embeddings)} photo(s). "
        "You can now test recognition below."
    )


def _ae_top5_html(matches, border_color="#4CAF50", bg_color="#E8F5E9"):
    """Render top-5 AE matches as HTML rows (reused by Tab 2 and Tab 3)."""
    rows = []
    for rank, (name, dist) in enumerate(matches):
        sim = 1.0 / (1.0 + dist)
        pct = int(sim * 100)
        bar_color = "#4CAF50" if rank == 0 else "#81C784" if rank < 3 else "#C8E6C9"
        bold = "600" if rank == 0 else "400"
        badge = "&#x1F3C6; " if rank == 0 else ""

        src = enrolled_gallery.get(name)
        thumb = (
            f'<img src="{_img_b64(src)}" '
            f'style="width:48px;height:48px;border-radius:50%;object-fit:cover;">'
            if src
            else '<div style="width:48px;height:48px;border-radius:50%;background:#ddd;"></div>'
        )
        rows.append(
            f'<div style="display:flex;align-items:center;gap:10px;margin:6px 0;">'
            f'{thumb}<div style="flex:1;">'
            f'<div style="font-weight:{bold};font-size:14px;">{badge}{_fmt(name)}'
            f'<span style="font-weight:400;font-size:12px;color:#888;margin-left:6px;">{pct}%</span></div>'
            f'<div style="background:#e8e8e8;border-radius:4px;height:16px;margin-top:3px;overflow:hidden;">'
            f'<div style="background:{bar_color};height:100%;width:{pct}%;border-radius:4px;"></div>'
            f'</div></div></div>'
        )
    return "".join(rows)


def tab3_test(image):
    if image is None:
        return "<p style='color:#888;'>Please upload a test image.</p>"

    # ── AE result — top 5 ─────────────────────────────────────────────────
    emb = _embed(image)
    matches = _top_k(emb, k=5)

    ae_html = (
        '<div style="background:#E8F5E9;border:2px solid #4CAF50;border-radius:12px;'
        'padding:20px;">'
        '<h3 style="margin:0 0 12px;color:#2E7D32;">Autoencoder — Top 5 Matches</h3>'
        f'{_ae_top5_html(matches)}'
        '<div style="font-size:12px;color:#558B2F;margin-top:12px;">'
        'The autoencoder can recognise newly enrolled people '
        '<b>without any retraining</b> — it just compares face patterns.</div>'
        '</div>'
    )

    # ── VGG result ────────────────────────────────────────────────────────
    probs = _vgg_probs(image)
    top_idx = int(np.argmax(probs))
    top_conf = float(probs[top_idx])
    top_vgg_name = vgg_class_names[top_idx]
    top_ae_name = matches[0][0]
    is_new = top_ae_name not in set(vgg_class_names)

    if is_new or top_conf < 0.3:
        vgg_html = (
            '<div style="background:#FFEBEE;border:2px solid #E53935;border-radius:12px;'
            'padding:20px;text-align:center;">'
            '<h3 style="margin:0 0 12px;color:#C62828;">VGG Classifier Result</h3>'
            '<div style="font-size:40px;margin:10px 0;">&#x1F6AB;</div>'
            '<div style="font-size:16px;font-weight:600;color:#B71C1C;">Unknown Person</div>'
            '<div style="font-size:13px;color:#C62828;margin-top:8px;">'
            'This person was <b>not in the VGG\'s training data</b>.<br>'
            'Retraining the entire model would be required.</div>'
            f'<div style="font-size:11px;color:#999;margin-top:10px;">'
            f"VGG's best guess: {_fmt(top_vgg_name)} ({top_conf:.1%})</div></div>"
        )
    else:
        vgg_html = (
            '<div style="background:#FFF3E0;border:2px solid #FF9800;border-radius:12px;'
            'padding:20px;text-align:center;">'
            '<h3 style="margin:0 0 12px;color:#E65100;">VGG Classifier Result</h3>'
            f'<div style="font-size:16px;font-weight:500;color:#E65100;">'
            f'{_fmt(top_vgg_name)} ({top_conf:.1%})</div>'
            f'<div style="font-size:12px;color:#BF360C;margin-top:8px;">'
            f'VGG can only recognise the {len(vgg_class_names)} people '
            'it was originally trained on.</div></div>'
        )

    return (
        '<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;">'
        f'{ae_html}{vgg_html}</div>'
    )


# ══════════════════════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════════════════════

example_paths = sorted(
    str(GALLERY_DIR / f)
    for f in os.listdir(GALLERY_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
)

with gr.Blocks(
    title="Face Recognition: Autoencoder vs VGG",
    theme=gr.themes.Soft(),
) as demo:

    gr.Markdown(
        "# Face Recognition Demo\n"
        "### Comparing two approaches: "
        "Autoencoder (flexible) vs VGG Classifier (fixed)"
    )

    # ── Tab 1 ─────────────────────────────────────────────────────────────
    with gr.Tab("How the Autoencoder Sees Faces"):
        gr.Markdown(
            "Upload a face photo to see how the autoencoder "
            "compresses and reconstructs it."
        )
        with gr.Row():
            t1_input = gr.Image(type="numpy", label="Upload a face photo", sources=["upload", "webcam"])
        with gr.Row():
            t1_btn = gr.Button("Show Reconstruction", variant="primary")
            t1_clear = gr.ClearButton(
                [t1_input], value="Clear", variant="secondary"
            )
        with gr.Row():
            t1_orig = gr.Image(label="Original (resized to 128 × 128)")
            t1_recon = gr.Image(label="Autoencoder Reconstruction")
        gr.Markdown(
            "> The autoencoder learned to compress and rebuild faces "
            "without being told who anyone is. The compressed representation "
            "is what we use for recognition."
        )
        if example_paths:
            gr.Examples(
                examples=example_paths[:6],
                inputs=t1_input,
                label="Try one of these",
            )
        t1_btn.click(tab1_run, inputs=t1_input, outputs=[t1_orig, t1_recon])
        t1_clear.add([t1_orig, t1_recon])

    # ── Tab 2 ─────────────────────────────────────────────────────────────
    with gr.Tab("Face Recognition"):
        gr.Markdown(
            "Upload a face and see how each model identifies it."
        )
        t2_gallery = gr.HTML(value=_gallery_html(), label="Enrolled People")
        with gr.Row():
            t2_input = gr.Image(type="numpy", label="Upload a test face", sources=["upload", "webcam"])
        with gr.Row():
            t2_btn = gr.Button("Identify", variant="primary")
            t2_clear = gr.ClearButton(
                [t2_input], value="Clear", variant="secondary"
            )
        t2_output = gr.HTML()
        gr.Markdown(
            "> **Left column** — the autoencoder's top 5 guesses, "
            "ranked by how similar the face pattern is.  \n"
            "> **Right column** — the VGG classifier's single best guess "
            "with its confidence level."
        )
        if example_paths:
            gr.Examples(
                examples=example_paths[:6],
                inputs=t2_input,
                label="Try one of these",
            )
        t2_btn.click(tab2_identify, inputs=t2_input, outputs=t2_output)
        t2_clear.add([t2_output])

    # ── Tab 3 ─────────────────────────────────────────────────────────────
    with gr.Tab("Enroll a New Person (No Retraining)"):
        gr.Markdown(
            "**This is the key difference.** Add a completely new person "
            "and see which model can recognise them — without retraining anything."
        )

        gr.Markdown("### Step 1 — Enroll")
        with gr.Row():
            t3_name = gr.Textbox(
                label="Person's name", placeholder="e.g. Jane Doe"
            )
            t3_photos = gr.File(
                file_count="multiple",
                label="Upload 3–5 face photos",
                file_types=["image"],
            )
        with gr.Row():
            t3_enroll_btn = gr.Button("Enroll This Person", variant="primary")
            t3_enroll_clear = gr.ClearButton(
                [t3_name, t3_photos], value="Clear", variant="secondary"
            )
        t3_status = gr.Markdown()

        gr.Markdown("### Step 2 — Test Recognition")
        with gr.Row():
            t3_test_img = gr.Image(type="numpy", label="Upload a new photo of this person", sources=["upload", "webcam"])
        with gr.Row():
            t3_test_btn = gr.Button("Test", variant="primary")
            t3_test_clear = gr.ClearButton(
                [t3_test_img], value="Clear", variant="secondary"
            )
        t3_result = gr.HTML()

        gr.Markdown(
            "> **Why does this matter?** The autoencoder learns a general "
            "understanding of faces, so it can recognise someone new from "
            "just a few examples. The VGG classifier was trained on a fixed "
            "list of people — adding someone new means retraining the entire "
            "model from scratch."
        )

        t3_enroll_btn.click(
            tab3_enroll, inputs=[t3_name, t3_photos], outputs=t3_status
        )
        t3_test_btn.click(tab3_test, inputs=t3_test_img, outputs=t3_result)
        t3_enroll_clear.add([t3_status])
        t3_test_clear.add([t3_result])

# ── Launch ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch()
