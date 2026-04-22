# 🚗 Drone-X — Car Detection Demo

A Streamlit app that runs a custom-trained **YOLOv8** model (`best.pt`) to detect and localise moving cars in uploaded images.

## Features
- 📤 Image upload (JPG / PNG / WebP / BMP)
- 🎯 Bounding boxes + confidence labels drawn on detections
- 📊 Average, count & peak confidence metrics
- ⚙️ Sidebar controls for **Confidence Threshold** and **IoU (NMS) Threshold**
- ⬇️ Download annotated image

## Local usage

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud deployment
1. Push this repo (including `best.pt`) to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Select your repo, branch (`main`), and set **Main file path** to `app.py`.
4. Click **Deploy** — no extra secrets needed.

> **Note:** `best.pt` is ~167 MB. Make sure Git LFS is enabled on the repo or the file is committed normally (GitHub allows files up to 2 GB).