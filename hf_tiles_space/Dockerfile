FROM python:3.12-slim

# HF Spaces convention: run as a non-root user with a writable home for caches
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface

WORKDIR /app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user app.py .
COPY --chown=user tile_ranges.json .

USER user
EXPOSE 7860

# pf_tiles.zarr is pulled at startup from the HF Dataset repo named by
# HF_DATASET_REPO (or mounted/copied locally and pointed to by TILES_ZARR).
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
