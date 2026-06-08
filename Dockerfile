FROM nvcr.io/nvidia/pytorch:22.08-py3

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# --------------------------------------------------
# System packages
# --------------------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        vim \
        libmkl-dev \
        libxrender1 \
        libxext6 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# --------------------------------------------------
# RTK — Rust Token Killer (CLI output compression for LLM)
# https://github.com/rtk-ai/rtk
# Optional build arg: --build-arg RTK_VERSION=v0.42.3
# --------------------------------------------------
ARG RTK_VERSION=
ENV PATH="/root/.local/bin:${PATH}"
COPY scripts/install_rtk.sh /tmp/install_rtk.sh
RUN chmod +x /tmp/install_rtk.sh && \
    RTK_VERSION="${RTK_VERSION}" bash /tmp/install_rtk.sh && \
    rm -f /tmp/install_rtk.sh && \
    ln -sf /root/.local/bin/rtk /usr/local/bin/rtk

RUN rtk --version && which rtk

# --------------------------------------------------
# Check built-in Python / pip
# --------------------------------------------------
RUN python --version && \
    which python && \
    python -m pip --version

# --------------------------------------------------
# Upgrade packaging tools
# --------------------------------------------------
RUN python -m pip install --upgrade pip setuptools wheel

# --------------------------------------------------
# Reinstall PyTorch stack with fixed version
# --------------------------------------------------
RUN python -m pip uninstall -y torch torchvision torchaudio || true

RUN python -m pip install --no-cache-dir \
    torch==1.13.1+cu117 \
    torchvision==0.14.1+cu117 \
    torchaudio==0.13.1 \
    --extra-index-url https://download.pytorch.org/whl/cu117

# --------------------------------------------------
# Scientific Python packages
# lifelines removed for now
# --------------------------------------------------
RUN python -m pip install --no-cache-dir \
    numpy==1.21.5 \
    scipy==1.7.3 \
    pandas==1.3.5 \
    scikit-learn==1.0.2 \
    hickle==5.0.2 \
    rdkit-pypi==2023.3.1b1 \
    networkx==2.6.3 \
    subword-nmt \
    seaborn

# --------------------------------------------------
# Verify scientific stack
# --------------------------------------------------
RUN python - <<'PY'
import numpy
import scipy
import pandas
import sklearn
import hickle
import rdkit
import networkx
import seaborn

print("numpy:", numpy.__version__)
print("scipy:", scipy.__version__)
print("pandas:", pandas.__version__)
print("sklearn:", sklearn.__version__)
print("hickle:", hickle.__version__)
print("rdkit: OK")
print("networkx:", networkx.__version__)
print("seaborn:", seaborn.__version__)
PY

# --------------------------------------------------
# Reinstall PyG stack
# --------------------------------------------------
RUN python -m pip uninstall -y \
    torch-scatter \
    torch-sparse \
    torch-cluster \
    torch-spline-conv \
    torch-geometric || true

RUN python -m pip install --no-cache-dir \
    torch-scatter==2.1.1+pt113cu117 \
    -f https://data.pyg.org/whl/torch-1.13.1+cu117.html

RUN python -m pip install --no-cache-dir \
    torch-sparse==0.6.17+pt113cu117 \
    -f https://data.pyg.org/whl/torch-1.13.1+cu117.html

RUN python -m pip install --no-cache-dir \
    torch-cluster==1.6.1+pt113cu117 \
    -f https://data.pyg.org/whl/torch-1.13.1+cu117.html

RUN python -m pip install --no-cache-dir \
    torch-spline-conv==1.2.2+pt113cu117 \
    -f https://data.pyg.org/whl/torch-1.13.1+cu117.html

RUN python -m pip install --no-cache-dir \
    torch-geometric==2.3.1

# --------------------------------------------------
# Verify torch / PyG
# --------------------------------------------------
RUN python - <<'PY'
import sys
import torch
print("Python:", sys.version)
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

import torch_geometric
import torch_scatter
print("PyG:", torch_geometric.__version__)
print("torch_scatter import: OK")
PY

# --------------------------------------------------
# Jupyter config
# --------------------------------------------------
RUN jupyter notebook --generate-config -y --no-browser && \
    echo "c.NotebookApp.ip='*'" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.allow_origin='*'" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.open_browser=False" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.port=8888" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.token=''" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.password=''" >> ~/.jupyter/jupyter_notebook_config.py

EXPOSE 8888

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]