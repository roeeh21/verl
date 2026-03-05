FROM nvidia/cuda:12.6.1-devel-ubuntu22.04 AS deps

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
# Install system dependencies, Python 3.11, and Google Cloud CLI
RUN apt-get update && \
    apt-get install -y software-properties-common git openssh-client curl wget build-essential && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.11 python3.11-dev python3.11-distutils python3.11-venv && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    apt-get update && apt-get install -y google-cloud-cli && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --set python3 /usr/bin/python3.11 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv and authentication tools
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/uv \
    pip install uv && \
    uv pip install --system keyrings.google-artifactregistry-auth

# Copy dependency files
COPY pyproject.toml setup.py requirements.txt requirements_ai21.txt ./

# Download our custom vllm wheel
RUN mkdir wheels && \
    gcloud storage cp gs://ai21-publishing-studio-experiments/research-serving/vllm_cu122/vllm-0.10.3.dev1+gbecb2083f.cu122-cp311-cp311-linux_x86_64.whl wheels/

# Install python requirements
RUN --mount=type=secret,id=key,dst=/root/.ssh/id_rsa \
    --mount=type=cache,target=/root/.cache/uv \
    gcloud config set project publishing-337912 && \
    ssh-keyscan bitbucket.org >> /root/.ssh/known_hosts && \
    PYTHONNOUSERSITE=1 uv pip install --system --no-sources --keyring-provider subprocess \
      --index-strategy unsafe-best-match \
      --extra-index-url https://pypi.ngc.nvidia.com \
      --extra-index-url https://oauth2accesstoken@us-python.pkg.dev/publishing-337912/infra-python/simple/ \
      --extra-index-url https://oauth2accesstoken@us-python.pkg.dev/publishing-337912/agents-python/simple/ \
      --extra-index-url https://oauth2accesstoken@us-python.pkg.dev/publishing-337912/studio-python/simple/ \
      --extra-index-url https://oauth2accesstoken@us-python.pkg.dev/publishing-337912/lm2-python/simple/ \
      --extra-index-url https://oauth2accesstoken@us-python.pkg.dev/publishing-337912/wordtune-python/simple/ \
      -r requirements.txt -r requirements_ai21.txt

RUN rm -vf wheels/vllm-0.10.3.dev1+gbecb2083f.cu122-cp311-cp311-linux_x86_64.whl

ENV LC_ALL=C.UTF-8
ENV PYTHONUNBUFFERED=1
ENV PYTHONNOUSERSITE=1
# Ensure PyTorch libs are discoverable by dynamic loader
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.11/dist-packages/torch/lib:/usr/local/lib/python3.11/site-packages/torch/lib:$LD_LIBRARY_PATH

ENTRYPOINT ["/bin/bash"]
