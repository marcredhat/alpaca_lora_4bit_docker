FROM nvidia/cuda:11.7.0-devel-ubuntu22.04
USER root

# Install Python
# Note that the package python-is-python3 will alias python3 as python
RUN apt-get update && apt-get install -y --no-install-recommends \
   krb5-user python3.10 python3-pip python-is-python3 ssh xz-utils

# Configure pip to install packages under /usr/local
# when building the Runtime image
RUN pip3 config set install.user false

# Install the Jupyter kernel gateway.
# The IPython kernel is automatically installed 
# under the name python3,
# so below we set the kernel name to python3.
RUN pip3 install "jupyter-kernel-gateway==2.5.1"

# Associate uid and gid 8536 with username cdsw
RUN \
  addgroup --gid 8536 cdsw && \
  adduser --disabled-password --gecos "CDSW User" --uid 8536 --gid 8536 cdsw


# Relax permissions to facilitate installation of Cloudera
# client files at startup
RUN for i in /bin /opt /usr /usr/share/java; do \
   mkdir -p ${i}; \
   chown cdsw ${i}; \
   chmod +rw ${i}; \
   for subfolder in `find ${i} -type d` ; do \
      chown cdsw ${subfolder}; \
      chmod +rw ${subfolder}; \
   done \
 done

RUN for i in /etc /etc/alternatives; do \
mkdir -p ${i}; \
chmod 777 ${i}; \
done

# Install any additional packages.
# apt-get install ...
# pip install ...

# Final touches are done by the cdsw user to avoid
# permission issues in CML
USER cdsw

# Set up Python symlink to /usr/local/bin/python3
RUN ln -s $(which python) /usr/local/bin/python3

# configure pip to install packages to /home/cdsw
# once the Runtime image is loaded into CML
RUN /bin/bash -c "echo -e '[install]\nuser = true'" > /etc/pip.conf


# syntax = docker/dockerfile:experimental

# Dockerfile is split into parts because we want to cache building the requirements and downloading the model, both of which can take a long time.

FROM nvidia/cuda:11.7.0-devel-ubuntu22.04 AS builder

RUN apt-get update && apt-get install -y python3 python3-pip git

RUN pip3 install --upgrade pip 

# Some of the requirements expect some python packages in their setup.py, just install them first.
RUN --mount=type=cache,target=/root/.cache/pip pip install --user torch==2.0.0
RUN --mount=type=cache,target=/root/.cache/pip pip install --user semantic-version==2.10.0 requests tqdm

# The docker build environment has trouble detecting CUDA version, build for all reasonable archs
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6"
COPY requirements.txt requirements.txt
RUN --mount=type=cache,target=/root/.cache pip install --user -r requirements.txt

# -------------------------------

# Download the model
FROM nvidia/cuda:11.7.0-devel-ubuntu22.04 AS downloader
RUN apt-get update && apt-get install -y wget

RUN wget --progress=bar:force:noscroll https://huggingface.co/decapoda-research/llama-7b-hf-int4/resolve/main/llama-7b-4bit.pt

RUN mkdir -p /var/cache/apt/archives/partial && apt-get autoclean

# -------------------------------

#FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel


RUN --mount=type=cache,target=/var/cache/apt apt-get update && apt-get install -y git python3 python3-pip

RUN ln -s `which python3` /usr/bin/python


# Copy the installed packages from the first stage
COPY --from=builder /root/.local /root/.local

RUN mkdir alpaca_lora_4bit
WORKDIR alpaca_lora_4bit

COPY --from=downloader llama-7b-4bit.pt llama-7b-4bit.pt

#RUN git clone --depth=1 --branch main https://github.com/andybarry/text-generation-webui-4bit.git text-generation-webui-tmp

RUN git clone --depth=1 --branch main https://github.com/oobabooga/text-generation-webui.git text-generation-webui-tmp

RUN --mount=type=cache,target=/root/.cache pip install --user markdown gradio

# Apply monkey patch
RUN cd text-generation-webui-tmp && printf '%s'"import custom_monkey_patch # apply monkey patch\nimport gc\n\n" | cat - server.py > tmpfile && mv tmpfile server.py

# Get the model config
RUN cd text-generation-webui-tmp && python download-model.py --text-only decapoda-research/llama-7b-hf && mv models/decapoda-research_llama-7b-hf ../llama-7b-4bit


# Get LoRA
RUN cd text-generation-webui-tmp && python download-model.py samwit/alpaca7b-lora && mv loras/samwit_alpaca7b-lora ../alpaca7b_lora

COPY *.py .
COPY text-generation-webui text-generation-webui
COPY monkeypatch .

RUN mv -f text-generation-webui-tmp/* text-generation-webui/

# Symlink for monkeypatch
RUN cd text-generation-webui && ln -s ../autograd_4bit.py ./autograd_4bit.py && ln -s ../matmul_utils_4bit.py .

# Swap to the 7bn parameter model
RUN sed -i 's/llama-13b-4bit/llama-7b-4bit/g' text-generation-webui/custom_monkey_patch.py && sed -i 's/alpaca13b_lora/alpaca7b_lora/g' text-generation-webui/custom_monkey_patch.py

# Run the server
WORKDIR /alpaca_lora_4bit/text-generation-webui
CMD ["python", "-u", "server.py", "--listen", "--chat"]



# Set Runtime label and environment variables metadata
#ML_RUNTIME_EDITOR and ML_RUNTIME_METADATA_VERSION must not be changed.
ENV ML_RUNTIME_EDITOR="PBJ Workbench" \
    ML_RUNTIME_METADATA_VERSION="2" \
    ML_RUNTIME_KERNEL="Python 3.10" \
    ML_RUNTIME_EDITION="Custom Edition" \
    ML_RUNTIME_SHORT_VERSION="1.0" \
    ML_RUNTIME_MAINTENANCE_VERSION="1" \
    ML_RUNTIME_JUPYTER_KERNEL_GATEWAY_CMD="/usr/local/bin/jupyter kernelgateway" \
    ML_RUNTIME_JUPYTER_KERNEL_NAME="python3" \
    ML_RUNTIME_DESCRIPTION="Marc Alpaca Lora 4 bit Custom PBJ Runtime"
          

ENV ML_RUNTIME_FULL_VERSION="$ML_RUNTIME_SHORT_VERSION.$ML_RUNTIME_MAINTENANCE_VERSION" 

LABEL com.cloudera.ml.runtime.editor=$ML_RUNTIME_EDITOR \
	    com.cloudera.ml.runtime.kernel=$ML_RUNTIME_KERNEL \
	    com.cloudera.ml.runtime.edition=$ML_RUNTIME_EDITION \
	    com.cloudera.ml.runtime.full-version=$ML_RUNTIME_FULL_VERSION \
      com.cloudera.ml.runtime.short-version=$ML_RUNTIME_SHORT_VERSION \
      com.cloudera.ml.runtime.maintenance-version=$ML_RUNTIME_MAINTENANCE_VERSION \
      com.cloudera.ml.runtime.description=$ML_RUNTIME_DESCRIPTION \
      com.cloudera.ml.runtime.runtime-metadata-version=$ML_RUNTIME_METADATA_VERSION

