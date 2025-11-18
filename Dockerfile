FROM ubuntu:noble

WORKDIR /grasp
ENV DEBIAN_FRONTEND=noninteractive \
  PYTHONUNBUFFERED=1 \
  GRASP_INDEX_DIR=/opt/grasp

RUN apt-get update && apt-get install -y \
  wget \
  bzip2 \
  && rm -rf /var/lib/apt/lists/*

# Install miniconda
ENV CONDA_DIR=/opt/conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
  bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
  rm /tmp/miniconda.sh && \
  $CONDA_DIR/bin/conda clean -afy

ENV PATH="$CONDA_DIR/bin:$PATH"
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Install dependencies
RUN conda install -n base -y -c pytorch -c nvidia -c conda-forge \
  python=3.12 faiss-gpu=1.11.0

# Copy files
COPY . .

# Install GRASP
RUN pip install --no-cache-dir .

# Run GRASP by default; override flags via `docker run grasp -- <args>`
ENTRYPOINT ["grasp"]
CMD ["--help"]
