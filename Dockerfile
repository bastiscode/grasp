FROM python:3.12-slim

WORKDIR /grasp
ENV PYTHONUNBUFFERED=1 \
  GRASP_INDEX_DIR=/opt/grasp

# Copy files
COPY . .

# Install GRASP with all dependencies pinned to known-good versions
RUN pip install --no-cache-dir -c constraints.txt .

# Run GRASP by default; override flags via `docker run grasp -- <args>`
ENTRYPOINT ["grasp"]
CMD ["--help"]
