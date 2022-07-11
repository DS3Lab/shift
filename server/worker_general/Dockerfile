FROM tensorflow/tensorflow:2.6.0-gpu

ARG CACHE_LOCATION
ARG RESULTS_LOCATION

# https://docs.docker.com/develop/develop-images/dockerfile_best-practices/#user
# https://github.com/moby/moby/issues/2259#issuecomment-48286811
# https://github.com/hexops/dockerfile
# Create user & setup volumes
RUN groupadd -g 10001 -r worker_general \
    && useradd -u 10000 --no-log-init -r -g worker_general worker_general \
    && mkdir -p $CACHE_LOCATION \
    && mkdir -p $RESULTS_LOCATION \
    && chown -R 10000:10001 $CACHE_LOCATION \
    && chown -R 10000:10001 $RESULTS_LOCATION
VOLUME ["${CACHE_LOCATION}"]
VOLUME ["${RESULTS_LOCATION}"]

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Where the code will reside
WORKDIR /app

# Libraries needed for runtime
COPY worker_general/requirements-external.txt worker_general/requirements-external-docker.txt ./
RUN pip install --no-cache-dir -r requirements-external-docker.txt -f https://download.pytorch.org/whl/torch_stable.html

# Schemas library
COPY common/schemas ./schemas
RUN pip install --no-cache-dir -e schemas

# Task2vec library
COPY common/task2vec ./task2vec
RUN pip install --no-cache-dir -e task2vec


# Pipeline library
COPY common/pipeline ./pipeline
RUN pip install --no-cache-dir -e pipeline

# Code
COPY worker_general ./worker_general

# Give code files and folders correct permissions
RUN chmod -R 644 . && chmod -R +X .

# Switch user for safety
USER worker_general

WORKDIR worker_general