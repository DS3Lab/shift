FROM tensorflow/tensorflow:2.9.1-gpu

ARG USERSPACE_LOCATION

# https://docs.docker.com/develop/develop-images/dockerfile_best-practices/#user
# https://github.com/moby/moby/issues/2259#issuecomment-48286811
# https://github.com/hexops/dockerfile
# Create user & setup volumes
RUN groupadd -g 10001 -r user_space \
    && useradd -u 10000 --no-log-init -r -g user_space user_space \
    && mkdir -p $USERSPACE_LOCATION \
    && chown -R 10000:10001 $USERSPACE_LOCATION
VOLUME ["${USERSPACE_LOCATION}"]

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Where the code will reside
WORKDIR /app

# Libraries needed for runtime
COPY userspace/requirements-us.txt userspace/requirements-us.txt ./
RUN pip install --no-cache-dir -r requirements-us.txt --extra-index-url https://download.pytorch.org/whl/cu116

# copy the entire userspace directory to the docker container
COPY userspace ./userspace

# Give code files and folders correct permissions
RUN chmod -R 644 . && chmod -R +X .

# Switch user for safety
USER user_space

WORKDIR userspace