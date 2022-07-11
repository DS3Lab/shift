FROM python:3.6-slim

# Add user
# https://docs.docker.com/develop/develop-images/dockerfile_best-practices/#user
RUN groupadd -r newgroup && useradd --no-log-init -r -g newgroup newuser

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# g++ needed in order to install pottery via pip, wget for dockerize
# See: https://www.docker.com/blog/intro-guide-to-dockerfile-best-practices/
# for details about this command
RUN apt-get update \
    && apt-get -y install --no-install-recommends g++ wget \
    && rm -rf /var/lib/apt/lists/*

# Install dockerize
ENV DOCKERIZE_VERSION v0.6.1
RUN wget https://github.com/jwilder/dockerize/releases/download/$DOCKERIZE_VERSION/dockerize-linux-amd64-$DOCKERIZE_VERSION.tar.gz \
    && tar -C /usr/local/bin -xzvf dockerize-linux-amd64-$DOCKERIZE_VERSION.tar.gz \
    && rm dockerize-linux-amd64-$DOCKERIZE_VERSION.tar.gz

# Where the code will reside
WORKDIR /app

# External libraries
COPY scheduler/requirements-external.txt .
RUN pip install --no-cache-dir -r requirements-external.txt

# Schemas library
COPY common/schemas ./schemas
RUN pip install --no-cache-dir -e schemas

# Redis library
COPY common/db_tools ./db_tools
RUN pip install --no-cache-dir -e db_tools

# Copy code after installing libraries - makes rebuilds faster
COPY scheduler ./scheduler

# Give code files and folders correct permissions
RUN chmod -R 644 . && chmod -R +X .

# Switch user for safety
USER newuser

WORKDIR scheduler