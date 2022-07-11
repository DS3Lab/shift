FROM postgres:13.2-alpine

# https://hub.docker.com/_/postgres
# Initialization scripts run only if no database exists yet
COPY init.sql /docker-entrypoint-initdb.d

# Switch user for safety
USER postgres