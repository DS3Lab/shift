FROM redis:6.0.10
COPY redis.conf /usr/local/etc/redis/redis.conf
# Switch user for safety
USER redis
CMD [ "redis-server", "/usr/local/etc/redis/redis.conf" ]