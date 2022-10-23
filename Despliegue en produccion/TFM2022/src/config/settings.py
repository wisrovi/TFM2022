import os

# initialize Redis connection settings
REDIS_HOST = "redis"
REDIS_PORT = 6379
REDIS_DB = 0

# initialize constants used for server queuing
#leer valor de la variable de entorno CLUSTER_NAME
NAME_QUEUE = os.environ.get("CLUSTER_NAME", "audio_queue")
BATCH_SIZE = 32
SERVER_SLEEP = 0.25
CLIENT_SLEEP = 0.25
