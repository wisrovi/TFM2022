import redis
from libraries.Util import Util
from config import settings

util = Util()

db = redis.StrictRedis(host=settings.REDIS_HOST,
                       port=settings.REDIS_PORT,
                       db=settings.REDIS_DB)

# ver estado de conexion con redis
print("redis conecttion", db.ping())