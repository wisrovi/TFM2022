import json


class Redis(object):
    import redis
    def __init__(self, host, port, db_n=0):
        self.host = host
        self.port = port
        self.db_n = db_n

        self.__connect()

    def __connect(self):
        self.r = self.redis.Redis(host=self.host, port=self.port, db=self.db_n)

    def crear_autoincremental(self, key):
        return self.r.incr(key)

    def save(self, key, value):
        if isinstance(value, dict):
            value = json.dumps(value)

        self.r.set(key, value)

    def save_pipeline(self, data: dict):
        pipe = self.r.pipeline()
        # recorrer dict para sacar todos los keys y values
        for k, v in data.items():
            pipe.set(k, v)
        pipe.execute()

    def read(self, key):
        try:
            data = self.r.get(key)
        except Exception as e:
            return None

        # convertir a dict si es posible
        try:
            data = json.loads(data)
        except:
            pass

        # bytes to string
        try:
            data = data.decode('utf-8')
        except:
            pass

        # convert string to int if possible
        try:
            data = int(data)
        except:
            pass

        return data

    def read_dict(self, key):
        return self.r.hgetall(key)


def start_conection(server, port):
    redis = Redis(host=server, port=port, db_n=0)
    return redis