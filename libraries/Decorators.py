import threading
from functools import wraps
from time import time


def count_elapsed_time(f):
    @wraps(f)
    def cronometro(*args, **kwargs):
        t_inicial = time()  # tomo la hora antes de ejecutar la funcion
        salida = f(*args, **kwargs)
        t_final = time()  # tomo la hora despues de ejecutar la funcion
        print('Tiempo transcurrido (en segundos): {}'.format(t_final - t_inicial))
        return salida

    return cronometro


def execute_in_thread_timer(seconds):
    def _execute_in_thread_timer(f):
        def wrapper(*args, **kwargs):
            thread_f = threading.Timer(seconds, f)
            thread_f.start()

            return thread_f

        return wrapper

    return _execute_in_thread_timer