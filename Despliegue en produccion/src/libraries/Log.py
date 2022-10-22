import logging
from time import strftime
logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.DEBUG,
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.FileHandler("/log/service_model.log"),
                              logging.StreamHandler()])