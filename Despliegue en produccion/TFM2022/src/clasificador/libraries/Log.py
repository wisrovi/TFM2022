import imp
import logging
import os
from time import strftime
CLUSTER_NAME = os.environ.get('CLUSTER_NAME', 'cluster1')

logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.DEBUG,
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.FileHandler(f"/log/model_{CLUSTER_NAME}.log"),
                              logging.StreamHandler()])