import os
import cv2
import tensorflow as tf
from typing import List
import numpy as np
import imageio
from matplotlib import pyplot as plt

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

