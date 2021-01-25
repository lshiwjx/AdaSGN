import os

os.environ['DISPLAY'] = 'localhost:10.0'
import matplotlib

matplotlib.use('Qt5Agg')

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D