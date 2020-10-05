import pandas as pd
import json
import numpy as np
from windrose import WindroseAxes
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import sys
def wind_rose(wr):
    wd = wr[:,0]
    ws = wr[:,1]
    ax = WindroseAxes.from_ax()
    ax.bar(wd, ws, normed=True, opening=0.8, edgecolor='white')
    ax.set_legend()
    plt.show()


if __name__ == '__main__':
	df = pd.read_csv(sys.argv[1])
	wind_resource = df[['drct', 'sped']].to_numpy(dtype = np.float32)
	wind_rose(wind_resource)