from pyHetWrap import pyHetLikeWrap
import numpy as np
from scipy import constants

Tobs = 1.0 * constants.Julian_year
dt = 10.0

params = np.array(
    [np.log(0.2e6), np.log(1e6), 0.5, 0.5, 2.0, 1e7, np.log(10.0), 0.6, 2.5, 0.7, 0.2]
)

# initialize
check = pyHetLikeWrap(params, Tobs, dt)

params2 = params.copy()
params2[0] *= 1.000001

# get it
check2 = check.get_like(params2)

check.udpate_heterodyne(params2)
breakpoint()
