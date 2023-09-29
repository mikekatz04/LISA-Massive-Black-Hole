from pyHetWrap import pyHetLikeWrap
import numpy as np
from scipy import constants

Tobs = 1.0 * constants.Julian_year
dt = 10.0

premove = np.genfromtxt("search_sources.dat")[:, 2:]
NS = len(premove)

m1, m2 = premove[:, :2].T.copy()
premove[:, 0] = (m1 * m2) ** (3 / 5) / (m1 + m2) ** (1 / 5)
premove[:, 1] = m1 + m2

premove[:, 0] = np.log(premove[:, 0])
premove[:, 1] = np.log(premove[:, 1])
premove[:, 6] = np.log(premove[:, 6])

# params = np.array(
#     [np.log(0.2e6), np.log(1e6), 0.5, 0.5, 2.0, 1e7, np.log(10.0), 0.6, 2.5, 0.7, 0.2]
# )

# initialize
is_noise_free = False
seg = 0
rep = 0
params = premove[rep].copy()
premove_in = premove.flatten().copy()

check = pyHetLikeWrap(params, Tobs, dt, is_noise_free, seg, rep, premove_in, NS)

# params2 = params.copy()
# params2[0] *= 1.000001

# # get it
# check2 = check.get_like(params2)

# check.udpate_heterodyne(params2)
breakpoint()
