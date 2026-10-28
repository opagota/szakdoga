import numpy as np  
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def load_npz(file="../run1028/trajectory_solution_168h_500m.npz"):
    """
    Loads .npz file and extracts latitude & longitude data
    """
    sol = np.load(file)
    time = sol['t']
    N = int(sol['y'].shape[0]/2)
    particle_lons = sol['y'][:N, :]  
    particle_lats = sol['y'][N:, :]

    return time, particle_lons, particle_lats


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculates the great circle distance between two points 
    on the earth [metres]
    lat1, lon1, lat2, lon2: degrees, floats or (N,) numpy arrays
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 

    return 6371000 * c

def calc_length(lons, lats):
    """
    Calculates length of 'yarn' at every timestep.
    """
    lon1, lat1 = lons[:-1, :], lats[:-1, :]
    lon2, lat2 = lons[1:, :], lats[1:, :]
    
    segment_distances = haversine(lon1, lat1, lon2, lat2)
    
    lengths = np.sum(segment_distances, axis=0)

    return lengths

time, particle_lons, particle_lats = load_npz()
time_hr = time / 3600
length = calc_length(particle_lons, particle_lats)

def linear(x, a, b):
    return a*x + b

mask = time_hr >= 90
popt, pcov = curve_fit(linear, time_hr[mask], np.log(length[mask]))
perr = np.sqrt(np.diag(pcov))

plt.plot(time_hr, length, "b.", label="Data")
plt.yscale("log")

fit_y = np.exp(linear(time_hr, *popt))
plt.plot(time_hr, fit_y, "r-", label=f"Fit: h = {popt[0]:.4f} 1/hr")
plt.xlabel("time [h]")
plt.ylabel("length [m]")
plt.title("Determination of topological entropy")
plt.grid(True, which="both", ls="--")
plt.legend()
plt.show()