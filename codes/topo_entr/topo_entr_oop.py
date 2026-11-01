import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt

class topo_entr():
    def __init__(self, isobar, model):
        self.isobar = isobar
        self.model = model

    def load_npz(self):
        sol = np.load(f"../data/trajectory_solution_168h_500m_{self.isobar}hPa_{self.model}.npz")
        time = sol['t']
        N = int(sol['y'].shape[0]/2)
        particle_lons = sol['y'][:N, :]  
        particle_lats = sol['y'][N:, :]
    
        return time, particle_lons, particle_lats

    def haversine(self, lon1, lat1, lon2, lat2):
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1 
        dlat = lat2 - lat1 
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a)) 

        return 6371000 * c

    def calc_length(self, lons, lats):
        lon1, lat1 = lons[:-1, :], lats[:-1, :]
        lon2, lat2 = lons[1:, :], lats[1:, :]
        
        segment_distances = self.haversine(lon1, lat1, lon2, lat2)
        
        lengths = np.sum(segment_distances, axis=0)

        return lengths

    def fit_h(self, ):
        def linear(x, a, b):
            return a*x + b
        
        time, particle_lons, particle_lats = self.load_npz()
        time_hr = time / 3600
        length = self.calc_length(particle_lons, particle_lats)

        mask = (time_hr >= 90) & (time_hr <= 150)
        popt, pcov = curve_fit(linear, time_hr[mask], np.log(length[mask]))
        perr = np.sqrt(np.diag(pcov))

        return popt[0]

#####################################################################################
configs_AIFS = {
    "AIFS500": topo_entr(500, "AIFS").fit_h(),
    "AIFS700": topo_entr(700, "AIFS").fit_h(),
    "AIFS850": topo_entr(850, "AIFS").fit_h()
}

configs_IFS = {
    "IFS500": topo_entr(500, "IFS").fit_h(),
    "IFS700": topo_entr(700, "IFS").fit_h(),
    "IFS850": topo_entr(850, "IFS").fit_h()
}

isobars = [500, 700, 850]
x = np.arange(len(isobars))
width = 0.35

bars_aifs = plt.bar(x - width/2, list(configs_AIFS.values()), width=width, color="blue", label="AIFS")
bars_ifs = plt.bar(x + width/2, list(configs_IFS.values()), width=width, color="red", label="IFS")

plt.bar_label(bars_aifs, fmt='%.4f', padding=3, fontsize=9)
plt.bar_label(bars_ifs, fmt='%.4f', padding=3, fontsize=9)

plt.xlabel("Nyomási szint [hPa]", fontsize=15)
plt.ylabel("Topologikus entrópia [1/óra]", fontsize=15)
plt.xticks(x, isobars)
plt.grid(True, which="both", ls="--")
plt.legend(fontsize=10, loc="upper left")
plt.margins(y=0.15)
plt.savefig(f"./barplot_h.png")
plt.show()

