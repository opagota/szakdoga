import numpy as np
from scipy.optimize import curve_fit
import matplotlib
import matplotlib.pyplot as plt
import os

class YarnEntropyPlotter:
    def __init__(self, isobar, model, lead_time=168):
        self.isobar = isobar
        self.model = model
        self.lead_time = lead_time

    def load_npz(self):
        filepath = f"../traj_sols/trajectory_solution_{self.lead_time}h_{self.isobar}hPa_{self.model}.npz"
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
            
        sol = np.load(filepath, allow_pickle=True)
        
        time = sol['time']
        
        particle_lons_list = list(sol['lons'])
        particle_lats_list = list(sol['lats'])
    
        return time, particle_lons_list, particle_lats_list

    def haversine(self, lon1, lat1, lon2, lat2):
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1 
        dlat = lat2 - lat1 
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a)) 
        return 6371000 * c

    def calc_length(self, lons_list, lats_list):
        lengths = []
        for lons, lats in zip(lons_list, lats_list):
            if len(lons) < 2:
                lengths.append(0.0)
                continue
            lon1, lat1 = lons[:-1], lats[:-1]
            lon2, lat2 = lons[1:], lats[1:]
            
            segment_distances = self.haversine(lon1, lat1, lon2, lat2)
            lengths.append(np.sum(segment_distances))

        return np.array(lengths)

    def plot_and_fit(self, fit_range=(90, 150)):
        time, particle_lons, particle_lats = self.load_npz()
        time_hr = time / 3600.0
        length = self.calc_length(particle_lons, particle_lats)

        def linear(x, a, b):
            return a * x + b
        
        mask = (time_hr >= fit_range[0]) & (time_hr <= fit_range[1])
        
        log_length = np.log(length)
        popt, pcov = curve_fit(linear, time_hr[mask], log_length[mask])
        topo_entropy = popt[0]
        
        print(f"--- Result: Model: {self.model.upper()}, Level: {self.isobar} hPa ---")
        print(f"Topological entropy (h): {topo_entropy:.6f} [1/hr]")
        
        plt.figure(figsize=(9, 6))
        
        plt.semilogy(time_hr, length / 1000.0, label='Fonalhossz', color='navy', lw=1.5)
        
        fit_x = time_hr[mask]
        fit_y_log = linear(fit_x, *popt)
        plt.semilogy(fit_x, np.exp(fit_y_log) / 1000.0, color='red', linestyle='--', lw=2.5, 
                     label=f'Fit (h = {topo_entropy:.4f} 1/h)')

        plt.xlabel("Idő [óra]", fontsize=13)
        plt.ylabel("Fonalhossz [km]", fontsize=13)
        plt.title(f"Fonalhossz növekedése és topologikus entrópia ({self.model.upper()}, {self.isobar} hPa)", fontsize=14)
        plt.grid(True, which="both", ls="--", alpha=0.6)
        plt.legend(fontsize=11, loc="upper left")
        
        plt.tight_layout()
        
        save_path = f"yarn_length_fit_{self.model}_{self.isobar}hPa.png"
        plt.savefig(save_path, dpi=300)
        print(f"Graph saved: {save_path}\n")
        
        # plt.show()

#####################################################################################
if __name__ == "__main__":
    SELECTED_MODEL = "aifs"
    SELECTED_ISOBAR = 500
    LEAD_TIME_HR = 168

    analyzer = YarnEntropyPlotter(isobar=SELECTED_ISOBAR, model=SELECTED_MODEL, lead_time=LEAD_TIME_HR)
    analyzer.plot_and_fit(fit_range=(90, 150))
