import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os

class YarnEntropyPlotter:
    def __init__(self, isobar, model, init_date, lead_time, spat_res, area):
        self.isobars = np.atleast_1d(isobar)
        self.init_dates = np.atleast_1d(init_date)
        
        self.model = model
        self.lead_time = lead_time
        self.spat_res = spat_res
        self.area = area

    def load_npz(self, iso, date):
        filepath = f"../traj_sols/{self.area}/trajectory_solution_{self.lead_time}h_{iso}hPa_{date}_{self.model}_{str(self.spat_res).replace('.', '')}.npz"
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
            
        sol = np.load(filepath, allow_pickle=True)
        time = sol['time']
        particle_lons_list = list(sol['lons'])
        particle_lats_list = list(sol['lats'])
    
        return time, particle_lons_list, particle_lats_list

    def haversine(self, lon1, lat1, lon2, lat2):
        lon1 = np.asarray(lon1, dtype=float)
        lat1 = np.asarray(lat1, dtype=float)
        lon2 = np.asarray(lon2, dtype=float)
        lat2 = np.asarray(lat2, dtype=float)

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
        def linear(x, a, b):
            return a * x + b
            
        for iso in self.isobars:
            for date in self.init_dates:
                try:
                    time, particle_lons, particle_lats = self.load_npz(iso, date)
                except FileNotFoundError as e:
                    print(e)
                    continue
                
                time_hr = time / 3600.0
                length = self.calc_length(particle_lons, particle_lats)
                
                mask = (time_hr >= fit_range[0]) & (time_hr <= fit_range[1])
                log_length = np.log(length)
                
                popt, pcov = curve_fit(linear, time_hr[mask], log_length[mask])
                topo_entropy = popt[0]
                
                print(f"--- Result: Model: {self.model.upper()}, Level: {iso} hPa, Date: {date}, Resolution: {self.spat_res}˚ ---")
                print(f"Topological entropy (h): {topo_entropy:.6f} [1/hr]")
                
                plt.figure(figsize=(9, 6))
                plt.semilogy(time_hr, length / 1000.0, label='Fonalhossz', color='navy', lw=1.5)
                
                fit_x = time_hr[mask]
                fit_y_log = linear(fit_x, *popt)
                plt.semilogy(fit_x, np.exp(fit_y_log) / 1000.0, color='red', linestyle='--', lw=2.5, 
                             label=f'Fit (h = {topo_entropy:.4f} 1/h)')

                plt.xlabel("Idő [óra]", fontsize=13)
                plt.ylabel("Fonalhossz [km]", fontsize=13)
                plt.title(f"Fonalhossz növekedése ({self.model.upper()}, {iso} hPa, {date}, {self.spat_res}°)", fontsize=14)
                plt.grid(True, which="both", ls="--", alpha=0.6)
                plt.legend(fontsize=11, loc="upper left")
                plt.tight_layout()
                
                os.makedirs(f"./{self.area}", exist_ok=True)
                
                save_path = f"./{self.area}/yarn_length_fit_{self.model}_{iso}hPa_{date}_{str(self.spat_res).replace('.', '')}.png"
                plt.savefig(save_path, dpi=300)
                plt.close()
                
                print(f"Graph saved: {save_path}\n")

#####################################################################################
if __name__ == "__main__":
    SELECTED_MODEL = "ifs"
    SELECTED_ISOBAR = [500, 700, 850]
    INIT_DATE = np.array([20260601, 20260701, 20260801, 20260901, 20261001])
    LEAD_TIME_HR = 168
    SPAT_RES = 0.005
    AREA = "jet" #hun, jet, trade

    analyzer = YarnEntropyPlotter(isobar=SELECTED_ISOBAR, model=SELECTED_MODEL, init_date=INIT_DATE, lead_time=LEAD_TIME_HR, spat_res=SPAT_RES, area=AREA)
    analyzer.plot_and_fit(fit_range=(90, 150))
