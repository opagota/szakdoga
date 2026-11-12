import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import t
import math

class YarnEntropyPlotter:
    def __init__(self, isobar, model, init_date, lead_time, spat_res, area):
        self.isobars = np.atleast_1d(isobars)
        self.models = np.atleast_1d(models)
        self.init_dates = np.atleast_1d(init_dates)
        self.lead_times = np.atleast_1d(lead_times)
        self.spat_res = np.atleast_1d(spat_res)
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
    
    def format_unc(self, val, err):
        if err <= 0 or math.isnan(err):
            return f'{val:.4f}'
        decimals = -int(math.floor(math.log10(abs(err))))
        decimals = max(0, decimals)
        val_rounded = round(val, decimals)
        err_int = int(round(err * (10**decimals)))

        return f'{val_rounded:.{decimals}f}({err_int})'

    def plot_and_fit(self):
        def linear(x, a, b):
            return a * x + b

        color_map = {
            'aifs': {'line': 'navy', 'fit': 'blue'},
            'ifs': {'line': 'darkred', 'fit': 'red'},
            'default': {'line': 'darkgreen', 'fit': 'limegreen'}
        }

        for iso in self.isobars:
            for date in self.init_dates:
                for lt in self.lead_times:
                    for res in self.spat_res:
                        
                        fig, ax = plt.subplots(figsize=(10, 7))
                        has_data = False

                        for model in self.models:
                            try:
                                time, particle_lons, particle_lats = self.load_npz(model, iso, date, lt, res)
                            except FileNotFoundError as e:
                                print(e)
                                continue

                            has_data = True
                            time_hr = time / 3600.0
                            length = self.calc_length(particle_lons, particle_lats)

                            if iso == 500:
                                fit_range = (90, 160)
                            elif iso == 700:
                                fit_range = (90, 150)
                            elif iso == 850:
                                fit_range = (125, 160)
                            else:
                                fit_range = (90, 150)
                                
                            mask = (time_hr >= fit_range[0]) & (time_hr <= fit_range[1])
                            log_length = np.log(length)

                            popt, pcov = curve_fit(linear, time_hr[mask], log_length[mask])
                            a_val = popt[0]
                            se = np.sqrt(pcov[0, 0])

                            n = np.sum(mask)
                            df = n - 2
                            alpha = 0.025
                            t_crit = t.ppf(1 - alpha, df)

                            moe = t_crit * se
                            lower = a_val - moe
                            upper = a_val + moe

                            h_text = self.format_unc(a_val, moe)

                            print(f"--- Result: Model: {model.upper()}, Level: {iso} hPa, Date: {date}, Resolution: {res}˚, Lead time: {lt}h ---")
                            print(f"Topological entropy (h): {h_text} [1/hr]")
                            print(f"Confidence Interval (95%): ({lower:.4f}, {upper:.4f})\n")

                            c = color_map.get(model.lower(), color_map['default'])

                            ax.semilogy(time_hr, length / 1000.0, label=f"Fonalhossz ({model.upper()})", color=c['line'], lw=1.5, alpha=0.7)

                            fit_x = time_hr[mask]
                            fit_y_log = linear(fit_x, *popt)

                            var_a, var_b, cov_ab = pcov[0, 0], pcov[1, 1], pcov[0, 1]
                            se_log = np.sqrt(var_a * fit_x**2 + var_b + 2 * fit_x * cov_ab)
                            moe_log = t_crit * se_log

                            fit_y_upper = np.exp(fit_y_log + moe_log) / 1000.0
                            fit_y_lower = np.exp(fit_y_log - moe_log) / 1000.0
                            fit_y = np.exp(fit_y_log) / 1000.0

                            ax.semilogy(
                                fit_x,
                                fit_y,
                                color=c['fit'],
                                linestyle="--",
                                lw=2.5,
                                label=f"{model.upper()} fit: h = {h_text} 1/h",
                            )
                            ax.fill_between(
                                fit_x,
                                fit_y_lower,
                                fit_y_upper,
                                color=c['fit'],
                                alpha=0.2,
                                label=f"{model.upper()} 95% CI",
                            )

                        if has_data:
                            plt.xlabel("Idő [óra]", fontsize=13)
                            plt.ylabel("Fonalhossz [km]", fontsize=13)
                            plt.title(f"Fonalhossz növekedése ({iso} hPa, {date}, {res}°)", fontsize=14)
                            plt.grid(True, which="both", ls="--", alpha=0.6)
                            plt.legend(fontsize=11, loc="upper left")
                            plt.tight_layout()

                            os.makedirs(f"./{self.area}", exist_ok=True)

                            save_path = (f"./combined/{self.area}/yarn_length_fit_compare_{iso}hPa_{date}_{lt}h_{str(res).replace('.', '')}.png")
                            plt.savefig(save_path, dpi=300)
                            print(f"Graph saved: {save_path}\n")
                        
                        plt.close()


#####################################################################################
if __name__ == "__main__":
    SELECTED_MODELS = ["aifs", "ifs"]
    SELECTED_ISOBARS = [500, 700, 850]
    INIT_DATES = [20260601, 20260701, 20260801, 20260901, 20261001]
    LEAD_TIMES_HR = [168]
    SPAT_RES = [0.005]
    AREA = "hun" # hun, jet, trade

    analyzer = YarnEntropyPlotter(
        isobars=SELECTED_ISOBARS, 
        models=SELECTED_MODELS, 
        init_dates=INIT_DATES, 
        lead_times=LEAD_TIMES_HR, 
        spat_res=SPAT_RES, 
        area=AREA
    )
    
    analyzer.plot_and_fit()
