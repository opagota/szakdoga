#restrict number of used CPUs
import os
os.environ["OMP_NUM_THREADS"] = "5"
os.environ["OPENBLAS_NUM_THREADS"] = "5"
os.environ["MKL_NUM_THREADS"] = "5"
os.environ["VECLIB_MAXIMUM_THREADS"] = "5"
os.environ["NUMEXPR_NUM_THREADS"] = "5"

#for data processing
import cfgrib
import numpy as np
import xarray as xr
import time
from scipy.integrate import RK45
from scipy.optimize import curve_fit
from scipy.interpolate import RegularGridInterpolator
#for plotting
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

class traj:

    def __init__(self, data_dir, model, init_date, lead_time, isobar):
        self.data_dir = data_dir
        self.model = model
        self.init_date = init_date
        self.lead_time = lead_time #between 0 and 360 h
        self.isobar = isobar

        filepath = f"{self.data_dir}/ecmwf_{self.model}_{self.init_date}_0_RAPTOR_pl.grib"
        self.dataset = cfgrib.open_datasets(filepath)[0]

        self._interp_cache = {} #cache for faster run

    #**********************************************************************************TRAJECTORY CALCULATIONS**************************************************************************************

    def get_spatial_interpolators(self, time):
        """Creates and stores the wind field interpolator for the given hour, so that it does not need to be created again"""
        time = int(time)
        if time not in self._interp_cache:
            ds_slice = self.dataset.sel(isobaricInhPa=self.isobar, step=np.timedelta64(time, "h"))
            lats = ds_slice["latitude"].values
            lons = ds_slice["longitude"].values
            u = ds_slice["u"].values
            v = ds_slice["v"].values
            
            #coords have to be strictly increasing
            if lats[0] > lats[-1]:
                lats = lats[::-1]
                u = u[::-1, :]
                v = v[::-1, :]
                
            interp_u = RegularGridInterpolator((lats, lons), u, bounds_error=False, fill_value=None)
            interp_v = RegularGridInterpolator((lats, lons), v, bounds_error=False, fill_value=None)
            self._interp_cache[time] = (interp_u, interp_v)
            
        return self._interp_cache[time]

    def interpolate_spatially(self, coords, time):
        lons = coords[0].copy()
        lats = coords[1].copy()
        
        if self.dataset["longitude"].values.max() > 180 and np.any(lons < 0):
            lons = np.where(lons < 0, lons + 360, lons)
            
        interp_u, interp_v = self.get_spatial_interpolators(time)
        
        # scipy expects column pairs: (lat, lon)
        pts = np.column_stack((lats, lons))
        u_vals = interp_u(pts)
        v_vals = interp_v(pts)
        
        return np.vstack([u_vals, v_vals])
    
    def interpolate_temporally(self, coords, time):
        t1 = int(np.floor(time / 6.0) * 6)
        t2 = t1 + 6
        if np.isclose(time, t1):
            return self.interpolate_spatially(coords, time=t1)
        try:
            uv1 = self.interpolate_spatially(coords, time=t1)
            uv2 = self.interpolate_spatially(coords, time=t2)
        except ValueError as e:
            raise ValueError(f"Cannot temporally interpolate to forecast hour {time}. Original error: {e}")
        w = (time - t1) / (t2 - t1)
        uv_interpolated = (1.0 - w) * uv1 + w * uv2
        return uv_interpolated

    def haversine(self, lon1, lat1, lon2, lat2):
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1 
        dlat = lat2 - lat1 
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a)) 
        return 6371000 * c
        
    def calculate_trajectory(self, lon0, lat0, t_span):
        lon0 = np.atleast_1d(lon0)
        lat0 = np.atleast_1d(lat0)
        N_initial = len(lon0)
        
        if len(lat0) != N_initial:
            raise ValueError("lon0 and lat0 must have the same length.")
            
        R = 6371000.0 # earth radius
        C = 180.0 / (np.pi * R) # conversion factor: from m to deg
        
        def ode_func(t, y):
            N_current = len(y) // 2 
            lons = y[:N_current]
            lats = y[N_current:]
            coords = np.array([lons, lats])
            
            try: 
                uv = self.interpolate_temporally(coords, time=t/3600) 
            except Exception as e:
                raise RuntimeError(f"Error evaluating wind at forecast hour {t}: {e}")
                
            u = uv[0]
            v = uv[1]
            
            lats_safe = np.clip(lats, -89.9, 89.9)
            cos_lat = np.cos(np.radians(lats_safe))
            
            dlon_dt = (u * C) / cos_lat
            dlat_dt = v * C
            
            return np.concatenate([dlon_dt, dlat_dt])
            
        def calc_dists(lons, lats):
            lon1 = lons[:-1]
            lat1 = lats[:-1]
            lon2 = lons[1:]
            lat2 = lats[1:]
            return self.haversine(lon1, lat1, lon2, lat2)
            
        def insert_new_particles(lons, lats, distances, threshold):
            new_lons = [lons[0]]
            new_lats = [lats[0]]
            
            for i in range(len(distances)):
                if distances[i] > threshold:
                    mid_lon = (lons[i] + lons[i+1]) / 2.0
                    mid_lat = (lats[i] + lats[i+1]) / 2.0
                    new_lons.append(mid_lon)
                    new_lats.append(mid_lat)
                    
                new_lons.append(lons[i+1])
                new_lats.append(lats[i+1])
                
            return np.array(new_lons), np.array(new_lats)
            
        def integrater(ode_f, t0, y0, t_bound, dist_threshold):
            tlist = []
            ylist = []
            
            t_current = t0
            y_current = y0
            
            tlist.append(t_current)
            ylist.append(y_current.copy())
            
            last_saved_t = t_current
            save_interval = 3600.0  # higher save_interval for faster run
            
            while t_current < t_bound:
                rk45 = RK45(ode_f, t_current, y_current, t_bound, max_step=600.0, atol=1e-3, rtol=1e-3)
                
                while rk45.status == 'running':
                    rk45.step()
                    
                    # memory protection
                    if (rk45.t - last_saved_t) >= save_interval or rk45.t >= t_bound:
                        tlist.append(rk45.t)
                        ylist.append(rk45.y.copy())
                        last_saved_t = rk45.t
                    
                    N_current = len(rk45.y) // 2
                    lons = rk45.y[:N_current]
                    lats = rk45.y[N_current:]
                    
                    distances = calc_dists(lons, lats)
                    
                    if np.any(distances > dist_threshold):
                        t_current = rk45.t
                        lons_updated, lats_updated = insert_new_particles(lons, lats, distances, dist_threshold)
                        y_current = np.concatenate([lons_updated, lats_updated])
                        
                        print(f"[{t_current:.1f} s simulation time] New particle inserted: {N_current} -> {len(lons_updated)} particles")
                        break
                else: 
                    t_current = t_bound

            return np.array(tlist), ylist

        
        y0 = np.concatenate([lon0, lat0]) 
        t_start = t_span[0]
        t_final = t_span[1]
        
        threshold_meters = 20000.0 

        time_arr, y_list_of_arrays = integrater(ode_func, t_start, y0, t_final, threshold_meters)

        particle_lons = []
        particle_lats = []
        
        for y in y_list_of_arrays:
            N = len(y) // 2
            particle_lons.append(y[:N])
            particle_lats.append(y[N:])
            
        return time_arr, particle_lons, particle_lats
        
    def plot_traj(self, time_arr, lons_list, lats_list, outpath=''):

        #solution lats & lons
        all_lons = np.concatenate(lons_list)
        all_lats = np.concatenate(lats_list)

        #calculate map extent (1-deg buffer)
        min_lon, max_lon = all_lons.min() - 1.0, all_lons.max() + 1.0
        min_lat, max_lat = all_lats.min() - 1.0, all_lats.max() + 1.0
        
        #setup fig
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={"projection": ccrs.PlateCarree()})
        ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())

        #geographical background
        ax.add_feature(cfeature.LAND, facecolor="#f5f5f5", zorder=0)
        ax.add_feature(cfeature.OCEAN, facecolor="#e0f2fe", zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor="#333333", zorder=1)
        ax.add_feature(cfeature.BORDERS, linestyle=":", alpha=0.5, zorder=1)

        #gridline & labels
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.6, linestyle="--", zorder=2)
        gl.top_labels = False
        gl.right_labels = False

        #plot yarn path and start/end points
        ax.plot(lons_list[0], lats_list[0], transform=ccrs.PlateCarree(), color="green", linewidth=2, label="Kezdeti fonal")
        ax.scatter(lons_list[0], lats_list[0], transform=ccrs.PlateCarree(), color="darkgreen", s=10, zorder=5)
        ax.plot(lons_list[-1], lats_list[-1], transform=ccrs.PlateCarree(), color="crimson", linewidth=2, label="Végső fonal")
        ax.scatter(lons_list[-1], lats_list[-1], transform=ccrs.PlateCarree(), color="darkred", s=10, zorder=5)
        ax.legend()
        
        #title & save
        ax.set_title(f"Részecske trajektóriák (ECMWF-{self.model.upper()}, 0.005°, {self.isobar} hPa, {self.lead_time} h)\n Zöld: kiindulópont, piros: végpont", fontsize=12, fontweight="bold", pad=10)
        plt.tight_layout()
        img_filename = f"trajs_{self.isobar}hPa_{self.lead_time}h_{self.model}.png"
        plt.savefig(img_filename, dpi=300)
        plt.close()
        return img_filename

#************************************************************************************************************************************************************************************************************************************

if __name__ == "__main__":
    lat_start = np.arange(45.0, 55.01, 0.05)
    lon_start = np.zeros(lat_start.shape) + 20
    #lon_start = np.zeros(lat_start.shape) + 60
    init_date = 20260601
    lead_time = 6
    isobar = 500

    t_span = (0, lead_time*3600)

    # --- AIFS SIMULATION ---
    print("\n" + "="*50)
    print("STARTING AIFS SIMULATION")
    print("="*50)
    start_time_aifs = time.time()
    
    traj_aifs = traj("../RAPTOR/met", "aifs", init_date, lead_time, isobar)
    time_aifs, particle_lons_aifs, particle_lats_aifs = traj_aifs.calculate_trajectory(lon_start, lat_start, t_span)
    
    img_file_aifs = traj_aifs.plot_traj(time_aifs, particle_lons_aifs, particle_lats_aifs)
    
    npz_file_aifs = f"trajectory_solution_{lead_time}h_{isobar}hPa_aifs.npz"
    np.savez_compressed(npz_file_aifs, time=time_aifs, lons=np.array(particle_lons_aifs, dtype=object), lats=np.array(particle_lats_aifs, dtype=object))
    
    end_time_aifs = time.time()
    run_time_aifs = end_time_aifs - start_time_aifs
    
    print(f"\n[AIFS READY]")
    print(f"Run time: ({run_time_aifs/60:.2f} minutes)")
    print(f"Saved figure: {img_file_aifs}")
    print(f"Saved .npz data: {npz_file_aifs}")


    # --- IFS SIMULATION ---
    print("\n" + "="*50)
    print("STARTING IFS SIMULATION")
    print("="*50)
    start_time_ifs = time.time()
    
    traj_ifs = traj("../RAPTOR/met", "ifs", init_date, lead_time, isobar)
    time_ifs, particle_lons_ifs, particle_lats_ifs = traj_ifs.calculate_trajectory(lon_start, lat_start, t_span)
    
    img_file_ifs = traj_ifs.plot_traj(time_ifs, particle_lons_ifs, particle_lats_ifs)
    
    npz_file_ifs = f"trajectory_solution_{lead_time}h_{isobar}hPa_ifs.npz"
    np.savez_compressed(npz_file_ifs, time=time_ifs, lons=np.array(particle_lons_ifs, dtype=object), lats=np.array(particle_lats_ifs, dtype=object))
    
    end_time_ifs = time.time()
    run_time_ifs = end_time_ifs - start_time_ifs
    
    print(f"\n[IFS READY]")
    print(f"Run time: ({run_time_ifs/60:.2f} minutes)")
    print(f"Saved figure: {img_file_ifs}")
    print(f"Saved .npz data: {npz_file_ifs}")

    print("\n****************************************** ALL READY ******************************************")
