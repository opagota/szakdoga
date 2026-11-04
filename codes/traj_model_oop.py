#for data processing
import cfgrib
import numpy as np
import xarray as xr
from scipy.integrate import solve_ivp
#for plotting
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

class traj():

    def __init__(self, data_dir, model, init_date, lead_time, isobar):
        self.data_dir = data_dir
        self.model = model
        self.init_date = init_date
        self.lead_time = lead_time #between 0 and 360 h
        self.isobar = isobar

        filepath = f"{self.data_dir}/ecmwf_{self.model}_{self.init_date}_0_RAPTOR_pl.grib"
        self.dataset = cfgrib.open_datasets(filepath)[0]
        
    def create_dataset(self, main_dir):
        return cfgrib.open_datasets(f"{self.data_dir}/ecmwf_{self.model}_{self.init_date}_0_RAPTOR_pl.grib")

    def interpolate_spatially(self, coords, time, interpolation="linear"):
            
        #slice the dataset
        ds_slice = self.dataset.sel(isobaricInhPa=self.isobar, step=np.timedelta64(int(time), "h"))

        #extract coordinates & convert input longitudes from (-180, 180) to (0, 360) (dataset uses 0-360)
        lons = coords[0].copy()
        lats = coords[1].copy()
        if ds_slice["longitude"].values.max() > 180 and np.any(lons < 0):
            lons = np.where(lons < 0, lons + 360, lons)

        #execute linear spatial interpolation
        u_interp = ds_slice["u"].interp({"longitude": xr.DataArray(lons, dims="points"), "latitude": xr.DataArray(lats, dims="points")}, method=interpolation)
        v_interp = ds_slice["v"].interp({"longitude": xr.DataArray(lons, dims="points"), "latitude": xr.DataArray(lats, dims="points")}, method=interpolation)

        #return interpolated values
        return np.vstack([u_interp.values, v_interp.values])

    def interpolate_temporally(self, coords, time, interpolation="linear"): #flexible 'time' variable is needed: RK'll use this method -> in spatial interpolation as well

        #find neighboring 6-hourly steps
        t1 = int(np.floor(time / 6.0) * 6)
        t2 = t1 + 6
        
        #if 'time' is a 6-hourly step exactly
        if np.isclose(time, t1):
            return self.interpolate_spatially(coords, time=t1, interpolation=interpolation)
        
        #fetch fields for lower and upper neighbor
        try:
            uv1 = self.interpolate_spatially(coords, self.isobar, time=t1, interpolation=interpolation)
            uv2 = self.interpolate_spatially(coords, self.isobar, time=t2, interpolation=interpolation)
          except ValueError as e:
            raise ValueError(f"Cannot temporally interpolate to forecast hour {time}. Ensure both steps {t1} and {t2} exist in the file. Original error: {e}")
        
        #calculate 'w' termporal weight factor (0 <= w <= 1)
        w = (time - t1) / (t2 - t1)

        #execture linear temporal interpolation
        uv_interpolated = (1.0 - w) * uv1 + w * uv2

        #return
        return uv_interpolated

    def calculate_trajectory(self, lon0, lat0, t_span, t_eval=None):
        lon0 = np.atleast_1d(lon0)
        lat0 = np.atleast_1d(lat0)
        N = len(lon0)

        if len(lat0) != N:
            raise ValueError("lon0 and lat0 must have the same length.")

        R = 6371000.0 #earth radius in meters
        C = 180.0 / (np.pi * R) #conversion factor: from m to deg

        def ode_func(t, y):
            # y contains [lon_1, ..., lon_N, lat_1, ..., lat_N]
            lons = y[:N]
            lats = y[N:]
            
            #construct coords array
            coords = np.array([lons, lats])

            #interpolate
            try:
              uv = self.interpolate_temporally(coords, time=t/3600, interpolation="linear") 
            except Exception as e:
              raise RuntimeError(f"Error evaluating wind at forecast hour {t}: {e}")

            #extract u and v
            u = uv[0]
            v = uv[1]

            #prevent division with 0
            lats_safe = np.clip(lats, -89.9, 89.9)
            cos_lat = np.cos(np.radians(lats_safe))

            #derivatives in deg/s (instead of m/s)
            dlon_dt = (u * C) / cos_lat
            dlat_dt = v * C

            return np.concatenate([dlon_dt, dlat_dt])

        #initial state vector
        y0 = np.concatenate([lon0, lat0])

        #solve ODE
        solution = solve_ivp(ode_func, t_span, y0, method="RK45", t_eval=t_eval, rtol=1e-3, atol=1e-3, max_step=600.0)  #shape of solution: (number_of_particles, num_time_steps)

        #save solution
        np.savez_compressed(f"./traj_sols/trajectory_solution_{self.lead_time}h_500m_{self.isobar}hPa_{self.model}.npz", t=solution.t, y=solution.y)

        #extract solutions
        N = len(lon0)
        time = solution.t
        particle_lons = solution.y[:N, :]  # Shape: (N, len(time))
        particle_lats = solution.y[N:, :]
        
        #return
        return solution, time, particle_lons, particle_lats

    def plot(self, sol, lon0=None, lat0=None,outpath=''):

        #solution lats & lons
        N = sol.y.shape[0] // 2
        lons = sol.y[:N, :] 
        lats = sol.y[N:, :]

        #calculate map extent (1-deg buffer)
        min_lon, max_lon = lons.min() - 1.0, lons.max() + 1.0
        min_lat, max_lat = lats.min() - 1.0, lats.max() + 1.0

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

        #plot every particle's path AND start/end points (Javítva: ciklusba illesztve)
        for i in range(N):
            ax.plot(lons[i], lats[i], transform=ccrs.PlateCarree(), linewidth=1.8, alpha=0.8, zorder=3)
            ax.plot(lons[i, 0], lats[i, 0], marker="o", color="green", markersize=5, transform=ccrs.PlateCarree(), zorder=4)
            ax.plot(lons[i, -1], lats[i, -1], marker="o", color="crimson", markersize=5, transform=ccrs.PlateCarree(), zorder=4)

        #title & save
        ax.set_title(f"Részecske trajektóriák (ECMWF-{self.model.upper()}, 0.005°, {self.isobar} hPa, {self.lead_time} h)\n Zöld: kiindulópont, piros: végpont", fontsize=12, fontweight="bold", pad=10)
        plt.tight_layout()
        plt.savefig(f"trajs_{self.isobar}hPa_{self.lead_time}h_{self.model}.png",dpi=300)
        plt.close()


#************************************************************************************************************************************************************************************************************************************

#initialization
lat_start = np.arange(45.0, 55.01, 0.25)
lon_start = np.zeros(lat_start.shape) + 20
init_date = 20260601
lead_time = 168
isobar = 500
tstep = 60  # seconds timestep --> this is the output timestep! The internal integration timestep is automatically set by Runge-Kutta.

#derived things
t_span = (0, lead_time*3600)
t_eval = np.arange(0, lead_time*3600, tstep)

#instantiation
traj_aifs = traj("../RAPTOR/met", "aifs", init_date, lead_time, isobar)
sim_aifs, time_aifs, particle_lons_aifs, particle_lats_aifs = traj_aifs.calculate_trajectory(lon_start, lat_start, t_span, t_eval=t_eval)
traj_aifs.plot(sim_aifs)

traj_ifs = traj("../RAPTOR/met", "ifs", init_date, lead_time, isobar)
sim_ifs, time_ifs, particle_lons_ifs, particle_lats_ifs = traj_ifs.calculate_trajectory(lon_start, lat_start, t_span, t_eval=t_eval)
traj_ifs.plot(sim_ifs)









