import xarray as xr
import numpy as np


def extractData(filename, t, p, j, i):
    '''
    A function to extract data from .grib file. 
    Extracts wind direction (°) and speed (m/s) for a 'p' pressure level index, 
    'j' latitude index, 'i' longitude index, and 't' step index.
    Data Description available at: https://www.ecmwf.int/en/forecasts/datasets/set-ix
    '''

    try:
        if i >= 1440: i = 0
        if j >= 721: j = 720
        elif j < 0: j = 0

        ds_wind = xr.open_dataset(filename, engine="cfgrib", filter_by_keys={'typeOfLevel': 'isobaricInhPa'}, decode_timedelta=True)
        u = ds_wind["u"].isel(step=t, isobaricInhPa=p, latitude=j, longitude=i).values
        v = ds_wind["v"].isel(step=t, isobaricInhPa=p, latitude=j, longitude=i).values
        
        wind_speed = float(np.sqrt(u**2 + v**2))
        wind_dir_rad = np.arctan2(u, v)
        wind_direction = float((np.degrees(wind_dir_rad) + 180) % 360)
        
    except Exception:
        wind_direction, wind_speed = None, None

    return wind_direction, wind_speed


## TESTING ##
grib_file = "./ecmwf_aifs.grib"

timesteps = [0, 1, 2]
pressure_indices = [2, 5] # 850 hPa and 500 hPa
lat_indices = [100]
lon_indices = [1100]

print(f"{'t index':^15} | {'p index':^15} | {'j index':^15} | {'i index':^15} | {'wdir (°)':^15} | {'wspeed (m/s)':^15}")
print("-" * 102)

for t in timesteps:
    for p in pressure_indices:
        for j in lat_indices:
            for i in lon_indices:
                
                wdir, wspeed = extractData(grib_file, t=t, p=p, j=j, i=i)
                
                if wdir is not None and wspeed is not None:
                    print(f"{t:^15} | {p:^15} | {j:^15} | {i:^15} | {wdir:^15.1f} | {wspeed:^15.2f}")
                else:
                    print(f"{t:^15} | {p:^15} | {j:^15} | {i:^15} | {'N/A':^15} | {'N/A':^15}")
