#! /usr/bin/env python
"""Summary statistics over all S2 velocity maps (veloc_x / veloc_y GeoTIFFs) in a directory."""
from glob import glob
import sys
import numpy as np
import xarray as xr
import rioxarray
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def stack(data_dir, comp):
    files = sorted(glob(f"{data_dir}/**/S2_*_veloc_{comp}.tif", recursive=True))
    if not files:
        sys.exit(f"no S2_*_veloc_{comp}.tif files found under {data_dir}")
    ref = rioxarray.open_rasterio(files[0], masked=True).squeeze(drop=True)
    das = [rioxarray.open_rasterio(f, masked=True).squeeze(drop=True).rio.reproject_match(ref) for f in files]
    return xr.concat(das, dim="pair"), files


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    vx, files = stack(data_dir, "x")
    vy, _ = stack(data_dir, "y")
    print(f"{len(files)} velocity maps")

    # median of the components first, then speed: the median of per-pair speeds is
    # biased high wherever true motion is below the noise level
    vx_med = vx.median("pair")
    vy_med = vy.median("pair")
    speed_med = np.hypot(vx_med, vy_med)
    speed_std = np.hypot(vx, vy).std("pair")
    count = vx.notnull().sum("pair")

    for name, da in [("median_veloc_x", vx_med), ("median_veloc_y", vy_med),
                     ("median_horizontal_velocity_magnitude", speed_med),
                     ("stdev_horizontal_velocity_magnitude", speed_std),
                     ("count_horizontal_velocity_magnitude", count.astype("float32"))]:
        da.rio.write_crs(vx.rio.crs).rio.to_raster(f"{name}.tif")

    frac = np.isfinite(vx).mean(("y", "x")).values
    print(f"valid fraction per pair: min={frac.min():.3f} median={np.median(frac):.3f} max={frac.max():.3f}")
    print(f"median speed over AOI: {float(speed_med.median()):.2f} m/yr; 95th pct: {float(speed_med.quantile(0.95)):.2f} m/yr")

    vmax = float(np.nanpercentile(speed_med.values, 99))
    f, ax = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    speed_med.plot(ax=ax[0], vmin=0, vmax=vmax, cmap="inferno", cbar_kwargs={"shrink": 0.7, "label": "velocity (m/yr)"})
    speed_std.plot(ax=ax[1], vmin=0, vmax=vmax, cmap="cividis", cbar_kwargs={"shrink": 0.7, "label": "standard deviation (m/yr)"})
    count.plot(ax=ax[2], vmin=0, cmap="Blues", cbar_kwargs={"shrink": 0.7, "label": "pixel count"})
    for a, t in zip(ax, ["median velocity", "velocity standard deviation", "valid pixel count"]):
        a.set_aspect("equal"); a.set_title(t); a.set_xticks([]); a.set_yticks([]); a.set_xlabel(""); a.set_ylabel("")
    f.tight_layout()
    f.savefig("velocity_summary_statistics.png", dpi=300)


if __name__ == "__main__":
    main()
