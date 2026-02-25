import numpy as np
import lightkurve as lk
import os
from astropy.table import Table
from astropy.io import fits

target = "TIC 149154927"   
tic_id = "149154927"

sr = lk.search_lightcurve(target, mission="TESS", author="SPOC")
lc = sr.download()  # if multiple results, use sr[0].download() etc.

lc = lc.remove_nans().remove_outliers(sigma=5)
#lc = lc.remove_quality_flags()          
#lc = lc.remove_quality_flags(bitmask="default")
lc = lc.normalize()

t = Table()
t["TIME"] = np.array(lc.time.value)
t["PDCSAP_FLUX"] = np.array(lc.flux.value)

# print(lc.flux.value)
# exit()

# include errors if present
if getattr(lc, "flux_err", None) is not None:
    t["PDCSAP_FLUX_ERR"] = np.array(lc.flux_err.value)

# include quality if present (depends on product / cleaning)
if "QUALITY" in lc.colnames:
    t["QUALITY"] = np.array(lc["QUALITY"])

os.chdir(f"/Users/new/Dropbox/Desktop/AST192_Papers_Plots/MultiModes-main/Lightcurves")

hdu = fits.BinTableHDU(t)
hdul = fits.HDUList([fits.PrimaryHDU(), hdu])
hdul.writeto(f"TIC_{tic_id}_clean.fits", overwrite=True)