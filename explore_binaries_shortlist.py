import lightkurve as lk
import numpy as np
import matplotlib.pyplot as plt
from lightkurve import search_targetpixelfile
import wotan
from astropy.timeseries import LombScargle
import tabulate
from tqdm import tqdm
import matplotlib.cm as cm
import pandas as pd
from astropy.table import Table
import re
import tempfile
from astroquery.mast import Observations
#Observations.MAST_REQUESTS_URL = "https://mast.stsci.edu"
import os
import matplotlib as mpl

mpl.rcParams['figure.dpi']   = 120      
mpl.rcParams['savefig.dpi']  = 500      

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

####### FUNCTIONS ########
def clean_tpf(t):
    t_clean = t.remove_nans().remove_outliers()
    return t_clean

def detrend(t,window_length):
    #t_clean = t.flatten()
    time = t.time.value
    flux_detrended, trend = wotan.flatten(time, t.flux.value, window_length=window_length, return_trend=True)
    lc_detrended = lk.LightCurve(time=time, flux=flux_detrended)
    return lc_detrended, trend

def download_sections(t,wanted):
    with tempfile.TemporaryDirectory() as tmp:
        rows = []
        for i, seq in enumerate(t.table["sequence_number"]):
            if seq in wanted:
                rows.append(i)

        tpfs = []
        for i in rows:
            downloaded_tpf = t[i].download()
            tpfs.append(downloaded_tpf)

    return tpfs

def read_data_from_file(path):
    obj_with_data_inst, bp_rp_with_data_inst, abs_magn_with_data_inst = [], [], []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if i < 2:      
                continue
            s = line.strip()
            parts = re.split(r"\s{2,}", s)
            if len(parts) < 3:
                continue

            obj_with_data_inst.append((parts[0]))
            bp_rp_with_data_inst.append(float(parts[1]))
            abs_magn_with_data_inst.append(float(parts[2]))

    return obj_with_data_inst, bp_rp_with_data_inst, abs_magn_with_data_inst

def instability_strip(tic_names, bp_rp_primaries, abs_magn_primaries, st):
    obj_with_data_inst, bp_rp_with_data_inst, abs_magn_with_data_inst = [], [], []
    for j in tqdm(range(15000-st)):
        if bp_rp_primaries[j] > 1.5:
            continue
        if abs_magn_primaries[j] > 5.0:
            continue
        s = tic_names[j]
        obj_tic = "TIC " + str(s)
        tpf = search_targetpixelfile(obj_tic, mission = name, author='SPOC')
        if(len(tpf)==0):
            continue
        print("tpf found for ", obj_tic)
        obj_with_data_inst.append(obj_tic)
        bp_rp_with_data_inst.append(bp_rp_primaries[j])
        abs_magn_with_data_inst.append(abs_magn_primaries[j])

    ## save data of identified objects to file ###
    headers = ['TIC', 'BP_RP', 'abs M']
    data_table = list(zip(obj_with_data_inst, bp_rp_with_data_inst, abs_magn_with_data_inst))
    print(tabulate.tabulate(data_table, headers=headers))
    tabletable = tabulate.tabulate(data_table, headers=headers)

    file_name = f"list_of_obj_in_inst_with_data_range_{st}_15000.dat"
    with open(file_name, "w") as f:
        f.write(tabletable)

    # arr = np.column_stack([obj_with_data_inst, bp_rp_with_data_inst, abs_magn_with_data_inst])
    # np.savetxt("list_of_obj_in_inst_with_data_range_{st}_15000.csv",
    #         arr, delimiter=",", header="TIC,BP_RP,abs_M")
    
    plt.figure()
    plt.scatter(bp_rp_primaries, abs_magn_primaries, s=5, color = 'deepskyblue')
    plt.scatter(bp_rp_with_data_inst, abs_magn_with_data_inst, s=20, color = 'mediumvioletred', label='candidates', marker = '*')
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.title('HR Diagram')
    plt.xlabel('BP-RP')
    plt.ylabel('abs M')
    plt.legend(loc='best')
    plt.savefig('HR_all.png', dpi=500)
    plt.show()
    
    return obj_with_data_inst, bp_rp_with_data_inst, abs_magn_with_data_inst

def plot_candidate(ax, tic_names, bp_rp_primaries, bp_rp_secondaries, bp_rp_cand, abs_magn_primaries, abs_magn_secondaries, abs_magn_cand, tic_candidate):
    tic_names = np.array(tic_names)
    bp_rp_cand = np.array(bp_rp_cand)
    abs_magn_cand = np.array(abs_magn_cand)

    try:
        idx = np.flatnonzero(tic_names == int(tic_candidate))
    except Exception:
        idx = np.flatnonzero(tic_names == str(tic_candidate))

    ax.scatter(bp_rp_primaries, abs_magn_primaries, s=25, color = 'deepskyblue', label='primaries')
    ax.scatter(bp_rp_secondaries, abs_magn_secondaries, s=1, color = 'navy', label='secondaries')
    ax.scatter(bp_rp_cand[idx], abs_magn_cand[idx], s=80, color = 'mediumvioletred', label=f'{tic_candidate}', marker = '*')
    ax.invert_yaxis()
    ax.grid(True)
    ax.set_title('HR Diagram')
    ax.set_xlabel('BP-RP')
    ax.set_ylabel('abs M')
    ax.legend(loc='best')
    #plt.savefig(f'HR_candidate_{tic_candidate}.png', dpi=500)

def eyeball_lightcurves(obj_with_data_inst, savefiles):
    for l, obj_lab in tqdm(enumerate(obj_with_data_inst)):
        tpf = search_targetpixelfile(obj_lab, mission = name, author='SPOC')

        ### find different sections for object ###
        wanted = list(tpf.table["sequence_number"][:5])
        tpfs = download_sections(tpf, wanted)
        cleaned_tpfs = []

        ### download and clean lightcurves up ###
        for i,t in enumerate(tpfs):
            lightcurve = t.to_lightcurve(aperture_mask=t.pipeline_mask)
            lc_clean = clean_tpf(lightcurve)
            cleaned_tpfs.append(lc_clean)

            lc_clean.scatter()
            plt.title(f'Lightcurves {obj_lab}')
            #plt.xlim([2735,2740])
            plt.grid(True)
            if(savefiles):
                plt.savefig(f"{obj_lab}_cleaned_prestitch_{i}.png", dpi=500)
            plt.show()

        ### stitch unfolded lightcurves together ###
        coll = lk.LightCurveCollection(cleaned_tpfs)
        lc_stitched = coll.stitch()

        ### get orbital period and orbital transit time ###
        pg = lc_stitched.to_periodogram(method='lombscargle', minimum_frequency=5, maximum_frequency=80)
        period_at_max_power = pg.period_at_max_power
        print(f"Period at max power: {period_at_max_power}")
        pg.plot()      
        plt.title(f'Periodogram of {obj_lab}')
        if(savefiles):
            plt.savefig(f"{obj_lab}_pg_period.png", dpi=500)
        plt.show()

#############################
#############################
######## import data ########
name = 'TESS'
savefiles = False
path = "WD_Binaries_HUGE.csv"
df = pd.read_csv(path)
st = 0#15000  ### early stop
#print(df.columns.to_series())

tic_names = df['TIC_1'].tolist()
gaia_id_names = df['source_id1'].tolist()
bp_rp_primaries = df['bp_rp1'].to_numpy(float)
magn_primaries = df['phot_g_mean_mag1'].to_numpy(float)
parallax = df['parallax1'].to_numpy(float)
separation = df['sep_AU'].to_numpy(float)
tic_secondaries = df['TIC_2'].tolist()
bp_rp_secondaries = df['bp_rp2'].to_numpy(float)
magn_secondaries = df['phot_g_mean_mag2'].to_numpy(float)
parallax_sec = df['parallax2'].to_numpy(float)
#print(len(tic_names))

abs_magn_primaries = magn_primaries+5-5*np.log10(1/(parallax*10**(-3)))
abs_magn_secondaries = magn_secondaries+5-5*np.log10(1/(parallax_sec*10**(-3)))

obj_with_data_inst = []
bp_rp_with_data_inst = []
abs_magn_with_data_inst = []

### once shortlist has been made ###
num_shortlist = []
bp_rp_shortlist = []
abs_magn_shortlist = []
tic_shortlist = []

with open("shortlist_obs.dat", "r") as f:
        for i, line in enumerate(f):
            if i < 2:      
                continue
            s = line.strip()
            num_shortlist.append(s)

obj_with_data_inst = np.array(obj_with_data_inst)
bp_rp_with_data_inst = np.array(bp_rp_with_data_inst)
abs_magn_with_data_inst = np.array(abs_magn_with_data_inst)

tic_names = np.array(tic_names)

for h,string_id in enumerate(num_shortlist):
    string_tic = 'TIC ' + str(string_id)
    tic_shortlist.append(string_tic)
    bp_rp_shortlist.append(bp_rp_primaries[np.flatnonzero(tic_names == int(string_id))])
    abs_magn_shortlist.append(abs_magn_primaries[np.flatnonzero(tic_names == int(string_id))])

num_shortlist = np.array(num_shortlist)
tic_shortlist = np.array(tic_shortlist)

for id in tic_shortlist:
    fig, axs = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    plot_candidate(axs[0], tic_shortlist, bp_rp_primaries, bp_rp_secondaries, bp_rp_shortlist, abs_magn_primaries, abs_magn_secondaries, abs_magn_shortlist, id)

    tpf = search_targetpixelfile(id, mission = name, author='SPOC')
    ### find different sections for object ###
    wanted = list(tpf.table["sequence_number"][:5])
    tpfs = download_sections(tpf, wanted)
    cleaned_tpfs = []

    ### download and clean lightcurves up ###
    #for i,t in enumerate(tpfs):
    #for i in range(len(tpfs)):
    t = tpfs[0]
    lightcurve = t.to_lightcurve(aperture_mask=t.pipeline_mask)
    lc_clean = clean_tpf(lightcurve)
    cleaned_tpfs.append(lc_clean)
    #if(i==1):
    axs[1].scatter(lc_clean.time.value, lc_clean.flux.value, color ='black', s=2)
    axs[1].set_xlabel('Time [d]')
    axs[1].set_ylabel(r'Flux [e $s^{-1}$]')
    axs[1].set_title("Lightcurve")
    axs[1].grid(True)
            

    coll = lk.LightCurveCollection(cleaned_tpfs)
    lc_stitched = coll.stitch()

    lc_detrend,_ = detrend(lc_stitched,window_length=1.5)
    try:
        pg = lc_detrend.to_periodogram(method='lombscargle', minimum_frequency=4, maximum_frequency=40)
    except:
        pg = lc_detrend.to_periodogram(method='lombscargle', minimum_frequency=0.5, maximum_frequency=20)
    period_at_max_power = pg.period_at_max_power
    axs[2].plot(pg.frequency, pg.power, color = 'black')
    axs[2].set_xlabel("Frequency [1/d]")
    axs[2].set_ylabel("Power")
    axs[2].set_title(f"Period at max power: {period_at_max_power:.3f}")  

    fig.suptitle(f'System {id}', fontsize=14)
    fig.savefig(f"/Users/new/Desktop/{id}_HR_LC_PG.png",dpi=500)
    plt.show()



