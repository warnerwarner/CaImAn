import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import tifffile
import scipy.ndimage
import scipy.signal


import caiman
from caiman.source_extraction.volpy.volparams import volparams
from caiman.source_extraction.volpy.volpy import VOLPY

# %%
def cut_and_interpolate(file_path, cut_size=110, filter_freqs=None, fs=440):
    data = []
    with tifffile.TiffFile(file_path) as tif:
        for page in tif.pages:
            data.append(page.asarray())
    data = np.array(data)
    assert np.min(data[0]) == np.max(data[0]), "First frame is not empty, it has already been cut"
    d_cut = data[cut_size:-cut_size]
    med_val = np.median(d_cut.mean(axis=0))
    std_val = np.std(d_cut.mean(axis=0))
    y_coords, x_coords = np.indices(d_cut.shape[1:])
    zero_mask = d_cut.mean(axis=0) < med_val - std_val
    valid = ~zero_mask
    valid_yx = np.column_stack((y_coords[valid], x_coords[valid]))
    missing_yx = np.column_stack((y_coords[zero_mask], x_coords[zero_mask]))

    valid_values = d_cut[:, valid]
    tree = scipy.spatial.cKDTree(valid_yx)
    _, indices = tree.query(missing_yx)
    interpolated_values = valid_values[:, indices]
    d_cut[:, zero_mask] = interpolated_values
    if filter_freqs is not None:
        print(f"Applying notch filters at frequencies: {filter_freqs} Hz")
        if isinstance(filter_freqs, (int, float)):
            filter_freqs = [filter_freqs]
        for f in filter_freqs:
            b, a = scipy.signal.iirnotch(f, Q=30, fs=fs)
            d_cut = scipy.signal.filtfilt(b, a, d_cut, axis=0)
    tifffile.imwrite(file_path, d_cut)

files_2_memmap = []
for file in os.listdir("/Volumes/tomdrive/voltage_imaging/denoised_tiffs/run016"):
    if 'splatted_aligned_to_session' in file and file[0] != '.':
        try:
            cut_and_interpolate(os.path.join("/Volumes/tomdrive/voltage_imaging/denoised_tiffs/run016", file), filter_freqs=[60, 120])
            print(f"{file} processed and ready for memmap.")
        except AssertionError as e:
            files_2_memmap.append(os.path.join("/Volumes/tomdrive/voltage_imaging/denoised_tiffs/run016", file))
            print(f"{file} skipped: {e}")
# %%
data = []
with tifffile.TiffFile(
    "/Volumes/tomdrive/voltage_imaging/denoised_tiffs/run016/denoised_patchSize_221_100_100_lossCoef_0.5_0.5_trainingSize_10_bsSize_3_3_splatted_aligned_to_session_00077.tif"
) as tif:
    for page in tif.pages:
        data.append(page.asarray())
data = np.array(data)
d_cut = data[110:-110]
med_val = np.median(d_cut.mean(axis=0))
std_val = np.std(d_cut.mean(axis=0))
y_coords, x_coords = np.indices(d_cut.shape[1:])
zero_mask = d_cut.mean(axis=0) < med_val - std_val
valid = ~zero_mask
valid_yx = np.column_stack((y_coords[valid], x_coords[valid]))
missing_yx = np.column_stack((y_coords[zero_mask], x_coords[zero_mask]))

valid_values = d_cut[:, valid]
tree = scipy.spatial.cKDTree(valid_yx)
_, indices = tree.query(missing_yx)
interpolated_values = valid_values[:, indices]
d_cut[:, zero_mask] = interpolated_values
# %%
print(np.min(data[0]) == np.max(data[0]))
# %%

roistack = tifffile.imread(
    "/Volumes/tomdrive/voltage_imaging/12-12-25/run016/roistack.tif"
)

for i in range(roistack.shape[0]):
    roistack[i] = scipy.ndimage.gaussian_filter(roistack[i], sigma=1)


# caiman.stop_server(dview=dview)
c, dview, n_processes = caiman.cluster.setup_cluster(
    backend="multiprocessing", n_processes=None, single_thread=True
)
print("Cluster set up with %d processes" % n_processes)

memmapped = caiman.save_memmap(files_2_memmap, base_name="memmap_", dview=dview, order="C")
# %% Shrinking the ROIs
def shrink_roi(roi, shrink_size=2):

    if np.sum(roi) == 0:
        return roi  # Return the original ROI if it's empty
    structure = np.ones((3, 3), dtype=bool)  # Define a 3x3 structuring element
    shrunk_roi = scipy.ndimage.binary_erosion(roi, structure=structure, iterations=shrink_size)
    return shrunk_roi.astype(roi.dtype)

shrunk_ROIs = []
for i in range(roistack.shape[0]):
    shrunk_ROIs.append(roistack[i].astype(np.bool) ^ shrink_roi(roistack[i].astype(np.bool), shrink_size=4))
for i in range(len(shrunk_ROIs)):
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(roistack[i].astype(np.bool), cmap='gray')
    ax[1].imshow(shrunk_ROIs[i], cmap='gray')
# %%

# %%
ROIs = np.array(shrunk_ROIs)  # region of interests
index = list(range(len(ROIs)))  # index of neurons
weights = None  # if None, use ROIs for initialization; to reuse weights check reuse weights block

fr = 440.0  # frame rate
template_size = (
    0.02  # half size of the window length for spike templates, default is 20 ms
)
context_size = (
    8  # number of pixels surrounding the ROI to censor from the background PCA
)
visualize_ROI = (
    False  # whether to visualize the region of interest inside the context region
)
flip_signal = False  # Important!! Flip signal or not, True for Voltron indicator, False for others
hp_freq_pb = 1 / 3  # parameter for high-pass filter to remove photobleaching
clip = 20  # maximum number of spikes to form spike template
threshold_method = "adaptive_threshold"  # adaptive_threshold or simple
min_spikes = 10  # minimal spikes to be found
pnorm = 0.25  # a variable deciding the amount of spikes chosen for adaptive threshold method
threshold = 2  # threshold for finding spikes only used in simple threshold method, Increase the threshold to find less spikes
do_plot = False  # plot detail of spikes, template for the last iteration
ridge_bg = 0.06  # ridge regression regularizer strength for background removal, larger value specifies stronger regularization
sub_freq = 20  # frequency for subthreshold extraction
weight_update = "NMF"  # ridge or NMF for weight update
n_iter = 5  # number of iterations alternating between estimating spike times and spatial filters
opts_dict = {
    "fnames": memmapped,
    "fr": fr,
    "ROIs": ROIs,
    "index": index,
    "weights": weights,
    "template_size": template_size,
    "context_size": context_size,
    "visualize_ROI": visualize_ROI,
    "flip_signal": flip_signal,
    "hp_freq_pb": hp_freq_pb,
    "clip": clip,
    "threshold_method": threshold_method,
    "min_spikes": min_spikes,
    "pnorm": pnorm,
    "threshold": threshold,
    "do_plot": do_plot,
    "ridge_bg": ridge_bg,
    "sub_freq": sub_freq,
    "weight_update": weight_update,
    "n_iter": n_iter,
    "nPC_bg": 4
}

opts = volparams(params_dict=opts_dict)
# %%
print(dir(opts))
# %%
vpy = VOLPY(n_processes=n_processes, dview=dview, params=opts)
vpy.fit(n_processes=n_processes, dview=dview)
# %%
vpy.estimates.keys()
# %%
idx = np.where(vpy.estimates["locality"] > 0)[
    0
]  # Select only neurons inside spatial priors
# %%
idx
# %%
# # Reconstructed movie
# mv_all = utils.reconstructed_movie(
#     vpy.estimates.copy(),
#     fnames=memmapped,
#     idx=idx,
#     scope=(0, arr.shape[0]),
#     flip_signal=flip_signal
# )

# mv_all.play(fr=30, magnification=1)
# mv_all.save(f"{FOLDER}/reconstructed_volpy.avi")

caiman.stop_server(dview=dview)

vpy.estimates.keys()
# %%
plt.plot(vpy.estimates["templates"][7])
# %%
len(vpy.estimates["spikes"])
for i in range(len(vpy.estimates["spikes"])):
    print(i, len(vpy.estimates["spikes"][i]))
    if len(vpy.estimates["spikes"][i]) > 0:
        spike_templates = vpy.estimates["templates"][i]
        print(spike_templates.shape)
        time = np.arange(spike_templates.shape[0]) / fr * 1000 - template_size * 1000

        plt.plot(time, spike_templates.T, alpha=0.25, color="black")
        plt.xticks(np.arange(-template_size * 1000, template_size * 1000 + 1, 5))
        plt.xlabel("ms")
# vpy.estimates['spikes'][2]
# %%
plt.figure("Spike templates", figsize=(3, 2))
# Set font to Arial
plt.rcParams["font.family"] = "Arial"
spike_templates = vpy.estimates["templates"][idx]
time = (
    np.arange(spike_templates.shape[1]) / fr * 1000 - template_size * 1000
)  # convert to ms
print(spike_templates.shape)
plt.plot(time, spike_templates.T, alpha=0.25, color="black")
avg_template = spike_templates.mean(axis=0)
plt.plot(time, avg_template, color="red")
plt.xticks(np.arange(-template_size * 1000, template_size * 1000 + 1, 5))
plt.xlabel("ms")
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["left"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
# %%
for i in vpy.estimates.keys():
    print(i)
# %%

for i in range(len(vpy.estimates["spikes"])):
    if len(vpy.estimates["spikes"][i]) > 0:
        plt.imshow(vpy.estimates["weights"][i], cmap="gray")
        plt.title(f"{i}, {vpy.estimates["locality"][i]}")
        plt.show()
# %%
print(vpy.estimates["locality"])
# %%
plt.plot(vpy.estimates['t'][5])
plt.plot(vpy.estimates['ts'][5])
plt.plot(vpy.estimates['t_rec'][5])
plt.plot(vpy.estimates['t_sub'][5])
plt.scatter(vpy.estimates['spikes'][5], np.ones(len(vpy.estimates['spikes'][5])), color='red', zorder=10, s=1)    
# %%
plt.plot(vpy.estimates['F0'][5])
ax2 = plt.twinx()
ax2.plot(vpy.estimates['dFF'][5])
# %%
lens = []
for file in files_2_memmap:
    with tifffile.TiffFile(file) as tif:
        lens.append(len(tif.pages))

lens = np.cumsum(lens)
# %%
roi_index =13
from scipy.ndimage import percentile_filter
denoised_roi_smooth = percentile_filter(
    vpy.estimates['ts'][roi_index], percentile=70, size=10, mode="nearest"
)
plt.plot(denoised_roi_smooth)
plt.scatter(vpy.estimates['spikes'][roi_index], np.ones(len(vpy.estimates['spikes'][roi_index])), color='red', zorder=10, s=1)
# %% looking at the raw signal
indexes = []

for file in files_2_memmap:
    indexes.append(file[-9:-4])


    
index = indexes[0]
raw_data_file = f"/Volumes/tomdrive/voltage_imaging/12-12-25/run016/vis-stim-force1s-FOV3_440Hz_8X_2p5x1SAM_0p71t-0p9s-FF_0p25ms-fb_bin1_filter_on_EOD-off_{index}_splatted_aligned_to_session.tif"
cut_size = 110
filter_freqs = [60, 120]  # frequencies to apply notch filter, set to None to skip filtering
fs = 440

data = []

with tifffile.TiffFile(raw_data_file) as tif:
    for page in tif.pages:
        data.append(page.asarray())
data = np.array(data)
d_cut = data[cut_size:-cut_size]
med_val = np.median(d_cut.mean(axis=0))
std_val = np.std(d_cut.mean(axis=0))
y_coords, x_coords = np.indices(d_cut.shape[1:])
zero_mask = d_cut.mean(axis=0) < med_val - std_val
valid = ~zero_mask
valid_yx = np.column_stack((y_coords[valid], x_coords[valid]))
missing_yx = np.column_stack((y_coords[zero_mask], x_coords[zero_mask]))

valid_values = d_cut[:, valid]
tree = scipy.spatial.cKDTree(valid_yx)
_, indices = tree.query(missing_yx)
interpolated_values = valid_values[:, indices]
d_cut[:, zero_mask] = interpolated_values
if filter_freqs is not None:
    print(f"Applying notch filters at frequencies: {filter_freqs} Hz")
    if isinstance(filter_freqs, (int, float)):
        filter_freqs = [filter_freqs]
    for f in filter_freqs:
        b, a = scipy.signal.iirnotch(f, Q=30, fs=fs)
        d_cut = scipy.signal.filtfilt(b, a, d_cut, axis=0)
# %%
c, dview, n_processes = caiman.cluster.setup_cluster(
    backend="multiprocessing", n_processes=None, single_thread=True
)
print("Cluster set up with %d processes" % n_processes)

memmapped = caiman.save_memmap([d_cut], base_name="memmap_", dview=dview, order="C")
# %%

opts.set("data", {'weights':vpy.estimates['weights'], 'fnames':memmapped})
opts.set("volspike", {"n_iter": 1})
vpy_raw = VOLPY(n_processes=n_processes, dview=dview, params=opts)
vpy_raw.fit(n_processes=n_processes, dview=dview)
# %%

caiman.stop_server(dview=dview)
print(vpy.estimates['weights'].shape)
# %%
print(vpy_raw.estimates['rawROI'][0].keys())

# %%
plt.plot(vpy_raw.estimates['rawROI'][10]['ts'])
plt.plot(vpy.estimates['ts'][10][:2000])
for i in vpy_raw.estimates['rawROI'][10]['spikes']:
    plt.axvline(i, color='red', alpha=0.5, zorder=-10)
for i in vpy.estimates['spikes'][10]:
    if i < 2000:
        plt.axvline(i, color='green', alpha=0.5, zorder=-10)
