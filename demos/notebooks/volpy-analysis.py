# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import tifffile
from tqdm import tqdm
import scipy.spatial


import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.utils.utils import download_demo, download_model
from caiman.source_extraction.volpy import utils
from caiman.source_extraction.volpy.volparams import volparams
from caiman.source_extraction.volpy.volpy import VOLPY
from caiman.summary_images import local_correlations_movie_offline, mean_image
from caiman.paths import caiman_datadir

# %%
data = []
with tifffile.TiffFile(
    "/Volumes/tomdrive/voltage_imaging/denoised_tiffs/run016/denoised_patchSize_221_100_100_lossCoef_0.5_0.5_trainingSize_5_bsSize_3_3_splatted_00011.tif"
) as tif:
    for page in tif.pages:
        data.append(page.asarray())
data = np.array(data)
print(data.shape)
# %% interpolating pixels with 0 value
# plt.hist(data[111:-111].flatten(), bins=100)
d_cut = data[110:-110]
med_val = np.median(d_cut.mean(axis=0))
std_val = np.std(d_cut.mean(axis=0))
print("cut data")
# %%
plt.imshow(d_cut.mean(axis=0))
plt.figure()
plt.hist(d_cut.mean(axis=0).flatten(), bins=100)
plt.axvline(med_val)
plt.axvline(med_val - std_val)
# %%
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
plt.imshow(d_cut[6])
plt.colorbar()
# %%
tifffile.imwrite(
    "/Volumes/tomdrive/voltage_imaging/denoised_tiffs/run016/denoised_patchSize_221_100_100_lossCoef_0.5_0.5_trainingSize_5_bsSize_3_3_splatted_00011_interpolated.tif",
    d_cut,
)
# %%
fnames = [
    "/Volumes/tomdrive/voltage_imaging/denoised_tiffs/run016/denoised_patchSize_221_100_100_lossCoef_0.5_0.5_trainingSize_5_bsSize_3_3_splatted_00011_interpolated.tif"
]
# Setup some parameters for data and motion correction dataset parameters
fr = 440  # sample rate of the movie
ROIs = None  # Region of interests
index = None  # index of neurons
weights = None  # reuse spatial weights by
# opts.change_params(params_dict={'weights':vpy.estimates['weights']})
# Motion correction parameters
pw_rigid = False  # flag for pw-rigid motion correction
gSig_filt = (3, 3)  # size of filter, in general gSig (see below),
# change this one if algorithm does not work
max_shifts = (5, 5)  # maximum allowed rigid shift
strides = (48, 48)  # start a new patch for pw-rigid motion correction every x pixels
overlaps = (24, 24)  # overlap between patches (size of patch strides+overlaps)
max_deviation_rigid = (
    3  # maximum deviation allowed for patch with respect to rigid shifts
)
border_nan = "copy"

opts_dict = {
    "fnames": fnames,
    "fr": fr,
    "index": index,
    "ROIs": ROIs,
    "weights": weights,
    "pw_rigid": pw_rigid,
    "max_shifts": max_shifts,
    "gSig_filt": gSig_filt,
    "strides": strides,
    "overlaps": overlaps,
    "max_deviation_rigid": max_deviation_rigid,
    "border_nan": border_nan,
}

opts = volparams(params_dict=opts_dict)
# %%

c, dview, n_processes = cm.cluster.setup_cluster(
    backend="multiprocessing", n_processes=None, single_thread=False
)
# %%

mc = MotionCorrect(fnames, dview=dview, **opts.get_group("motion"))
mc.motion_correct(save_movie=True)
dview.terminate()
# %%

m_orig = cm.load(fnames)
m_rig = cm.load(mc.mmap_file)
m_orig.fr = 440
m_rig.fr = 440
ds_ratio = 1.0
moviehandle = cm.concatenate(
    [
        m_orig.resize(1, 1, ds_ratio) - mc.min_mov * mc.nonneg_movie,
        m_rig.resize(1, 1, ds_ratio),
    ],
    axis=2,
)
min_, max_ = np.min(moviehandle), np.max(moviehandle)
moviehandle = cm.movie((moviehandle - min_) / (max_ - min_) * 255, dtype="uint8")
print(moviehandle.shape)
# %%

# Movie subtracted from the baseline
m_rig2 = m_rig.computeDFF(secsWindow=1)[0][:1000]
moviehandle1 = -m_rig2
min_, max_ = np.min(moviehandle1), np.max(moviehandle1)
moviehandle1 = cm.movie((moviehandle1 - min_) / (max_ - min_) * 255, dtype="uint8")
# %%
tifffile.imwrite(
    "/Volumes/tomdrive/voltage_imaging/denoised_tiffs/run016/volpy_mc_comparison_denoised_patchSize_221_100_100_lossCoef_0.5_0.5_trainingSize_5_bsSize_3_3_splatted_00011.tif",
    moviehandle,
)

# %%

c, dview, n_processes = cm.cluster.setup_cluster(
    backend="multiprocessing", n_processes=None, single_thread=False
)
border_to_0 = 0 if mc.border_nan == "copy" else mc.border_to_0
fname_new = cm.save_memmap_join(
    mc.mmap_file, base_name="memmap_", add_to_mov=border_to_0, dview=dview, n_chunks=10
)
dview.terminate()

# Change fnames to the new motion corrected one
opts.change_params(params_dict={"fnames": fname_new})
# %%

if "dview" in locals():
    cm.stop_server(dview=dview)
c, dview, n_processes = cm.cluster.setup_cluster(
    backend="multiprocessing", n_processes=None, single_thread=False
)
# %%

img = mean_image(mc.mmap_file[0], window=1000, dview=dview)
img = (img - np.mean(img)) / np.std(img)

gaussian_blur = False  # Use gaussian blur when there is too much noise in the video
Cn = local_correlations_movie_offline(
    mc.mmap_file[0],
    fr=fr,
    window=fr * 4,
    stride=fr * 4,
    winSize_baseline=fr,
    remove_baseline=True,
    gaussian_blur=gaussian_blur,
    dview=dview,
).max(axis=0)
img_corr = (Cn - np.mean(Cn)) / np.std(Cn)
summary_images = np.stack([img, img, img_corr], axis=0).astype(np.float32)
# Save summary images which could be further used in the VolPy GUI
cm.movie(summary_images).save(fnames[0] + "_summary_images.tif")

fig, axs = plt.subplots(1, 2)
axs[0].imshow(summary_images[0])
axs[1].imshow(summary_images[2])
axs[0].set_title("mean image")
axs[1].set_title("corr image")


# %%

weights_path = download_model("mask_rcnn")
ROIs = utils.mrcnn_inference_pytorch(
    img=summary_images.transpose([1, 2, 0]),
    size_range=[5, 22],
    weights_path=weights_path,
    display_result=True,
)  # size parameter decides size range of masks to be selected
cm.movie(ROIs).save(fnames[0] + "_mrcnn_ROIs.hdf5")
# %%
roistack = tifffile.imread(
    "/Volumes/tomdrive/voltage_imaging/12-12-25/run016/roistack.tif"
)

for i in range(roistack.shape[0]):
    roistack[i] = scipy.ndimage.gaussian_filter(roistack[i], sigma=1)

plt.figure("roistack")
plt.imshow(roistack.sum(axis=0), cmap="gray")
plt.axis("off")
ROIs = roistack
# %%

fig, axs = plt.subplots(1, 2)
axs[0].imshow(summary_images[0])
axs[1].imshow(ROIs.sum(0))
axs[0].set_title("mean image")
axs[1].set_title("masks")

# %%

cm.stop_server(dview=dview)
c, dview, n_processes = cm.cluster.setup_cluster(
    backend="multiprocessing", n_processes=None, single_thread=False, maxtasksperchild=1
)

# %%

ROIs = ROIs  # region of interests
index = list(range(len(ROIs)))  # index of neurons
weights = None  # if None, use ROIs for initialization; to reuse weights check reuse weights block

template_size = (
    0.2  # half size of the window length for spike templates, default is 20 ms
)
context_size = (
    2  # number of pixels surrounding the ROI to censor from the background PCA
)
censor_size = 0
visualize_ROI = (
    False  # whether to visualize the region of interest inside the context region
)
flip_signal = False  # Important!! Flip signal or not, True for Voltron indicator, False for others
hp_freq_pb = 1 / 3  # parameter for high-pass filter to remove photobleaching
clip = 64  # maximum number of spikes to form spike template
threshold_method = "adaptive_threshold"  # adaptive_threshold or simple
min_spikes = 1  # minimal spikes to be found
pnorm = (
    0.5  # a variable deciding the amount of spikes chosen for adaptive threshold method
)
threshold = 1  # threshold for finding spikes only used in simple threshold method, Increase the threshold to find less spikes
do_plot = False  # plot detail of spikes, template for the last iteration
ridge_bg = 0.01  # ridge regression regularizer strength for background removement, larger value specifies stronger regularization
sub_freq = 20  # frequency for subthreshold extraction
weight_update = "NMF"  # ridge or NMF for weight update
n_iter = 5  # number of iterations alternating between estimating spike times and spatial filters

opts_dict = {
    "fnames": fname_new,
    "fr": fr,
    "ROIs": ROIs[:5],
    "index": index[:5],
    "weights": weights,
    "template_size": template_size,
    "context_size": context_size,
    "censor_size": censor_size,
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
}

opts.change_params(params_dict=opts_dict)
# %%
index
# %%

vpy = VOLPY(n_processes=n_processes, dview=dview, params=opts)
vpy.fit(n_processes=n_processes, dview=dview)
# %%

print(np.where(vpy.estimates["locality"])[0])  # neurons that pass locality test
idx = np.where(vpy.estimates["locality"] > 0)[0]
utils.view_components(vpy.estimates, d_cut.mean(axis=0), [idx[10]])
# %%
print(idx)

# %%

plt.figure("ROIs", figsize=(6, 14))
for i, roi_i in enumerate(idx):
    plt.subplot(len(idx) // 3 + 1, 3, i + 1)
    plt.imshow(vpy.estimates["weights"][roi_i], cmap="gray")
    plt.axis("off")
plt.tight_layout()

plt.figure("ROIs vs mean image", figsize=(10, 3))
plt.subplot(1, 2, 1)
plt.imshow(np.array(vpy.estimates["weights"]).max(axis=0), cmap="gray")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.axis("off")
