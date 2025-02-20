import warnings
import os
import subprocess as sp

import numpy as np
import statsmodels.api as sm

from joblib import Parallel, delayed

from nilearn import image
from nilearn.maskers import NiftiMasker
from nilearn.masking import apply_mask, unmask

from tqdm import tqdm

from compneuro_atrophy_mapping.data import FULL_MNI_152_HOLELESS_BRAIN

# Ignore the RuntimeWarnings to ignore the division by zero raised by numpy
warnings.filterwarnings("ignore", category=RuntimeWarning)


def run_vbm(vbm_directory: str):
    """
    Run VBM on the T1 data.

    Parameters
    ----------
    vbm_directory : str
        Path to the directory containing the T1 data.

    Returns
    -------
    None
    """
    # Get the FSL binary directory
    FSLBIN = os.path.join(os.getenv("FSLDIR"), "bin")

    if not os.path.exists(vbm_directory):
        raise IsADirectoryError(f"[ERROR-VBM]: The provided directory {vbm_directory} does not exist.")

    cmds = [f"cd {vbm_directory} && {FSLBIN}/bin/fslvbm_1_bet -b",
            f"cd {vbm_directory} && {FSLBIN}/bin/fslvbm_2_template -n",
            f"cd {vbm_directory} && {FSLBIN}/bin/fslvbm_3_proc"]
    outputs = []
    for i, cmd in enumerate(cmds):
        outputs.append(f"VBM step {i} logs: {sp.run(cmd, shell=True, capture_output=True)}\n")

    # Print the logs
    print(outputs)

    # Check if the VBM was successful
    if (not os.path.exists(os.path.join(vbm_directory, "GM_mod_merg_s4.nii.gz")) or
        not os.path.exists(os.path.join(vbm_directory, "struc")) or
        not os.path.exists(os.path.join(vbm_directory, "stats"))):
        err_msg = "[ERROR]: VBM failed. Check the logs."
        raise RuntimeError(err_msg)


def fit_single_voxel(voxel_data, design_mat):
    model = sm.OLS(voxel_data, design_mat)
    results = model.fit()
    return (results.params,
            results.tvalues,
            results.pvalues,
            results.rsquared)


def fit_voxelwise_ols_parallel(masked_data, design_matrix, n_jobs=-1):
    n_voxels = masked_data.shape[1]
    n_regressors = design_matrix.shape[1]
    results = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(fit_single_voxel)(
            masked_data[:, voxel],
            design_matrix
        ) for voxel in tqdm(range(n_voxels))
    )

    # Initialize output arrays
    betas = np.zeros((n_voxels, n_regressors))
    t_stats = np.zeros((n_voxels, n_regressors))
    p_values = np.zeros((n_voxels, n_regressors))
    r_squared = np.zeros(n_voxels)
    
    # Unpack results
    for voxel, (params, tvalues, pvalues, rsquared) in enumerate(results):
        betas[voxel, :] = params
        t_stats[voxel, :] = tvalues
        p_values[voxel, :] = pvalues
        r_squared[voxel] = rsquared
    
    return betas, t_stats, p_values, r_squared


def fit_ols(masked_cn_im: np.ndarray, design_mat_cn: np.ndarray):
    # Initialize the maps
    n_voxels = masked_cn_im.shape[1]
    n_regressors = design_mat_cn.shape[1]
    betas = np.zeros([n_voxels, n_regressors])
    t_stats = np.zeros([n_voxels, n_regressors])
    p_values = np.zeros([n_voxels, n_regressors])
    r_squared = np.zeros(n_voxels)
    for voxel in tqdm(range(n_voxels)):
        results = fit_single_voxel(voxel_data = masked_cn_im[:, voxel],
                                   design_mat = design_mat_cn)
        betas[voxel, :] = results[0]
        t_stats[voxel, :] = results[1]
        p_values[voxel, :] = results[2]
        r_squared[voxel] = results[3]

    return betas, t_stats, p_values, r_squared


def compute_wmaps_from_vbm(gm_mod_merg_control: str,
                           gm_mod_merg_studygroup: str,
                           design_matrix_cn: np.ndarray,
                           design_matrix_studygroup: np.ndarray,
                           n_jobs: int=None) -> dict:
    """_summary_

    Parameters
    ----------
    gm_mod_merg_control : str
        _description_
    gm_mod_merg_studygroup : str
        _description_
    design_matrix_cn : np.ndarray
        To fit the normative model.
    design_matrix_studygroup : np.ndarray
        To compute the expected VBM signal from the normative model.

    Returns
    -------
    dict
        Dictionary containing the gm_common_mask, beta, t, p-val, and W-score maps.
    """

    # Read the GM_mod_merg images
    gm_mod_control = image.load_img(gm_mod_merg_control)
    gm_mod_studygroup = image.load_img(gm_mod_merg_studygroup)

    # Read the GM masks
    gm_mask_cn_path = os.path.join(os.path.dirname(gm_mod_merg_control),
                                   "GM_mask.nii.gz")
    gm_mask_cn = image.load_img(gm_mask_cn_path)
    
    gm_mask_studygroup_path = os.path.join(os.path.dirname(gm_mod_merg_studygroup),
                                           "GM_mask.nii.gz")
    gm_mask_studygroup = image.load_img(gm_mask_studygroup_path)

    # Compute the joint mask
    gm_common_mask = image.math_img("np.where((mask1 * mask2) > 0, 1.0, 0.0)",
                                    mask1=gm_mask_cn,
                                    mask2=gm_mask_studygroup)

    # Initialize the masker and fit it to the common mask
    masker_mni = NiftiMasker(mask_img=FULL_MNI_152_HOLELESS_BRAIN).fit()

    # Apply the joint mask
    vbm_response_cn_masked = masker_mni.transform(gm_mod_control)
    vbm_response_studygroup_masked = masker_mni.transform(gm_mod_studygroup)

    # Unmask to operate in image form if needed
    gm_mod_control = masker_mni.inverse_transform(vbm_response_cn_masked)
    gm_mod_studygroup = masker_mni.inverse_transform(vbm_response_studygroup_masked)

    # Fit the OLS model voxelwise
    print("[INFO] Fitting the voxelwise OLS model.")
    if n_jobs == 1:  # Single job
        betas, t_stats, p_values, r_squared = fit_ols(vbm_response_cn_masked,
                                                      design_matrix_cn)
    elif n_jobs < 1 and n_jobs != -1:  # Throw error because n_jobs must be higher or equal to 1
        raise ValueError("[ERROR]: n_jobs must be higher than 0 in compute_wmaps_from_vbm.")
    else:  # n_jobs > 1
        betas, t_stats, p_values, r_squared = fit_voxelwise_ols_parallel(vbm_response_cn_masked,
                                                                         design_matrix_cn,
                                                                         n_jobs=n_jobs)

    # Unmask the statistical maps
    print("[INFO] Finished fitting voxelwise OLS model, check the R-square map to see the goodness of fit.")
    beta_maps = masker_mni.inverse_transform(betas.T)
    t_maps = masker_mni.inverse_transform(t_stats.T)
    p_values_maps = masker_mni.inverse_transform(p_values.T)
    r_squared_map = masker_mni.inverse_transform(r_squared)

    # Compute residuals
    predicted = design_matrix_cn @ betas.T
    residuals = vbm_response_cn_masked - predicted
    residuals_im = masker_mni.inverse_transform(residuals)
    sd_of_residuals = image.math_img("np.std(a, axis=3)", a=residuals_im)

    # Compute the W-score maps
    wmaps = []
    predicted_studygroup = design_matrix_studygroup @ betas.T
    gm_predicted_studygroup_im = masker_mni.inverse_transform(predicted_studygroup)
    for i in range(gm_mod_studygroup.shape[3]):
        pred = image.index_img(gm_predicted_studygroup_im, i)
        obsr = image.index_img(gm_mod_studygroup, i)
        wmap = image.math_img("(a - b) / c",
                              a=obsr,
                              b=pred,
                              c=sd_of_residuals)
        wmaps.append(wmap)

    # Concatenate the wmaps
    wmaps = image.concat_imgs(wmaps)

    # Remove NaN values from wmaps
    wmaps = image.math_img("np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)",
                           a=wmaps)
    print("[INFO] Finished computing the W-score maps.")

    # Organize the results
    results = {"betamaps": beta_maps,
                "tmaps": t_maps,
                "pvalues": p_values_maps,
                "wmaps": wmaps,
                "sd_of_residuals": sd_of_residuals,
                "r_squared": r_squared_map}

    # Apply the common subject mask to the outputs before saving (only GM results wanted)
    results = {key: unmask(apply_mask(results[key], gm_common_mask), gm_common_mask) for key in results.keys()}
    # Include the common mask
    results["gm_common_mask"] = gm_common_mask

    return results


def compute_wmaps_from_model():
    # TODO: Design how to reuse an already fit model to compute Wmaps on new data.
    # NOTE: It is not as trivial as reusing the betas. Masking and unmasking must be handled with care.
    pass