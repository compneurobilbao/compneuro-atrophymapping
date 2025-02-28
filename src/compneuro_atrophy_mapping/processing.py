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


def compute_wmaps_from_vbm(gm_mod_merg_path: str,
                           design_matrix: np.ndarray,
                           groups_list: list,
                           n_jobs: int=1) -> dict:
    """_summary_

    Parameters
    ----------
    gm_mod_merg : str
        Path to the GM_mod_merg_sX.nii.gz file. Both study groups must be merged in the same file.
    design_matrix : np.ndarray
        Design matrix for the OLS model. It MUST be in the same order as the GM_mod_merg file.
    groups_list : list
        A list containing which rows of the design matrix correspond to each group. It is a list of
        1s and 0s, where 1s indicate the rows corresponding to the control group.
    n_jobs : int, optional
        Number of jobs to run in parallel, by default 1

    Returns
    -------
    dict
        Dictionary containing the gm_common_mask, beta, t, p-val, and W-score maps.

    Raises
    ------
    ValueError
        When n_jobs is less than 1 and not -1 (use all available cores).
    """
    if n_jobs < 1 and n_jobs != -1:  # Throw error because n_jobs must be higher or equal to 1
        raise ValueError("[ERROR]: n_jobs must be higher than 0 in compute_wmaps_from_vbm.")

    # Read the GM_mod_merg image (contains controls and studygroup)
    gm_mod = image.load_img(gm_mod_merg_path)

    # Read the GM mask
    gm_mask_path = os.path.join(os.path.dirname(gm_mod_merg_path), "GM_mask.nii.gz")
    gm_mask = image.load_img(gm_mask_path)

    # Initialize a masker to fit to the GM mask
    masker_gm = NiftiMasker(mask_img=gm_mask).fit()

    # Get the indices of the images belonging to each clinical group from the groups list
    control_indices = [i for i, group in enumerate(groups_list) if group == 1]
    studygroup_indices = [i for i, group in enumerate(groups_list) if group == 0]
    # Index the images according to the indices. Getting the images for each group
    gm_mod_control = image.index_img(gm_mod, control_indices)
    gm_mod_studygroup = image.index_img(gm_mod, studygroup_indices)
    # Mask the control VBM image to the GM mask (convert to np.array)
    gm_mod_control_masked = masker_gm.transform(gm_mod_control)

    # Index the design matrix according to the indices
    design_matrix_cn = design_matrix[control_indices, :]
    design_matrix_studygroup = design_matrix[studygroup_indices, :]

    # Fit the OLS model voxelwise
    print("[INFO] Fitting the voxelwise OLS model.")
    if n_jobs == 1:  # Single job
        betas, t_stats, p_values, r_squared = fit_ols(gm_mod_control_masked, design_matrix_cn)
    else:  # n_jobs > 1
        betas, t_stats, p_values, r_squared = fit_voxelwise_ols_parallel(gm_mod_control_masked,
                                                                         design_matrix_cn,
                                                                         n_jobs=n_jobs)

    # Unmask the statistical maps
    print("[INFO] Finished fitting voxelwise OLS model, check the R-square map to see the goodness of fit.")
    beta_maps = masker_gm.inverse_transform(betas.T)
    t_maps = masker_gm.inverse_transform(t_stats.T)
    p_values_maps = masker_gm.inverse_transform(p_values.T)
    r_squared_map = masker_gm.inverse_transform(r_squared)

    # Compute residuals and their standard deviation in the 4th dimension
    predicted = design_matrix_cn @ betas.T
    residuals = gm_mod_control_masked - predicted
    sd_of_residuals = np.std(residuals, axis=0)
    sd_of_residuals_im = masker_gm.inverse_transform(sd_of_residuals)

    # Compute the W-score maps. Image indexing is slow, but result is lighter than the alternative.
    print("[INFO] Computing the W-score maps.")
    wmaps = []
    predicted_studygroup = design_matrix_studygroup @ betas.T
    gm_predicted_studygroup_im = masker_gm.inverse_transform(predicted_studygroup)
    for i in range(gm_mod_studygroup.shape[3]):
        pred = image.index_img(gm_predicted_studygroup_im, i)
        obsr = image.index_img(gm_mod_studygroup, i)
        wmap = image.math_img("(a - b) / c",
                              a=obsr,
                              b=pred,
                              c=sd_of_residuals_im)
        wmaps.append(wmap)

    # Alternative wmap computation.
    # Same result, but faster since it uses the masker.
    # However, the filesize is around 2x bigger.
    # obsr = masker_gm.transform(gm_mod_studygroup)
    # wmaps_alt = (obsr - predicted_studygroup) / sd_of_residuals
    # wmaps_alt = masker_gm.inverse_transform(wmaps_alt)

    # Concatenate the wmaps
    wmaps = image.concat_imgs(wmaps)

    # Remove NaN values from wmaps
    wmaps = image.math_img("np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)",
                           a=wmaps)
    # wmaps_alt = image.math_img("np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)",
    #                        a=wmaps_alt)
    print("[INFO] Finished computing the W-score maps.")

    # Organize the results
    results = {"betamaps": beta_maps,
                "tmaps": t_maps,
                "pvalues": p_values_maps,
                "wmaps": wmaps,
                # "wmaps_alt": wmaps_alt,
                "sd_of_residuals": sd_of_residuals_im,
                "r_squared": r_squared_map}

    # Apply the common subject mask to the outputs before saving (only GM results wanted)
    results = {key: unmask(apply_mask(results[key], gm_mask), gm_mask) for key in results.keys()}
    # Include the common mask
    results["gm_mask"] = gm_mask

    return results


def compute_wmaps_from_model():
    # TODO: Design how to reuse an already fit model to compute Wmaps on new data.
    # NOTE: It is not as trivial as reusing the betas. Masking and unmasking must be handled with care.
    pass