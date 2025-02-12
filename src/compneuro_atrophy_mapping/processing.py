import os
import subprocess as sp

import numpy as np
import statsmodels.api as sm

from nilearn import image
from nilearn.maskers import NiftiMasker

from tqdm import tqdm


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
    if not os.path.exists(vbm_directory):
        raise IsADirectoryError(f"[ERROR-VBM]: The provided directory {vbm_directory} does not exist.")

    cmds = [f"cd {vbm_directory} && $FSLDIR/bin/$FSLDIR/bin/fslvbm_1_bet -b",
            f"cd {vbm_directory} && $FSLDIR/bin/$FSLDIR/bin/fslvbm_2_template -n",
            f"cd {vbm_directory} && $FSLDIR/bin/$FSLDIR/bin/fslvbm_3_proc"]
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


def compute_wmaps_from_vbm(gm_mod_merg_control: str,
                           gm_mod_merg_studygroup: str,
                           design_matrix_cn: np.ndarray,
                           design_matrix_studygroup: np.ndarray) -> dict:
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

    # Initialize the masker
    masker = NiftiMasker(mask_img=gm_common_mask).fit()

    # Apply the joint mask
    vbm_response_cn_masked = masker.transform(gm_mod_control)
    vbm_response_studygroup_masked = masker.transform(gm_mod_studygroup)

    # Unmask to operate in image form if needed
    gm_mod_control = masker.inverse_transform(vbm_response_cn_masked)
    gm_mod_studygroup = masker.inverse_transform(vbm_response_studygroup_masked)

    # Initialize the maps
    n_voxels = vbm_response_cn_masked.shape[1]
    n_regressors = design_matrix_cn.shape[1]
    betas = np.zeros([n_voxels, n_regressors])
    t_stats = np.zeros([n_voxels, n_regressors])
    p_values = np.zeros([n_voxels, n_regressors])

    # Fit the OLS model voxelwise
    print("[INFO]: Fitting the voxelwise OLS model.")
    for voxel in tqdm(range(n_voxels)):
        y = vbm_response_cn_masked[:, voxel]
        model = sm.OLS(y, design_matrix_cn)
        results = model.fit()
    
        betas[voxel, :] = results.params
        t_stats[voxel, :] = results.tvalues
        p_values[voxel, :] = results.pvalues

    # Unmask the statistical maps
    beta_maps = masker.inverse_transform(betas.T)
    t_maps = masker.inverse_transform(t_stats.T)
    p_values_maps = masker.inverse_transform(p_values.T)

    # Compute residuals
    predicted = design_matrix_cn @ betas.T
    residuals = vbm_response_cn_masked - predicted
    residuals_im = masker.inverse_transform(residuals)
    sd_of_residuals = image.math_img("np.std(a, axis=3)", a=residuals_im)

    # Compute the W-score maps
    wmaps = []
    predicted_studygroup = design_matrix_studygroup @ betas.T
    gm_predicted_studygroup_im = masker.inverse_transform(predicted_studygroup)
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

    # Return the results
    results = {"betamaps": beta_maps,
                "tmaps": t_maps,
                "pvalues": p_values_maps,
                "wmaps": wmaps,
                "gm_common_mask": gm_common_mask,
                "masker": masker,
                "sd_of_residuals": sd_of_residuals}

    return results


def compute_wmaps_from_model():
    # TODO: Design how to reuse an already fit model to compute Wmaps on new data.
    # NOTE: It is not as trivial as reusing the betas. Masking and unmasking must be handled with care.
    pass