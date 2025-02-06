import os
import subprocess as sp

import numpy as np
import nibabel as nib

from nilearn import image
from nilearn.maskers import NiftiMasker
from nilearn.glm.first_level import FirstLevelModel
from nilearn.masking import apply_mask, unmask


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
        outputs.append(f"VBM step {i} logs: {sp.run(cmd, shell=True, capture_output=True)}")

    # Print the logs
    print(outputs)

    # Check if the VBM was successful
    if (not os.path.exists(os.path.join(vbm_directory, "GM_mod_merg_s4.nii.gz")) or
        not os.path.exists(os.path.join(vbm_directory, "struc")) or
        not os.path.exists(os.path.join(vbm_directory, "stats"))):
        err_msg = "[ERROR]: VBM failed. Check the logs."
        raise RuntimeError(err_msg)


def compute_atrophy_wmaps(gm_mod_merg_control: str,
                          gm_mod_merg_studygroup: str,
                          design_matrix_cn: np.ndarray,
                          design_matrix_studygroup: np.ndarray,
                          output_dir: str) -> nib.nifti1.Nifti1Image:
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
    output_dir : str
        _description_

    Returns
    -------
    nib.nifti1.Nifti1Image
        _description_
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
    gm_common_mask = image.math_img("np.where((mask1 + mask2) > 0, 1.0, 0.0)",
                                    mask1=gm_mask_cn,
                                    mask2=gm_mask_studygroup)
    # Initialize the masker
    masker = NiftiMasker(mask_img=gm_common_mask).fit()

    # Apply the joint mask
    vbm_response_cn = unmask(apply_mask(gm_mod_control, gm_common_mask),
                                     gm_common_mask)

    vbm_response_studygroup_masked = masker.transform(gm_mod_studygroup)

    # Define and fit the FirstLevelModel
    flm = FirstLevelModel(mask_img=gm_common_mask,
                          noise_model="ols",
                          standardize=False,
                          minimize_memory=False)
    flm.fit(vbm_response_cn, design_matrices=design_matrix_cn)

    # Build contrast (identity of design matrix) and get the beta maps and the residuals
    design_contrast = np.eye(design_matrix_cn.shape[1])
    beta_maps = flm.compute_contrast(contrast_def=design_contrast,
                                     output_type="effect_size")
    beta_maps_masked = masker.transform(beta_maps)
    flm_residuals = flm.residuals[0]

    # Get the residuals from the FirstLevelModel, compute their standard deviation
    sd_residuals = image.math_img("np.std(residuals, axis=3)", residuals=flm_residuals)
    sd_residuals_masked = masker.transform(sd_residuals)

    # Predict the VBM signal from the normative model in the studygroup from its design matrix
    # (y = X @ Beta)
    predicted_vbm_studygroup = design_matrix_studygroup.to_numpy() @ beta_maps_masked

    # Compute the wmaps
    original_minus_predicted = vbm_response_studygroup_masked - predicted_vbm_studygroup
    wmaps = original_minus_predicted / sd_residuals_masked

    # Remove the infinite values
    wmaps[wmaps > 1e3] = 0
    wmaps = masker.inverse_transform(wmaps)

    # Save (original - predicted)
    original_minus_predicted = masker.inverse_transform(original_minus_predicted)
    original_minus_predicted.to_filename(os.path.join(output_dir,
                                                      "original_minus_predicted.nii.gz"))

    # Save the wmaps
    wmaps.to_filename(os.path.join(output_dir, "wmaps.nii.gz"))

    # Save the betamaps
    beta_maps.to_filename(os.path.join(output_dir, "betamaps.nii.gz"))

    # Save the residuals SD
    sd_residuals.to_filename(os.path.join(output_dir, "sd_residuals.nii.gz"))
    

    return wmaps