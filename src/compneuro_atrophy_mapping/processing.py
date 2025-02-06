import os
import subprocess as sp

import numpy as np
import nibabel as nib

from nilearn import image
from nilearn.glm.second_level import SecondLevelModel
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
                          design_matrix: np.ndarray,
                          output_dir: str) -> nib.nifti1.Nifti1Image:
    """_summary_

    Parameters
    ----------
    gm_mod_merg_control : str
        _description_
    gm_mod_merg_studygroup : str
        _description_
    design_matrix : np.ndarray
        _description_
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

    # Aply masks
    vbm_response_cn = unmask(apply_mask(gm_mod_control, gm_mask_cn),
                             gm_mask_cn)
    vbm_response_studygroup = unmask(apply_mask(gm_mod_studygroup, gm_mask_studygroup),
                                     gm_mask_studygroup)

    # SecondLevel Model
    slm = SecondLevelModel(mask_img=gm_mask_cn, minimize_memory=False)
    slm.fit(vbm_response_cn, design_matrix=design_matrix)

    # Build contrast (assume the first column is a column of ones, meaning all subjects are CN)
    design_contrast = np.array([1] + (design_matrix.shape[1] - 1) * [0])
    
    _ = slm.compute_contrast(second_level_contrast=design_contrast)

    # Get Residuals and compute wmaps
    sd_of_residuals = image.math_img("np.std(residuals, axis=3)", residuals=slm.residuals)

    wmaps = image.math_img("(orig - predicted) / sd_of_residuals[..., np.newaxis]",
                               orig=vbm_response_studygroup,
                               predicted=slm.predicted,
                               sd_of_residuals=sd_of_residuals)

    # Save the wmaps
    wmaps.to_filename(os.path.join(output_dir, "wmaps.nii.gz"))

    return wmaps