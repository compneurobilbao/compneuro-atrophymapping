import importlib.resources as pkg_resources

from os.path import join as opj

from nilearn import image
from compneuro_atrophy_mapping import data

# Locate the full MNI152 holeless brain mask
datasets_path = str(pkg_resources.files(data))
full_mni152_holeless_brain_path = opj(datasets_path, "full_MNI152_T1_2mm_brain_mask.nii.gz")

# Load the full MNI152 holeless brain mask
FULL_MNI_152_HOLELESS_BRAIN = image.load_img(full_mni152_holeless_brain_path)

# Cleanup so you can only import the template and the template path
del data
del datasets_path