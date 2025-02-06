import os.path as osp
import warnings
import subprocess as sp

import polars as pl

from os import makedirs
from argparse import ArgumentParser


def setup_parser():
    parser = ArgumentParser(description="Compute atrophy maps from VBM data")

    # VBM arguments
    parser.add_argument("-vbmcn", "--control_vbm_path",
                        type=str,
                        help="Path to the control VBM data, to the GM_mod_merg_sX.nii.gz file",
                        required=False)
    parser.add_argument("-vbmpat", "--studygroup_vbm_path",
                        type=str,
                        help="Path to the clinical group VBM data, to the GM_mod_merg_sX.nii.gz file",
                        required=False)

    # T1 arguments
    help_control_t1_dir = "Path to the control T1 data, a collection of NIfTI files."
    parser.add_argument("-t1cn", "--control_t1_dir",
                        type=str,
                        help=help_control_t1_dir,
                        required=False)
    help_studygrp_t1_dir = "Path to the study group T1 data, a collection of NIfTI files."
    parser.add_argument("-t1pat", "--studygroup_t1_dir",
                        type=str,
                        help=help_studygrp_t1_dir,
                        required=False)

    # Design matrix of the control group
    help_dsgn_mat = ("Path to the design matrix of the CONTROL group.\n"
                     "The design matrix can either be in FSL format (e.g.: .mat file,\n"
                     "preferred) or in TSV format with no header.\n"
                     "The design matrix should have the following columns, always\n"
                     "preceded by a column of 1s:\n"
                     "{1s, Age, Age^2, Sex, TIV, [MRI B0], [Center]}")
    parser.add_argument("-d", "--control_design_matrix",
                        type=str,
                        help=help_dsgn_mat,
                        required=False)

    parser.add_argument("-o", "--output_dir",
                        type=str,
                        help="Path to the output directory to store the wmaps",
                        required=True)

    args = parser.parse_args()
    return args


def check_args_and_data():
    args = setup_parser()

    # Check if the VBM data was provided (control group)
    if args.control_vbm_path is not None:
        if osp.exists(args.control_vbm_path) and "GM_mod_merg" in args.control_vbm_path:
            args.has_control_vbm_path = True
        else:
            err_msg = (f"Control VBM data not found at {args.control_vbm_path}, or it\n"
                       "does not have 'GM_mod_merg' in the filename.")
            raise FileNotFoundError(err_msg)
        args.has_control_vbm_path = True
    else:
        args.has_control_vbm_path = False

    # Check if the VBM data was provided (study group)
    if args.studygroup_vbm_path is not None:
        args.has_studygroup_vbm_path = True
        if osp.exists(args.studygroup_vbm_path) and "GM_mod_merg" in args.studygroup_vbm_path:
            args.has_control_vbm_path = True
        else:
            err_msg = (f"Study group VBM data not found at {args.studygroup_vbm_path},\n"
                       "or it does not have 'GM_mod_merg' in the filename.")
            raise FileNotFoundError(err_msg)
    else:
        args.has_studygroup_vbm_path = False


    # Check if the T1 data was provided (control group)
    if args.control_t1_dir is not None:
        if not osp.isdir(args.control_t1_dir):
            err_msg = f"Control T1 data directory not found at {args.control_t1_dir}."
            raise FileNotFoundError(err_msg)
        args.has_control_t1_dir = True
    else:
        args.has_control_t1_dir = False

    # Check if the T1 data was provided (study group)
    if args.studygroup_t1_dir is not None:
        if not osp.isdir(args.studygroup_t1_dir):
            err_msg = f"Control T1 data directory not found at {args.studygroup_t1_dir}."
            raise FileNotFoundError(err_msg)
        args.has_studygroup_t1_dir = True
    else:
        args.has_studygroup_t1_dir = False

    # Check if any of the groups is lacking both VBM and T1 data
    if ((not args.has_control_vbm_path and not args.has_control_t1_dir) or
        (not args.has_studygroup_vbm_path and not args.has_studygroup_t1_dir)):
        err_msg = ("Both groups must have either VBM or T1 data.")
        raise ValueError(err_msg)
    # Check if we have to run VBM
    elif (args.has_control_t1_dir and not args.has_control_vbm_path):
        args.need_control_vbm = True
    elif (args.has_studygroup_t1_dir and not args.has_studygroup_vbm_path):
        args.need_studygroup_vbm = True
    else:
        args.need_control_vbm = False
        args.need_studygroup_vbm = False

    # Check if the design matrix is in correct format
    if osp.exists(args.control_design_matrix):
        if (not args.control_design_matrix.endswith(".mat") and
            not args.control_design_matrix.endswith(".tsv")):
            err_msg = ("Design matrix must be in FSL (.mat) or TSV format.")
            raise ValueError(err_msg)
        elif args.control_design_matrix.endswith(".tsv"):
            args.df_design_mat = pl.read_csv(args.control_design_matrix,
                                             has_header=False,
                                             separator="\t").cast(pl.Float32).to_pandas()
        # If FSL format, then convert it to TXT, read it, and store it as a numpy array
        elif args.control_design_matrix.endswith(".mat"):
            design_txt_path = args.control_design_matrix.replace(".mat", ".txt")
            # Convert from FSL format (.mat) to plain text format, nd read
            cmd = f"$FSLDIR/bin/Vest2Text {args.control_design_matrix} {design_txt_path}"
            sp.run(cmd, shell=True)
            args.df_design_mat = pl.read_csv(design_txt_path,
                                             separator=" ",
                                             has_header=False).cast(pl.Float32).to_pandas()

        if args.df_design_mat.shape[1] < 3:
                err_msg = ("Design matrix must have at least 3 columns (no header!):\n"
                           "{1s, Age, Sex}. Optionals: {TIV, [MRI B0], [Center]}")
                raise ValueError(err_msg)

        # Check if the first column is all ones
        if not all(args.df_design_mat.to_numpy()[:, 0] == 1):
            err_msg = ("The first column of the design matrix must be a column of 1s.")
            raise ValueError(err_msg)
    else:
        raise FileNotFoundError(f"Design matrix not found at {args.control_design_matrix}")

    # Check if output path exists, if not create it
    if not osp.exists(args.output_dir):
        makedirs(args.output_dir, exist_ok=True)
        warnings.warn("Output directory does not exist. Creating it.")
            

    return args
