from io import BytesIO
import os.path as osp
import warnings
import subprocess as sp

from os import makedirs
from argparse import ArgumentParser

import polars as pl


def setup_wmap_pipeline_parser():
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
    help_dsgn_mat_cn = ("Path to the design matrix of the CONTROL group.\n"
                     "The design matrix can either be in FSL format (e.g.: .mat file,\n"
                     "preferred) or in TSV format with no header.\n"
                     "The design matrix should have the following columns, always\n"
                     "preceded by a column of 1s:\n"
                     "{1s, Age, Age^2, Sex, TIV, [MRI B0], [Center]}")
    parser.add_argument("-dcn", "--control_design_matrix",
                        type=str,
                        help=help_dsgn_mat_cn,
                        required=False)

    help_dsgn_mat_studygroup = ("Path to the design matrix of the STUDY group.\n"
                     "The design matrix can either be in FSL format (e.g.: .mat file,\n"
                     "preferred) or in TSV format with no header.\n"
                     "The design matrix should have the following columns, always\n"
                     "preceded by a column of 1s:\n"
                     "{1s, Age, Age^2, Sex, TIV, [MRI B0], [Center]}")
    parser.add_argument("-dpat", "--studygroup_design_matrix",
                        type=str,
                        help=help_dsgn_mat_studygroup,
                        required=False)

    help_n_jobs = ("Number of jobs to run in parallel when fitting the OLS model. "
                   "If -1, then all available CPUs will be used.")
    parser.add_argument("-n", "--n_jobs",
                        type=int,
                        help=help_n_jobs,
                        required=False,
                        default=1)

    parser.add_argument("-o", "--output_dir",
                        type=str,
                        help="Path to the output directory to store the wmaps",
                        required=True)

    args = parser.parse_args()
    return args


def setup_apply_model_parser():
    # TODO: Design how to reuse an already fit model to compute Wmaps on new data.
    # NOTE: It is not as trivial as reusing the betas. Masking and unmasking must be handled with care.
    parser = ArgumentParser(description="Atrophy maps from a given model.")
    parser
    pass


def _standardize_df(df: pl.DataFrame):
    # Standardize all columns except the first one
    df_standardized = df.with_columns([
        (pl.col(col) - pl.col(col).mean()) / pl.col(col).std()
        for col in df.columns[1:]
    ])

    return df_standardized


def _check_df_design_mat(design_mat_path: pl.DataFrame):
    # Check if the design matrix is in correct format
    if osp.exists(design_mat_path):
        if (not design_mat_path.endswith(".mat") and
            not design_mat_path.endswith(".tsv") and
            not design_mat_path.endswith(".txt")):
            err_msg = ("Design matrix must be in FSL (.mat), TSV, or .txt (with ' ' separator) format.")
            raise ValueError(err_msg)
        elif design_mat_path.endswith(".tsv"):
            df_design_mat = pl.read_csv(design_mat_path,
                                             has_header=False,
                                             separator="\t").cast(pl.Float32)
        elif design_mat_path.endswith(".txt"):
            df_design_mat = pl.read_csv(design_mat_path,
                                             has_header=False,
                                             separator=" ").cast(pl.Float32)
        # If FSL format, then convert it to TXT, read it, and store it as a numpy array
        elif design_mat_path.endswith(".mat"):
            design_txt_path = design_mat_path.replace(".mat", ".txt")
            # Convert from FSL format (.mat) to plain text format, nd read
            cmd = f"$FSLDIR/bin/Vest2Text {design_mat_path} {design_txt_path}"
            sp.run(cmd, shell=True)
            df_design_mat = pl.read_csv(design_txt_path,
                                             separator=" ",
                                             has_header=False).cast(pl.Float32)

        if df_design_mat.shape[1] < 3:
                err_msg = ("Design matrix must have at least 3 columns (no header!):\n"
                           "{1s, Age, Sex}. Optionals: {TIV, [MRI B0], [Center]}")
                raise ValueError(err_msg)

        # Standardize the design matrix
        df_design_mat_standardized = _standardize_df(df_design_mat)

        # Check if the first column is all ones
        if not all(df_design_mat_standardized.to_numpy()[:, 0] == 1):
            err_msg = ("[INFO] The first column of the design matrix must be a column of 1s. "
                       "Adding a preceding column of 1s to the design matrix.")
            print(err_msg)
            # Add the column of ones to the dataframe
            ones_col = pl.Series("ones", df_design_mat_standardized.shape[0] * [1])
            df_design_mat_standardized.insert_column(0, ones_col)
        df_design_mat_standardized_pandas = df_design_mat_standardized.to_pandas()

        return df_design_mat_standardized_pandas

    else:
        raise FileNotFoundError(f"Design matrix not found at {design_mat_path}")



def check_wmap_pipeline_args():
    args = setup_wmap_pipeline_parser()

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

    # Check if the design matrix is correct
    args.df_design_mat_cn = _check_df_design_mat(args.control_design_matrix)
    args.df_design_mat_studygroup = _check_df_design_mat(args.studygroup_design_matrix)

    # Check n_jobs
    if args.n_jobs < 1 and args.n_jobs != -1:
        err_msg = "Number of jobs must be at least 1."
        raise ValueError(err_msg)
    elif args.n_jobs == -1:
        print("[INFO] Running voxelwise OLS fitting in parallel with all available CPUs.")

    # Check if output path exists, if not create it
    if not osp.exists(args.output_dir):
        makedirs(args.output_dir, exist_ok=True)
        warnings.warn("Output directory does not exist. Creating it.")

    return args


def stdout_to_dataframe(stdout: bytes,
                         header: bool = True,
                         sep: str = "\t") -> pl.DataFrame:

    stdout_bytes = BytesIO(stdout)

    return pl.read_csv(stdout_bytes, has_header=header, separator=sep)
