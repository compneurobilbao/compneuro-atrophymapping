import subprocess as sp
import os.path as op
from os import getenv

from argparse import ArgumentParser

def _setup_parser():
    parser = ArgumentParser()
    parser.add_argument("--wmap", "-w", type=str, required=True)
    parser.add_argument("--threshold", "-t", type=float, required=True)

    args = parser.parse_args()

    return args


def _check_args(args) -> None:
    if not op.exists(args.wmap) or not op.isfile(args.wmap):
        raise ValueError(f"File {args.wmap} does not exist or it is not a file.")

    if args.threshold <= 0:
        raise ValueError("Threshold must be higher than 0.")


def run_pipeline():
    args = _setup_parser()
    _check_args(args)

    # Get the FSL binary directory
    FSLBIN = op.join(getenv("FSLDIR"), "bin")

    # Extract the directory
    wmap = args.wmap
    threshold = args.threshold
    wd = op.dirname(wmap)

    # Threshold the wmap for getting the growth map
    out_path_atrophy = op.join(wd, f"growth_map_thr_{str(threshold).replace('.', 'p')}.nii.gz")
    comm = f"{FSLBIN}/fslmaths {args.wmap} -thr {threshold} -bin {out_path_atrophy}"
    sp.run(comm, shell=True, check=True)

    # Make the negative wmap
    out_path_nwmap = op.join(wd, "neg_wmap.nii.gz")
    comm = f"{FSLBIN}/fslmaths {args.wmap} -mul -1 {out_path_nwmap}"
    sp.run(comm, shell=True, check=True)

    # Threshold the negative wmap to get the atrophy map, then delete the negative wmap
    out_path_atrophy = op.join(wd, f"atrophy_map_thr_{str(threshold).replace('.', 'p')}.nii.gz")
    comm = f"{FSLBIN}/fslmaths {out_path_nwmap} -thr {threshold} -bin {out_path_atrophy}"
    sp.run(comm, shell=True, check=True)
    sp.run(f"rm {out_path_nwmap}", shell=True, check=True)


if __name__ == "__main__":
    run_pipeline()