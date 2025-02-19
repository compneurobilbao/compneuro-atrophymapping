import io
import polars as pl
import subprocess as sp
import os.path as op

from os import makedirs, remove, getenv
from argparse import ArgumentParser
from compneuro_atrophy_mapping.utils import stdout_to_dataframe

import numpy as np

from nilearn import image


def _setup_parser():
    parser = ArgumentParser()
    parser.add_argument("--clusterim", "-c",
                        type=str, required=True,
                        help="Path to the binary image of the clusters.")
    parser.add_argument("--percentile", "-p",
                        type=float, required=True,
                        help="Percentile of the cluster sizes to keep.")
    parser.add_argument("--merge", "-m",
                        type=bool, required=False,
                        default=True,
                        help="Whether to merge the clusters in a 4D volume or not.")

    args = parser.parse_args()

    return args


def _check_args(args) -> None:
    if not op.exists(args.clusterim) or not op.isfile(args.clusterim):
        raise ValueError(f"File {args.clusterim} does not exist or it is not a file.")

    if args.percentile < 0 or args.percentile > 100:
        raise ValueError("Percentile must be higher than 0 and lower than 100.")


def run_pipeline() -> None:
    # Take a binary image of the clusters after thresholding with `binarize_wmaps`, and eliminate clusters
    # that are below a certain size percentile (percentiles computed per subject).
    # The idea is to eliminate small clusters that are likely to be noise.

    # Get arguments
    args = _setup_parser()
    _check_args(args)

    # Get the FSL binary directory
    FSLBIN = op.join(getenv("FSLDIR"), "bin")

    # Make the output directory
    wd = op.dirname(args.clusterim)
    prc_str = str(args.percentile).replace('.', 'p')
    out_dir = op.abspath(op.join(wd, f"filtered_clusters_percentile_{prc_str}"))
    print("### [INFO] Output directory: ", out_dir)
    makedirs(out_dir, exist_ok=True)

    # Load the binary image of the clusters
    cluster_im = image.load_img(args.clusterim)
    n_ims = cluster_im.shape[-1]

    for i in range(n_ims):
        # For each volume in the 4D image, extract the volume and compute the clusters. Output the sizes of the clusters
        cmd = (f"{FSLBIN}/fslroi {args.clusterim} {out_dir}/vol_{i}.nii.gz {i} 1 && "
        f"{FSLBIN}/fsl-cluster -i {out_dir}/vol_{i}.nii.gz -t 0.1 --osize={out_dir}/vol_{i}_sizes.nii.gz")
        output = sp.run(cmd, shell=True, capture_output=True).stdout
        # Get the cluster table in a polars DataFrame
        cluster_table = stdout_to_dataframe(output, header=True, sep="\t")

        # Compute the cluster size threshold based on the percentile given by the user
        try:
            prc_threshold = np.percentile(cluster_table["Voxels"].to_numpy(), args.percentile)
        except IndexError:  # If there are no clusters, set the threshold to 0
            prc_threshold = 0
        # Binarize the clusters based on the computed size threshold
        cmd = (f"{FSLBIN}/fslmaths {out_dir}/vol_{i}_sizes.nii.gz -thr {prc_threshold} -bin "
              f"{out_dir}/vol_{i}_prc_{prc_str}_clusters.nii.gz")
        sp.run(cmd, shell=True)

        # Remove extra files
        remove(f"{out_dir}/vol_{i}.nii.gz")
        remove(f"{out_dir}/vol_{i}_sizes.nii.gz")

        # Print logging
        print(f"### [INFO] Filtered clusters for volume {i + 1}/{n_ims}")

    print(f"### [INFO] Done filtering clusters of {args.clusterim} with a cluster-size percentile of {args.percentile}.")

    # If merge is True, merge the filtered clusters into a 4D volume
    if args.merge:
        print("### [INFO] Merging the filtered clusters into a 4D volume.")
        # Create the merge.txt file with the file paths of the filtered clusters
        mrg_txt = op.join(out_dir, "merge.txt")
        with open(mrg_txt, "w") as f:
            for i in range(n_ims):
                f.write(f"{out_dir}/vol_{i}_prc_{prc_str}_clusters.nii.gz\n")

        # Merge the filtered clusters
        cmd = f"{FSLBIN}/fslmerge -t {out_dir}/all_filtered_clusters_prc_{prc_str}.nii.gz $(cat {mrg_txt})"
        sp.run(cmd, shell=True)
        print("### [INFO] Done merging the filtered clusters.")

    print("### [INFO] Done!")


if __name__ == "__main__":
    run_pipeline()
