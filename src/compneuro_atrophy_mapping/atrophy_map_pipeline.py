import os.path as op

import compneuro_atrophy_mapping.utils as utils
import compneuro_atrophy_mapping.processing as proc


def run_pipeline():
    # Setup the arguments and all inputs
    args = utils.check_wmap_pipeline_args()

    # Run VBM if needed
    if args.need_control_vbm:
        proc.run_vbm(args.control_t1_dir)
    if args.need_studygroup_vbm:
        proc.run_vbm(args.studygroup_t1_dir)

    # Compute atrophy maps
    results = proc.compute_wmaps_from_vbm(gm_mod_merg_control=args.control_vbm_path,
                                         gm_mod_merg_studygroup=args.studygroup_vbm_path,
                                         design_matrix_cn=args.df_design_mat_cn,
                                         design_matrix_studygroup=args.df_design_mat_studygroup,
                                         n_jobs=args.n_jobs)

    # Save the results
    print(f"[INFO] Saving the results to: {op.abspath(args.output_dir)}")
    output_dir = args.output_dir
    for name, map in results.items():
        map.to_filename(op.join(output_dir, f"{name}.nii.gz"))


if __name__ == "__main__":
    run_pipeline()
