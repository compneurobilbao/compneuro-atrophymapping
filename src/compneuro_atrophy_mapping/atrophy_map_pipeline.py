import compneuro_atrophy_mapping.utils as utils
import compneuro_atrophy_mapping.processing as proc


def run_pipeline():
    # Setup the arguments and all inputs
    args = utils.check_args_and_data()

    # Run VBM if needed
    if args.need_control_vbm:
        proc.run_vbm(args.control_t1_dir)
    if args.need_studygroup_vbm:
        proc.run_vbm(args.studygroup_t1_dir)

    # Compute atrophy maps
    proc.compute_atrophy_wmaps(gm_mod_merg_control=args.control_vbm_path,
                               gm_mod_merg_studygroup=args.studygroup_vbm_path,
                               design_matrix_cn=args.df_design_mat_cn,
                               design_matrix_studygroup=args.df_design_mat_studygroup,
                               output_dir=args.output_dir)

if __name__ == "__main__":
    run_pipeline()
