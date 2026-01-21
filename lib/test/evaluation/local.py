from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/data/code_Lon/VLT/HIPB_up_large/data/got10k_lmdb'
    settings.got10k_path = '/data/code_Lon/VLT/HIPB_up_large/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/data/code_Lon/VLT/HIPB_up_large/data/itb'
    settings.lasot_extension_subset_path_path = '/data/code_Lon/VLT/HIPB_up_large/data/lasot_extension_subset'
    settings.lasot_lmdb_path = '/data/code_Lon/VLT/HIPB_up_large/data/lasot_lmdb'
    settings.lasot_path = '/data/testing_dataset/LaSOT/'
    settings.network_path = '/data/code_Lon/VLT/HIPB_up_large/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/data/code_Lon/VLT/HIPB_up_large/data/nfs'
    settings.otb_path = '/data/code_Lon/VLT/HIPB_up_large/data/otb'
    settings.prj_dir = '/data/code_Lon/VLT/HIPB_up_large'
    settings.result_plot_path = '/data/code_Lon/VLT/HIPB_up_large/output/test/result_plots'
    settings.results_path = '/data/code_Lon/VLT/HIPB_up_large/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/data/code_Lon/VLT/HIPB_up_large/output'
    settings.segmentation_path = '/data/code_Lon/VLT/HIPB_up_large/output/test/segmentation_results'
    settings.tc128_path = '/data/code_Lon/VLT/HIPB_up_large/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/data/testing_dataset/TNL2K/'
    settings.tpl_path = ''
    settings.trackingnet_path = '/data/code_Lon/VLT/HIPB_up_large/data/trackingnet'
    settings.uav_path = '/data/code_Lon/VLT/HIPB_up_large/data/uav'
    settings.vot18_path = '/data/code_Lon/VLT/HIPB_up_large/data/vot2018'
    settings.vot22_path = '/data/code_Lon/VLT/HIPB_up_large/data/vot2022'
    settings.mgit_test_path = '/data/testing_dataset/MGIT-Test'
    settings.mgit_info_path = '/data/testing_dataset/MGIT-Info'
    settings.vot_path = '/data/code_Lon/VLT/HIPB_up_large/data/VOT2019'
    settings.youtubevos_dir = ''
    settings.vasttrack_dir = '/data/testing_dataset/vastrackTest/'


    return settings

