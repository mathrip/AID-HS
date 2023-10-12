
#define user paths for different user systems
import os
import pwd
from configparser import ConfigParser, NoOptionError, NoSectionError

# get scripts dir (parent dir of dir that this file is in)
SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# read config file from scripts_dir
config_fname = os.path.join(SCRIPTS_DIR, 'config.ini')
config = ConfigParser()
config.read(config_fname)

try:
    DATA_PATH = config.get('DEFAULT', 'data_path')
    print(f'Setting DATA_PATH to {DATA_PATH}')
except NoOptionError as e:
    print(f'No aidhs_data_path defined in {config_fname}')
    DATA_PATH = ""

try:
    BASE_PATH = config.get('develop', 'base_path')
    print(f'Setting BASE_PATH to {BASE_PATH}')
except (NoOptionError, NoSectionError) as e:
    print(f"No base_path defined in {config_fname}!")
    BASE_PATH = ""
try:
    EXPERIMENT_PATH = config.get('develop', 'experiment_path')
    print(f'Setting EXPERIMENT_PATH to {EXPERIMENT_PATH}')
except (NoOptionError, NoSectionError) as e:
    print(f"No experiment_path defined in {config_fname}!")
    EXPERIMENT_PATH = ""
try:
    PARAMS_PATH = config.get('develop', 'params_path')
    print(f'Setting PARAMS_PATH to {PARAMS_PATH}')
except (NoOptionError, NoSectionError) as e:
    print(f"No params_path defined in {config_fname}!")
    PARAMS_PATH = ""
try:
    FS_SUBJECTS_PATH = config.get('develop', 'fs_subjects_path')
    print(f'Setting FS_SUBJECTS_PATH to {FS_SUBJECTS_PATH}')
except (NoOptionError, NoSectionError) as e:
    print(f"No fs_subjects_path defined in {config_fname}!")
    FS_SUBJECTS_PATH = ""
try:
    HIPPUNFOLD_SUBJECTS_PATH = config.get('develop', 'hippunfold_subjects_path')
    print(f'Setting HIPPUNFOLD_SUBJECTS_PATH to {HIPPUNFOLD_SUBJECTS_PATH}')
except (NoOptionError, NoSectionError) as e:
    print(f"No fs_subjects_path defined in {config_fname}!")
    HIPPUNFOLD_SUBJECTS_PATH = ""
try:
    BIDS_SUBJECTS_PATH = config.get('develop', 'bids_subjects_path')
    print(f'Setting BIDS_SUBJECTS_PATH to {BIDS_SUBJECTS_PATH}')
except (NoOptionError, NoSectionError) as e:
    print(f"No fs_subjects_path defined in {config_fname}!")
    BIDS_SUBJECTS_PATH = ""


# paths to important data files - relative to BASE_PATH
DEMOGRAPHIC_FEATURES_FILE = os.path.join(DATA_PATH, "demographics_file.csv")

# params file
CLIPPING_PARAMS_FILE='clipping_parameters_sigma.json'
ICV_PARAMS_FILE='icv_correct_parameters.hdf5'
COMBAT_PARAMS_FILE='combat_parameters_avg.hdf5'
NORM_CONTROLS_PARAMS_FILE='norm_asym_parameters.hdf5'
FS_STATS_FILE='stats/aparc.DKTatlas+aseg.deep.volume.stats'

# templates
HIPPO_MASK_FILE = os.path.join(PARAMS_PATH,'templates/tmp_hemi-L_space-unfolded_den-0p5mm_label-hipp_midthickness_mask.surf.gii')
DENTATE_MASK_FILE = os.path.join(PARAMS_PATH,'templates/tmp_hemi-L_space-unfolded_den-0p5mm_label-dentate_midthickness_mask.surf.gii')
SURFACE_UNFOLD_FILE = os.path.join(PARAMS_PATH,'templates/tmp_hemi-L_space-unfolded_den-0p5mm_label-hipp_midthickness.surf.gii')
SURFACE_FOLD_FILE = os.path.join(PARAMS_PATH,'templates/tmp_hemi-L_space-T1w_den-0p5mm_label-hipp_midthickness.surf.gii')
SURFACE_UNFOLD_DENTATE_FILE = os.path.join(PARAMS_PATH,'templates/tmp_hemi-L_space-unfolded_den-0p5mm_label-dentate_midthickness.surf.gii')
SURFACE_FOLD_DENTATE_FILE = os.path.join(PARAMS_PATH,'templates/tmp_hemi-L_space-T1w_den-0p5mm_label-dentate_midthickness.surf.gii')
SUBFIELDS_LABEL_FILE = os.path.join(PARAMS_PATH,'templates/tmp_hemi-L_space-T1w_den-0p5mm_label-hipp_subfields.label.gii')
HBT_LABEL_FILE = os.path.join(PARAMS_PATH,'templates/tmp_hemi-L_space-unfolded_den-0p5mm_label-hipp_HBT.label.gii')


# default values
DEFAULT_HDF5_FILE_ROOT = "{site_code}_{group}_featurematrix.hdf5" # filename of hdf5 files
# number of vertices per hemi
NVERT_HIPP = 7262 
NVERT_DG = 1788
NVERT_AVG = 1


# list of sites code used for training
SITE_CODES=['H1', 'H11', 'H16', 'H29']