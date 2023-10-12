""" This script run the data preprocessing on new_subjects
1. smooth surface-based features and clip outliers vertices
2. average surface-based data into one value per hippocampi
3. remove site bias with combat on average features
4. compute asymmetries and normalisation on harmonised data
"""

#Import library
import os
import sys
import numpy as np
import pandas as pd
import time
import tempfile
import argparse

from aidhs.aidhs_cohort_hip import AidhsCohort
from aidhs.data_preprocessing import Preprocess, Feature
from aidhs.paths import DATA_PATH, BASE_PATH, PARAMS_PATH, ICV_PARAMS_FILE, NORM_CONTROLS_PARAMS_FILE, COMBAT_PARAMS_FILE, CLIPPING_PARAMS_FILE, SITE_CODES, DEMOGRAPHIC_FEATURES_FILE
from aidhs.tools_print import get_m


def create_dataset_file(subjects_ids, save_file):
    df=pd.DataFrame()
    if  isinstance(subjects_ids, str):
        subjects_ids=[subjects_ids]
    df['subject_id']=subjects_ids
    df['split']=['test' for subject in subjects_ids]
    df.to_csv(save_file)

def which_combat_file(site_code):
    file_site=os.path.join(BASE_PATH, f'AIDHS_{site_code}', f'{site_code}_combat_parameters.hdf5')
    if site_code in SITE_CODES:
        print(get_m(f'Use combat parameters from AID-HS cohort', None, 'INFO'))
        return os.path.join(PARAMS_PATH, COMBAT_PARAMS_FILE)
    elif os.path.isfile(file_site):
        print(get_m(f'Use combat parameters from site', None, 'INFO'))
        return file_site
    else:
        print(get_m(f'Could not find combat parameters for {site_code}', None, 'WARNING'))
        return 'None'

def check_demographic_file(demographic_file, subject_ids):
    #check demographic file has the right columns
    try:
        df = pd.read_csv(demographic_file, index_col=None)
        df.keys().str.contains('ID')
        df.keys().str.contains('Age')
        df.keys().str.contains('Sex')
    except Exception as e:
        sys.exit(get_m(f'Error with the demographic file provided for the harmonisation\n{e}', None, 'ERROR'))
    #check demographic file has the right subjects
    if set(subject_ids).issubset(set(np.array(df['ID']))):
        return demographic_file
    else:
        sys.exit(get_m(f'Missing subject in the demographic file', None, 'ERROR'))


def new_site_harmonisation(subject_ids, dataset, features, site_code, output_dir=BASE_PATH):

    ### INITIALISE ###
    #check enough subjects for harmonisation
    if len(np.unique(subject_ids))<20:
        print(get_m(f'We recommend to use at least 20 subjects for an acurate harmonisation of the data. Here you are using only {len(np.unique(subject_ids))}', None, 'WARNING'))


    demographic_file = os.path.join(BASE_PATH, DEMOGRAPHIC_FEATURES_FILE)
    check_demographic_file(demographic_file, subject_ids)
   
    ### COMBAT DISTRIBUTED DATA ###
    #-----------------------------------------------------------------------------------------------
    print(get_m(f'Compute combat harmonisation parameters for new site', None, 'STEP'))
  
    #create cohort for the new subject
    c_smooth= AidhsCohort(hdf5_file_root='{site_code}_{group}_featurematrix_smoothed_avg.hdf5', dataset=dataset)
    #create object combat
    combat =Preprocess(c_smooth,
                           site_codes=[site_code],
                           write_hdf5_file_root="AIDHS_{site_code}/{site_code}_combat_parameters.hdf5",
                           data_dir=output_dir)
    #features names
    for feature in features:
        print(feature)
        combat.get_combat_new_site_parameters(feature, demographic_file)

def run_data_processing_new_subjects(subject_ids, site_code, output_dir=BASE_PATH, do_harmonisation=False, harmonisation_only=False ):

    # initialise surface_features and smoothing kernel
    surface_features = {
                '.label-dentate.curvature':1,
                '.label-dentate.gyrification':1,
                '.label-hipp.curvature':1,
                '.label-hipp.gauss-curv_filtered_sm1': None,
                '.label-hipp.gyrification':1,
                '.label-hipp.thickness':1
            }


    # initialise base features
    base_features=[
        '.label-{}.curvature.sm1',
        '.label-{}.gauss-curv_filtered_sm1',
        '.label-{}.gyrification.sm1',
        '.label-{}.thickness.sm1',
        '.label-avg.hippunfold_volume',
        ]
    feat = Feature()
    features_smooth_avg = [feature.format('avg') for feature in base_features]
    features_combat_avg = [feat.combat_feat(feature) for feature in features_smooth_avg]

    print(time.asctime(time.localtime(time.time())))

    ### INITIALISE ###
    #create dataset
    tmp = tempfile.NamedTemporaryFile(mode="w")
    create_dataset_file(subject_ids, tmp.name)


    ### SMOOTH SURFACE-BASE FEATURES ###
    #-----------------------------------------------------------------------------------------------
    print('PROCESS: CLIP AND SMOOTH SURFACE-BASED FEATURES TO REMOVE OUTLIERS')
    #create cohort to smooth
    c_raw = AidhsCohort(hdf5_file_root='{site_code}_{group}_featurematrix.hdf5', dataset=tmp.name)
    #create object
    smoothing = Preprocess(c_raw , 
                        write_hdf5_file_root='{site_code}_{group}_featurematrix_smoothed.hdf5', 
                        data_dir=output_dir)

    #file to store subject with outliers vertices
    outliers_file=os.path.join(output_dir, 'list_subject_extreme_vertices.csv')

    #file with clipping parameters
    clipping_parameters = os.path.join(PARAMS_PATH, CLIPPING_PARAMS_FILE)

    for feature in np.sort(list(set(surface_features))):
        print(feature)
        try:
            smoothing.smooth_data(feature, surface_features[feature], clipping_parameters, outliers_file  )
        except Exception as e: 
            print(f'Error for feature {feature} : \n {e}')



    ### AVERAGE SURFACE BASED FEATURES ###
    # -----------------------------------------------------------------------------------------------
    print("PROCESS : AVERAGE SURFACE BASED FEATURES AND EXTRACT VOLUMES")

    # create cohort to average
    c_smooth = AidhsCohort(hdf5_file_root="{site_code}_{group}_featurematrix_smoothed.hdf5", dataset=tmp.name)
    # create objec
    avg = Preprocess(
    c_smooth,
    write_hdf5_file_root="{site_code}_{group}_featurematrix_smoothed_avg.hdf5",
    data_dir=output_dir,
    )

    # call function to extract volumes / commpute average features
    for feature in base_features:
        print(feature)
        if 'volume' in feature:
            avg.extract_volumes_avg(feature)
        else:
            avg.compute_avg_feature(feature)


    ### CORRECT HIPPOCAMPAL VOLUMES FOR INTRACRANIAL VOLUME ###
    # -----------------------------------------------------------------------------------------------
    print("PROCESS : CORRECT HIPPOCAMPAL VOLUMES FOR INTRACRANIAL VOLUME")

    # create cohort to correct icv 
    c_avg = AidhsCohort(hdf5_file_root="{site_code}_{group}_featurematrix_smoothed_avg.hdf5", dataset=tmp.name)
  
    # create object 
    icv_correct = Preprocess(
       c_avg,
       write_hdf5_file_root="{site_code}_{group}_featurematrix_smoothed_avg.hdf5",
       data_dir=BASE_PATH,
    )

    # call function to comput ICV correction
    for feature in features_smooth_avg:
        if 'volume' in feature:
            print(feature)
            icv_correct.correct_volume_ICV(feature, params_icv_correct=os.path.join(PARAMS_PATH,ICV_PARAMS_FILE))


    #add features to list and update
    base_features.append('.label-avg.hippunfold_volume_icvcorr')
    features_smooth_avg = [feature.format('avg') for feature in base_features]
    features_combat_avg = [feat.combat_feat(feature) for feature in features_smooth_avg]

    if do_harmonisation:
        ### COMPUTE HARMONISATION ###
        # -----------------------------------------------------------------------------------------------
        print("PROCESS : RUN HARMONISATION")
        new_site_harmonisation(subject_ids, dataset=tmp.name, features=features_smooth_avg, site_code=site_code, output_dir=BASE_PATH)

    if not harmonisation_only:
        ### COMBAT AVG FEATURES ###
        # -----------------------------------------------------------------------------------------------
        print("PROCESS : COMBAT AVG FEATURES")
        # get combat parameters file
        combat_params_file = which_combat_file(site_code)
        # create cohort to combat
        c_avg = AidhsCohort(hdf5_file_root="{site_code}_{group}_featurematrix_smoothed_avg.hdf5", dataset=tmp.name)
        # create object combat
        combat = Preprocess(
        c_avg,
        write_hdf5_file_root="{site_code}_{group}_featurematrix_combat_avg.hdf5",
        data_dir=output_dir,
        )
        # call function to combat data
        for feature in features_smooth_avg:
            print(feature)
            combat.combat_new_subject(feature, combat_params_file=combat_params_file)

        ###  ASYMMETRIES ON AVG COMBAT FEATURES ###
        # -----------------------------------------------------------------------------------------------
        print('PROCESS: ASYMMETRIES & NORMALISATION')
            
        #create cohort to normalise
        c_norm = AidhsCohort(hdf5_file_root='{site_code}_{group}_featurematrix_combat_avg.hdf5', dataset=tmp.name)
        # create object normalisation
        norm = Preprocess(c_norm, 
                            write_hdf5_file_root='{site_code}_{group}_featurematrix_norm_avg.hdf5', 
                            data_dir=output_dir)
        # call function to normalise data
        for feature in features_combat_avg:
            print(feature)
            norm.asymmetry_internorm_subject(feature, params_norm=os.path.join(PARAMS_PATH, NORM_CONTROLS_PARAMS_FILE))


    print(time.asctime(time.localtime(time.time())))

def run_pipeline_preprocessing(site_code, list_ids=None, sub_id=None, output_dir=BASE_PATH,  harmonisation_only=False, verbose=False):
    site_code = str(site_code)
    subject_ids=None
    if list_ids != None:
        list_ids=os.path.join(DATA_PATH, list_ids)
        try:
            sub_list_df=pd.read_csv(list_ids)
            print(sub_list_df)
            subject_ids=np.array(sub_list_df.ID.values)
            print(subject_ids)
        except:
            subject_ids=np.array(np.loadtxt(list_ids, dtype='str', ndmin=1)) 
        # else:
        #     sys.exit(get_m(f'Could not open {list_ids}', None, 'ERROR'))             
    elif sub_id != None:
        subject_ids=np.array([sub_id])
    else:
        print(get_m(f'No ids were provided', None, 'ERROR'))
        print(get_m(f'Please specify both subject(s) and site_code ...', None, 'ERROR'))
        sys.exit(-1) 
       
    #check that combat parameters exist for this site or compute it
    combat_params_file = which_combat_file(site_code)
    if combat_params_file=='None':
        print(get_m(f'Compute combat parameters for {site_code} with subjects {subject_ids}', None, 'INFO'))
        do_harmonisation = True
        #check that demographic file exist and is adequate
        demographic_file = os.path.join(DATA_PATH, DEMOGRAPHIC_FEATURES_FILE) 
        if os.path.isfile(demographic_file):
            print(get_m(f'Use demographic file {demographic_file}', None, 'INFO'))
            demographic_file = check_demographic_file(demographic_file, subject_ids) 
        else:
            sys.exit(get_m(f'Could not find demographic file {demographic_file}', None, 'ERROR'))
    else:
        do_harmonisation=False
    #compute the combat parameters for a new site
    run_data_processing_new_subjects(subject_ids, site_code=site_code, output_dir=output_dir, do_harmonisation=do_harmonisation, harmonisation_only=harmonisation_only)


if __name__ == '__main__':

    #parse commandline arguments 
    parser = argparse.ArgumentParser(description='data-processing on new subject')
    parser.add_argument("-id","--id",
                        help="Subject ID.",
                        default=None,
                        required=False,
                        )
    parser.add_argument("-ids","--list_ids",
                        default=None,
                        help="File containing list of ids. Can be txt or csv with 'ID' column",
                        required=False,
                        )
    parser.add_argument("-site",
                        "--site_code",
                        help="Site code",
                        required=True,
                        )
    parser.add_argument('--harmo_only', 
                        action="store_true", 
                        help='only compute the harmonisation combat parameters, no further process',
                        required=False,
                        default=False,
                        )
    parser.add_argument("--debug_mode", 
                        help="mode to debug error", 
                        required=False,
                        default=False,
                        action="store_true",
                        )

    
    args = parser.parse_args()
    print(args)
   
    run_pipeline_preprocessing(
                    site_code=args.site_code,
                    list_ids=args.list_ids,
                    sub_id=args.id,
                    harmonisation_only = args.harmo_only,
                    verbose = args.debug_mode,
                    )


