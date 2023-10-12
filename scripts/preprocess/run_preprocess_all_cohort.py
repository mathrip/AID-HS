""" This script run the data preprocessing on a whole cohort using Preprocess class 
1. smooth surface-based features and clip outliers vertices
2. average surface-based data into one value per hippocampi
3. remove site bias with combat on average features
4. compute asymmetries and normalisation on harmonised data
"""

#Import library
import os
import sys
import numpy as np

from aidhs.aidhs_cohort_hip import AidhsCohort
from aidhs.data_preprocessing import Preprocess, Feature
from aidhs.paths import BASE_PATH
import time

print(time.asctime(time.localtime(time.time())))

# Initialise sites  
site_codes = [
            "H1",
            "H16",
            "H11",
            "H29"
             ]

# initialise surface_features and smoothing kernel
surface_features = {
            '.label-dentate.curvature':1,
            '.label-dentate.gyrification':1,
            '.label-hipp.curvature':1,
            '.label-hipp.gauss-curv_filtered_sm1':None,
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
features_asym_avg = [feat.asym_feat(feature) for feature in features_combat_avg]

#TODO: add find clipping parameters 
#Need to run notebook analysis_distribution_raw_features.ipynb to extract clipping parameters

### SMOOTH SURFACE-BASE FEATURES ###
#-----------------------------------------------------------------------------------------------
print('PROCESS: CLIP AND SMOOTH SURFACE-BASED FEATURES TO REMOVE OUTLIERS')
#create cohort to smooth
c_raw = AidhsCohort(hdf5_file_root='{site_code}_{group}_featurematrix_rawT1.hdf5', dataset='dataset_C-P-DC_alltrain.csv')
#create object
smoothing = Preprocess(c_raw , 
                       site_codes=site_codes, 
                       write_hdf5_file_root='{site_code}_{group}_featurematrix_smoothed.hdf5', 
                       data_dir=BASE_PATH)

#file to store subject with outliers vertices
outliers_file=os.path.join(BASE_PATH, 'list_subject_extreme_vertices.csv')

#file with clipping parameters
clipping_parameters = os.path.join(BASE_PATH,'clipping_parameters_sigma.json')

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
c_smooth = AidhsCohort(hdf5_file_root="{site_code}_{group}_featurematrix_smoothed.hdf5", dataset='dataset_C-P-DC_alltrain.csv')
# create objec
avg = Preprocess(
   c_smooth,
   site_codes=site_codes,
   write_hdf5_file_root="{site_code}_{group}_featurematrix_smoothed_avg.hdf5",
   data_dir=BASE_PATH,
)

# call function to extract volumes and correct for / commpute average features
for feature in base_features:
    print(feature)
    if 'volume' in feature:
        avg.extract_volumes_avg(feature)
    else:
        avg.compute_avg_feature(feature)


### CORRECT HIPPOCAMPAL VOLUMES FOR INTRACRANIAL VOLUME ###
# -----------------------------------------------------------------------------------------------
print("PROCESS : AVERAGE SURFACE BASED FEATURES AND EXTRACT VOLUMES")

# create cohort to correct icv 
c_avg = AidhsCohort(hdf5_file_root="{site_code}_{group}_featurematrix_smoothed_avg.hdf5", dataset='dataset_C-P-DC_alltrain.csv')
# create cohort to compute parameters correct icv 
c_avg_controls = AidhsCohort(hdf5_file_root="{site_code}_{group}_featurematrix_smoothed_avg.hdf5", dataset='dataset_C.csv')

# create object 
icv_correct = Preprocess(
   c_avg,
   site_codes=site_codes,
   write_hdf5_file_root="{site_code}_{group}_featurematrix_smoothed_avg.hdf5",
   data_dir=BASE_PATH,
)

# call function to comput ICV correction
for feature in features_smooth_avg:
    if 'volume' in feature:
        print(feature)
        icv_correct.compute_parameters_correction_volume_ICV(feature, c_avg_controls, params_icv_correct=os.path.join(BASE_PATH,"icv_correct_parameters.hdf5"))
        icv_correct.correct_volume_ICV(feature, params_icv_correct=os.path.join(BASE_PATH,"icv_correct_parameters.hdf5"))

#add features to list and update
base_features.append('.label-avg.hippunfold_volume_icvcorr')
features_smooth_avg = [feature.format('avg') for feature in base_features]
features_combat_avg = [feat.combat_feat(feature) for feature in features_smooth_avg]
features_asym_avg = [feat.asym_feat(feature) for feature in features_combat_avg]


### COMBAT AVG FEATURES ###
# -----------------------------------------------------------------------------------------------
print("PROCESS : COMBAT AVG FEATURES")

# create cohort to combat
c_avg = AidhsCohort(hdf5_file_root="{site_code}_{group}_featurematrix_smoothed_avg.hdf5", dataset='dataset_C-P-DC_alltrain.csv')
# create object combat
combat = Preprocess(
   c_avg,
   site_codes=site_codes,
   write_hdf5_file_root="{site_code}_{group}_featurematrix_combat_avg_TMP.hdf5",
   data_dir=BASE_PATH,
)
# call function to combat data
for feature in features_smooth_avg:
    print(feature)
    combat.combat_whole_cohort(
       feature, outliers_file=None, combat_params_file=os.path.join(BASE_PATH,"combat_parameters_avg.hdf5")
   )
    # to produce features without combat
    # combat.combat_whole_cohort(feature, outliers_file=None, combat_params_file=None)


###  ASYMMETRIES ON AVG COMBAT FEATURES ###
# -----------------------------------------------------------------------------------------------
print('PROCESS: ASYMMETRIES & NORMALISATION')
    
#create cohort to normalise
c_norm = AidhsCohort(hdf5_file_root='{site_code}_{group}_featurematrix_combat_avg.hdf5', dataset='dataset_C-P-DC_alltrain.csv')
# create cohort of controls for inter normalisation if differente
c_controls = AidhsCohort(hdf5_file_root='{site_code}_{group}_featurematrix_combat_avg.hdf5', dataset='dataset_C.csv')
# create object normalisation
norm = Preprocess(c_norm, 
                    site_codes=site_codes, 
                    write_hdf5_file_root='{site_code}_{group}_featurematrix_norm_avg.hdf5', 
                    data_dir=BASE_PATH)
# call function to normalise data
for feature in features_combat_avg:
    print(feature)
    #save parameters of normalisation by controls 
    norm.compute_mean_std_controls(feature, cohort=c_controls, asym=True, params_norm=os.path.join(BASE_PATH, "norm_asym_parameters.hdf5"))
    # apply asym and normalisation
    norm.asymmetry_internorm_subject(feature, params_norm=os.path.join(BASE_PATH, "norm_asym_parameters.hdf5"))


print(time.asctime(time.localtime(time.time())))



