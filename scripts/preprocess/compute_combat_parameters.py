### script to calculate reference combat parameters from AID-HS cohort to use in distributed harmonisation for new site

import sys
import os
import numpy as np
import pandas as pd
import pickle
import neuroCombat as nc

import aidhs.distributedCombat as dc
from aidhs.paths import BASE_PATH
from aidhs.aidhs_cohort_hip import AidhsCohort, AidhsSubject
from aidhs.data_preprocessing import Preprocess, Feature

new_site_code = 'H99'

site_combat_path = os.path.join(BASE_PATH,'distributed_combat')
print(site_combat_path)
if not os.path.isdir(site_combat_path):
    os.makedirs(site_combat_path)

site_codes = ['H1', 'H11','H16','H29']
c_combat =  AidhsCohort(hdf5_file_root='{site_code}_{group}_featurematrix_combat_avg.hdf5',  dataset='dataset_C-P-DC_alltrain.csv',
                      data_dir=BASE_PATH)


preprocessor=Preprocess(c_combat, site_codes=site_codes)

#load in precombat data
ref_subject_ids = c_combat.get_subject_ids(site_codes=site_codes, lesional_only=False)

# initialise base features
base_features=[
    # '.label-{}.curvature.sm1',
    # '.label-{}.gauss-curv_filtered_sm1',
    # '.label-{}.gyrification.sm1',
    # '.label-{}.thickness.sm1',
    '.label-avg.hippunfold_volume_icvcorr',
    ]
feat = Feature()
features_smooth_avg = [feature.format('avg') for feature in base_features]
features_combat_avg = [feat.combat_feat(feature) for feature in features_smooth_avg]
features_asym_avg = [feat.asym_feat(feature) for feature in features_combat_avg]


for fi,feature in enumerate(features_smooth_avg):
    print("harmonising :", feature)
    # find adapted mask
    if 'label-dentate' in feature:
        mask = c_combat.dentate_mask
    elif 'label-avg' in feature:
        mask = c_combat.avg_mask
    else:
        mask = c_combat.hippo_mask
    #load cohort
    precombat_features=[]
    combat_subject_include = np.zeros(len(ref_subject_ids), dtype=bool)
    new_site_codes=np.zeros(len(ref_subject_ids))
    print('loading')
    for k, subject in enumerate(ref_subject_ids):
        # get the reference index and cohort object for the site, 0 whole cohort, 1 new cohort
        site_code_index = new_site_codes[k]
        subj = AidhsSubject(subject, cohort=c_combat)
        # exclude outliers and subject without feature
        if subj.has_features(features_combat_avg[fi]):
            lh = subj.load_feature_values(features_combat_avg[fi], hemi="lh")[mask]
            rh = subj.load_feature_values(features_combat_avg[fi], hemi="rh")[mask]
            combined_hemis = np.hstack([lh, rh])
            precombat_features.append(combined_hemis)
            combat_subject_include[k] = True
        else:
            combat_subject_include[k] = False

    #load covars
    precombat_features = np.array(precombat_features).T
    covars = preprocessor.load_covars(ref_subject_ids)
    covars = covars[combat_subject_include].copy().reset_index()
    N=len(covars)
    bat = pd.Series(pd.Categorical(np.repeat('H0', N), categories=['H0', new_site_code]))
    covars['site_scanner']=bat
    covars = covars[['ages','sex','group','site_scanner']]

    print('calculating')

    #DO COMBAT steps
    #use var estimates from basic combat
    com_out = nc.neuroCombat(precombat_features, covars, 'site_scanner')
    with open(os.path.join(site_combat_path,f'combat_{feature}_var.pickle'), "wb") as f:
        pickle.dump(com_out['estimates']['var.pooled'], f)
    #calculate reference estimates for distributed combat
    _ = dc.distributedCombat_site(precombat_features, bat, covars[['ages','sex','group']], 
                              file=os.path.join(site_combat_path,
                                                f'combat_{feature}.pickle'), ref_batch = 'H0')