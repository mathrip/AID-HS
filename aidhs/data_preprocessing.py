from aidhs.paths import (
    DEMOGRAPHIC_FEATURES_FILE,
    HIPPUNFOLD_SUBJECTS_PATH,
    FS_SUBJECTS_PATH,
    FS_STATS_FILE,
    BASE_PATH,
    PARAMS_PATH
    
)
import sys
import pandas as pd
import numpy as np
import os
import h5py
import csv
import logging
import random
import json
import pickle
import aidhs.mesh_tools as mt
from aidhs.aidhs_cohort_hip import AidhsSubject
from neuroCombat import neuroCombat, neuroCombatFromTraining
import aidhs.distributedCombat as dc


class Preprocess:
    def __init__(self, cohort, site_codes=None, write_hdf5_file_root=None, data_dir=BASE_PATH, params_dir=PARAMS_PATH):
        self.log = logging.getLogger(__name__)
        self.cohort = cohort
        self.data_dir = data_dir
        self.params_dir = params_dir
        self._covars = None
        self.feat = Feature()
        self.write_hdf5_file_root = write_hdf5_file_root
        self.site_codes = site_codes
        # filter subject ids based on site codes
        if self.site_codes is None:
            self.site_codes = self.cohort.get_sites()
        self.subject_ids = self.cohort.get_subject_ids(site_codes=self.site_codes, lesional_only=False)
        # calibration_smoothing: curve to calibrate on surface mesh
        self._calibration_smoothing = None
        return

    def save_cohort_features(self, feature_name, features, subject_ids, hemis=['lh','rh']):
        assert len(features) == len(subject_ids)
        for s, subject in enumerate(subject_ids):
            subj = AidhsSubject(subject, cohort=self.cohort)
            subj.write_feature_values( feature_name,
                                             features[s], hemis=hemis,
                                             hdf5_file_root=self.write_hdf5_file_root)
    

    def calibration_smoothing(self, feature):
        """caliration curve for smoothing surface mesh'"""
        # find adapted mask
        if 'label-dentate' in feature:
            surf = self.cohort.surf_dentate
            start_v=200
            label='label-dentate'
        else:
            surf = self.cohort.surf
            start_v=5000
            label='label-hipp'
        if self._calibration_smoothing is None:
            line, model = mt.calibrate_smoothing(surf['coords'], surf['faces'], start_v=start_v, label=label)
            self._calibration_smoothing=(line,model)
        return self._calibration_smoothing

    def clip_data_with_NaN(self, vals, params):
        """ clip data with NaN and then fillnanvertices to remove very extreme feature values """
        min_p = float(params[0])
        max_p = float(params[1])
        num = (vals < min_p).sum() + (vals > max_p).sum()
        vals[(vals<min_p)] = np.nan
        vals[(vals>max_p)] = np.nan
        return vals, num

    def fillnanvertices(self,V, neighbours):
        """
        Fills NaNs by iteratively computing the mean of the non-nan nearest neighbours until no NaNs remain. 
        """
        import copy 
        Vnew = copy.deepcopy(V)
        while np.isnan(np.sum(Vnew)):
            # index of vertices containing nan
            vrows = np.unique(np.where(np.isnan(Vnew))[0])
            # replace with the nanmean of neighbouring vertices
            for n in vrows:
                Vnew[n] = np.nanmean(Vnew[neighbours[n]], 0)
        return Vnew

    def smooth_data(self, feature, fwhm, clipping_params=None, outliers_file=None):
        """ smooth features with given fwhm for all subject and save in new hdf5 file"""
        # create smooth name
        feature_smooth = self.feat.smooth_feat(feature, fwhm)
        # find adapted mask
        if 'label-dentate' in feature:
            label='label-dentate'
            neighbours = self.cohort.neighbours_dentate
            mask = self.cohort.dentate_mask 
        else:
            label='label-hipp'
            neighbours = self.cohort.neighbours
            mask = self.cohort.hippo_mask 
        #initialise
        if clipping_params!=None:
            print(f'Clip data to remove very extreme values using {clipping_params}')
            with open(os.path.join(clipping_params), "r") as f:
                params = json.loads(f.read())
        else:
            params = None
        subject_include = []
        vals_matrix_lh=[]
        vals_matrix_rh=[]
        for id_sub in self.subject_ids:
            self.log.info(id_sub)
            # create subject object<
            subj = AidhsSubject(id_sub, cohort=self.cohort)
            # smooth data only if the feature exist
            if subj.has_features(feature):
                # load feature's value for this subject
                vals_lh = subj.load_feature_values(feature, hemi='lh')
                vals_rh = subj.load_feature_values(feature, hemi='rh')
                #harmonise curvature from hippunfold because left opposite right
                print(id_sub)
                if ('curvature' in feature) or ('mean-curv' in feature):
                    print('clean curv')
                    vals_lh = -1* vals_lh
                # clip data to remove outliers vertices
                if params!=None:
                    #clip data with NaN and then interpolate to fill NaN
                    vals_lh, num_lh = self.clip_data_with_NaN(vals_lh, params[feature])
                    vals_rh, num_rh = self.clip_data_with_NaN(vals_rh, params[feature])
                    if (num_lh>0) or (num_rh>0):
                        print(f'WARNING: subject:{id_sub} - feature: {feature} - {num_lh + num_rh} extremes vertices')
                        header_name = ['subject', 'feature', 'num vertices outliers left', 'num vertices outliers right']
                        if outliers_file!=None:
                            need_header=False
                            if not os.path.isfile(outliers_file):
                                need_header=True
                            with open(outliers_file, 'a') as f:
                                writer = csv.writer(f)
                                if need_header:
                                    writer.writerow(header_name)
                                writer.writerow([id_sub, feature, num_lh, num_rh])
                    vals_lh = self.fillnanvertices(vals_lh, neighbours)
                    vals_rh = self.fillnanvertices(vals_rh, neighbours)
                vals_matrix_lh.append(vals_lh)
                vals_matrix_rh.append(vals_rh)
                subject_include.append(id_sub)
            else:
                self.log.info('feature {} does not exist for subject {}'.format(feature,id_sub))              
        # smoothed data if fwhm
        vals_matrix_lh= np.array(vals_matrix_lh)
        vals_matrix_rh= np.array(vals_matrix_rh)
        if fwhm:
            #find number iteration from calibration smoothing
            x,y = self.calibration_smoothing(feature)
            idx = (np.abs(y- fwhm)).argmin()
            n_iter=int(np.round(x[idx]))
            print(f"smoothing with {n_iter} iterations ...")
            vals_matrix_lh = mt.smooth_array(vals_matrix_lh.T, neighbours ,n_iter=n_iter,
                                                         cortex_mask=mask)
            vals_matrix_rh = mt.smooth_array(vals_matrix_rh.T, neighbours ,n_iter=n_iter,
                                                         cortex_mask=mask)
        else:
            print("no smoothing")
            vals_matrix_lh = vals_matrix_lh.T
            vals_matrix_rh = vals_matrix_rh.T
            
        smooth_vals_hemis=np.array(
                 np.hstack([vals_matrix_lh[mask].T, vals_matrix_rh[mask].T])
                )
        #write features in hdf5
        print('Smoothing finished \n Saving data')
        self.save_cohort_features(feature_smooth,
                                      smooth_vals_hemis,
                                      np.array(subject_include))
     
    @property
    def covars(self):
        if self._covars is None:
            self._covars = self.load_covars()
        return self._covars

    def load_covars(self, subject_ids=None, demographic_file=DEMOGRAPHIC_FEATURES_FILE):
        if not os.path.isfile(demographic_file):
            demographic_file = os.path.join(self.data_dir,demographic_file)
        if subject_ids is None:
            subject_ids = self.subject_ids
        covars = pd.DataFrame()
        ages = []
        sex = []
        group = []
        sites_scanners = []
        for subject in subject_ids:
            subj = AidhsSubject(subject, cohort=self.cohort)
            a, s = subj.get_demographic_features(["Age at preop", "Sex"], csv_file = demographic_file)
            ages.append(a)
            sex.append(s)
            group.append(subj.is_patient)
            sites_scanners.append(subj.site_code + "_" + subj.scanner)

        covars["ages"] = ages
        covars["sex"] = sex
        covars["group"] = group
        covars["site_scanner"] = sites_scanners
        covars["ID"] = subject_ids

        #clean missing values in demographics
        covars["ages"] = covars.groupby("site_scanner").transform(lambda x: x.fillna(x.mean()))["ages"]
        covars["sex"] = covars.groupby("site_scanner").transform(lambda x: x.fillna(random.choice([0, 1])))["sex"]
        return covars
    
    def remove_isolated_subs(self, covars, precombat_features):
        """remove subjects where they are sole examples from the site (for FLAIR)"""

        df = pd.DataFrame(covars.groupby("site_scanner").count()["ages"])
        single_subject_sites = list(df.index[covars.groupby("site_scanner").count()["ages"] == 1])
        mask = np.zeros(len(covars)).astype(bool)
        for site_scan in single_subject_sites:
            mask += covars.site_scanner == site_scan
        precombat_features = precombat_features[~mask]
        covars = covars[~mask]
        return covars, precombat_features

    def compute_avg_feature(self, feature):
        """
        Average the pervertex to single value in hippocampi
        """
        # load in features using cohort + subject class
        subject_include = np.zeros(len(self.subject_ids), dtype=bool)
        vals_avg = []
        for k, subject in enumerate(self.subject_ids):
            subj = AidhsSubject(subject, cohort=self.cohort)
            # exclude outliers and subject without feature
            vals_lh = subj.combine_DG_hippo(feature, hemi="lh")
            vals_rh = subj.combine_DG_hippo(feature, hemi="rh")
            if (vals_lh.sum()!=0) & (vals_rh.sum()!=0):
                if 'surfarea' in feature:
                    vals_avg_lh = np.sum(vals_lh)
                    vals_avg_rh = np.sum(vals_rh)
                else:
                    vals_avg_lh = np.mean(vals_lh)
                    vals_avg_rh = np.mean(vals_rh)
                vals_avg.append(np.hstack([vals_avg_lh, vals_avg_rh]))
                subject_include[k] = True
            else:
                print("exclude")
                subject_include[k] = False
        if vals_avg:
            vals_avg = np.array(vals_avg)  
            self.save_cohort_features(feature.format('avg'), vals_avg, np.array(self.subject_ids)[np.array(subject_include)])
        else:
            print('no data to save')
            pass

    def extract_volumes_avg(self, feature):
        """
        Extract average volumes from Freesurfer or Hippunfold
        """
        # load in features using cohort + subject class
        subject_include = np.zeros(len(self.subject_ids), dtype=bool)
        vals = []
        for k, subject in enumerate(self.subject_ids):
            #extract volumes
            if 'FS' in feature:
                vals_lh = extract_volume_freesurfer(os.path.join(FS_SUBJECTS_PATH, 'sub-'+subject), hemi='lh')
                vals_rh = extract_volume_freesurfer(os.path.join(FS_SUBJECTS_PATH, 'sub-'+subject), hemi='rh')
            elif 'hippunfold' in feature:
                vals_lh = extract_volume_hippunfold(os.path.join(HIPPUNFOLD_SUBJECTS_PATH, 'hippunfold',f'sub-{subject}','anat', f'sub-{subject}_'), hemi='lh')
                vals_rh = extract_volume_hippunfold(os.path.join(HIPPUNFOLD_SUBJECTS_PATH, 'hippunfold',f'sub-{subject}','anat', f'sub-{subject}_'), hemi='rh')
            else:
                return
            if (vals_lh!=0) & (vals_rh!=0):
                vals.append(np.hstack([vals_lh, vals_rh]))
                subject_include[k] = True
            else:
                print("exclude")
                subject_include[k] = False
        if vals:
            vals = np.array(vals)  
            self.save_cohort_features(feature, vals, np.array(self.subject_ids)[np.array(subject_include)])
        else:
            print('no data to save')
            pass
    
    def compute_parameters_correction_volume_ICV(self, feature, cohort, params_icv_correct=None):
        """
        Compute thh parameters for correcting hippocampal volume with intracranial volume 
        """
        from sklearn.linear_model import LinearRegression
        controls_ids = cohort.get_subject_ids(group="control")
        # Give warning if list of controls empty
        if len(controls_ids) == 0:
            print("WARNING: there is no controls in this cohort to do inter-normalisation")
        vals_array = []
        icv_array=[]
        included_subj = []
        for id_sub in controls_ids:
            # create subject object
            subj = AidhsSubject(id_sub, cohort=cohort)
            # append data to compute mean and std if feature exist
            if subj.has_features(feature):
                included_subj.append(True)
                # load feature's value for this subject
                vals_lh = subj.load_feature_values(feature, hemi="lh")
                vals_rh = subj.load_feature_values(feature, hemi="rh")
                vals_array.append(np.array(np.hstack([vals_lh, vals_rh])))
                # load ICV volume for this subject
                id_FS, folder_FS = subj.get_demographic_features(["Freesurfer_ids", "Freesurfer_folder"])
                icv = extract_totalbrainvolume_freesurfer(os.path.join(folder_FS, id_FS))
                icv_array.append(np.hstack([icv, icv]))
            # Calculate asymmetry
            else:
                included_subj.append(False)
                pass
        vals_array=np.array(vals_array)
        icv_array=np.array(icv_array)
        included_subj=np.array(included_subj)
        print("Use {} control cohort to compute parameters for intracranial corection".format(included_subj.sum()))
        params = {}
        m=[]
        mean_icv = []
        for h, hemi in enumerate(['lh','rh']):
            X = icv_array[included_subj, h].reshape(-1, 1)
            y = vals_array[included_subj, h]
            reg = LinearRegression().fit(X, y)
            # get mean and std from controls
            m.append(reg.coef_[0])
            mean_icv.append(X.mean())
        params['m'] = np.array(m)
        params['mean_icv'] = np.array(mean_icv)  
        # save parameters in json
        if params_icv_correct!=None:
            self.save_norm_combat_parameters(feature, params, params_icv_correct)
        return params

    def correct_volume_ICV(self, feature, params_icv_correct=None):
        """
        Correct hippocampal volume from intracranial volume 
        """
        from sklearn.linear_model import LinearRegression
        feature_corrected = self.feat.icvcorr_feat(feature)
        # loop over subjects
        vals_array = []
        icv_array = []
        included_subjects=np.zeros(len(self.subject_ids),dtype=bool)
        for k,id_sub in enumerate(self.subject_ids):
            # create subject object
            subj = AidhsSubject(id_sub, cohort=self.cohort)
            if subj.has_features(feature):
                included_subjects[k] = True
                # load feature's value for this subject
                vals_lh = subj.load_feature_values(feature, hemi="lh")
                vals_rh = subj.load_feature_values(feature, hemi="rh")
                vals_array.append(np.array(np.hstack([vals_lh, vals_rh])))
                # load ICV volume for this subject
                icv = extract_totalbrainvolume_freesurfer(os.path.join(FS_SUBJECTS_PATH, 'sub-'+id_sub))
                icv_array.append(np.hstack([icv, icv]))
            else:
                print('exlude subject {}'.format(id_sub))
                included_subjects[k] = False
                
        vals_array=np.array(vals_array)
        icv_array=np.array(icv_array)
        # remove exclude subjects
        included_subjects = np.array(self.subject_ids)[included_subjects]
        if params_icv_correct is not None:
            print(f'Use parameters for ICV correction from {params_icv_correct}')
            params = self.read_norm_combat_parameters(feature, params_icv_correct)
            mean_icv = params['mean_icv']
            m = params['m']
        else:
            print("Need to compute the parameters for the ICV correction first")
            sys.exit()
        vals_corrected = vals_array - m*(icv_array-mean_icv)
        # save subject
        print('Correction by ICV finished \n Saving data')
        self.save_cohort_features(feature_corrected, vals_corrected, included_subjects) 

    def shrink_combat_estimates(self, estimates):
        """ shrink combat estimates to reduce size file"""
        #combined mod.mean with stand.mean
        stand_mean =  estimates['stand.mean'][:, 0] + estimates['mod.mean'].mean(axis=1)
        estimates['stand.mean'] = stand_mean
        #save the number of subjects to un-shrink later
        estimates['num_subjects']= np.array([estimates['mod.mean'].shape[1]])
        #remove mod.mean to reduce estimates size
        del estimates['mod.mean']
        return estimates

    def unshrink_combat_estimates(self, estimates):
        """ unshrink combat estimates to use as input in neuroCombatFromTraining"""
        num_subjects = estimates['num_subjects'][0]
        mod_mean = np.zeros((len(estimates['stand.mean']),num_subjects ))
        estimates['mod.mean'] = mod_mean
        estimates['stand.mean'] = np.tile(estimates['stand.mean'], (num_subjects,1)).T
        return estimates

    def save_norm_combat_parameters(self, feature, estimates, hdf5_file):
        """Save estimates from combat and normalisation parameters on hdf5"""
        if not os.path.isfile(hdf5_file):
            hdf5_file_context = h5py.File(hdf5_file, "a")
        else:
            hdf5_file_context = h5py.File(hdf5_file, "r+")

        with hdf5_file_context as f:
            list_params = list(set(estimates))
            for parameter_name in list_params:
                parameter = estimates[parameter_name]
                parameter = np.array(parameter)
                dtype = parameter.dtype
                dtype = parameter.dtype

                group = f.require_group(feature)
                if dtype == "O":
                    dset = group.require_dataset(
                        parameter_name, shape=np.shape(parameter), dtype="S10", compression="gzip", compression_opts=9
                    )
                    dset.attrs["values"] = list(parameter)
                else:
                    dset = group.require_dataset(
                        parameter_name, shape=np.shape(parameter), dtype=dtype, compression="gzip", compression_opts=9
                    )
                    dset[:] = parameter

    def read_norm_combat_parameters(self, feature, hdf5_file):
        """reconstruct estimates dictionnary from the combat parameters hdf5 file"""
        hdf5_file_context = h5py.File(hdf5_file, "r")
        estimates = {}
        with hdf5_file_context as f:
            feat_dir = f[feature]
            parameters = feat_dir.keys()
            for param in parameters:
                if feat_dir[param].dtype == "S10":
                    estimates[param] = feat_dir[param].attrs["values"].astype(np.str)
                else:
                    estimates[param] = feat_dir[param][:]
        return estimates

    def combat_whole_cohort(self, feature_name, outliers_file=None, combat_params_file=None):
        """Harmonise data between site/scanner with age, sex and disease status as covariate
        using neuroComBat (Fortin et al., 2018, Neuroimage) and save in hdf5
        Args:
            feature_name (str): name of the feature, usually smoothed data.
            outliers_file (str): file name of the csv containing subject ID to exclude from harmonisation

        Returns:
            estimates : Combat parameters used for the harmonisation. Need to save for new patient harmonisation.
            info : dictionary of information from combat
        """
        # read morphological outliers from cohort.
        if outliers_file is not None:
            outliers = list(pd.read_csv(os.path.join(self.data_dir, outliers_file), header=0)["ID"])
        else:
            outliers = []
        # find adapted mask
        if 'label-dentate' in feature_name:
            mask = self.cohort.dentate_mask
        elif 'label-avg' in feature_name:
            mask = self.cohort.avg_mask
        else:
            mask = self.cohort.hippo_mask 
        # load in features using cohort + subject class
        combat_subject_include = np.zeros(len(self.subject_ids), dtype=bool)
        precombat_features = []
        for k, subject in enumerate(self.subject_ids):
            subj = AidhsSubject(subject, cohort=self.cohort)
            # exclude outliers and subject without feature
            if (subj.has_features(feature_name)) & (subject not in outliers):
                lh = subj.load_feature_values(feature_name, hemi="lh")[mask]
                rh = subj.load_feature_values(feature_name, hemi="rh")[mask]
                combined_hemis = np.hstack([lh, rh])
                precombat_features.append(combined_hemis)
                combat_subject_include[k] = True
            else:
                print("exclude")
                combat_subject_include[k] = False
        if precombat_features:
            precombat_features = np.array(precombat_features)
            # load in covariates - age, sex, group, site and scanner unless provided
            covars = self.covars[combat_subject_include].copy()
            # check for nan
            index_nan = pd.isnull(covars).any(1).to_numpy().nonzero()[0]
            if len(index_nan) != 0:
                print(
                    "There is missing information in the covariates for subjects {}. \
                Combat aborted".format(
                        np.array(covars["ID"])[index_nan]
                    )
                )
            else:
                # function to check for single subjects
                covars, precombat_features = self.remove_isolated_subs(covars, precombat_features)
                covars = covars.reset_index(drop=True)

                if combat_params_file is not None:
                    dict_combat = neuroCombat(
                        precombat_features.T,
                        covars,
                        batch_col="site_scanner",
                        categorical_cols=["sex", "group"],
                        continuous_cols="ages",
                    )
                    #
                    save_data = dict_combat["data"].T
                    # save combat parameters
                    shrink_estimates = self.shrink_combat_estimates(dict_combat["estimates"])
                    self.save_norm_combat_parameters(feature_name, shrink_estimates, combat_params_file)
                else:
                    #do not apply combat
                    save_data = precombat_features
                
                print("Combat finished \n Saving data")
                post_combat_feature_name = self.feat.combat_feat(feature_name)
                self.save_cohort_features(post_combat_feature_name, save_data, np.array(covars["ID"]))
        else:
            print('no data to combat harmonised')
            pass

    def get_combat_new_site_parameters(self,feature, demographic_file,):
        """Harmonise new site data to post-combat whole cohort and save combat parameters in
        new hdf5 file. 
        Args:
            feature_name (str): name of the feature

        """
        # find adapted mask
        if 'label-dentate' in feature:
            mask = self.cohort.dentate_mask
        elif 'label-avg' in feature:
            mask = self.cohort.avg_mask
        else:
            mask = self.cohort.hippo_mask
        site_code=self.site_codes[0]
        site_combat_path = os.path.join(self.data_dir,f'AIDHS_{site_code}','distributed_combat')
        if not os.path.isdir(site_combat_path):
            os.makedirs(site_combat_path)
        aidhs_combat_path = os.path.join(self.params_dir,'distributed_combat')
        listids = self.subject_ids
        print(listids)
        site_codes = np.zeros(len(listids))
        precombat_features=[]
        combat_subject_include = np.zeros(len(listids), dtype=bool)
        demos=[]
        for k, subject in enumerate(listids):
            # get the reference index and cohort object for the site, 0 whole cohort, 1 new cohort
            site_code_index = site_codes[k]
            subj = AidhsSubject(subject, cohort=self.cohort)
            # exclude outliers and subject without feature
            if (subj.has_features(feature)) :
                lh = subj.load_feature_values(feature, hemi="lh")[mask]
                rh = subj.load_feature_values(feature, hemi="rh")[mask]
                combined_hemis = np.hstack([lh, rh])
                precombat_features.append(combined_hemis)
                combat_subject_include[k] = True
            else:
                combat_subject_include[k] = False
              
        if len(np.array(listids)[np.array(combat_subject_include)])==0:
            print(f'Cannot compute harmonisation for {feature} because no subject found with this feature')
            return
        # load in covariates - age, sex, group, site and scanner unless provided    
        new_site_covars = self.load_covars(subject_ids=np.array(listids)[np.array(combat_subject_include)], demographic_file=demographic_file).copy()
        #check site_scanner codes are the same for all subjects
        if len(new_site_covars['site_scanner'].unique())==1:
            site_scanner = new_site_covars['site_scanner'].unique()[0]
        else:
            print('Subjects on the list come from different site or scanner.\
            Make sure all your subject come from same site and scanner for the harmonisation process')
            sys.exit()
        bat = pd.Series(pd.Categorical(np.array(new_site_covars['site_scanner']),
                                       categories=['H0', site_scanner]))       
        # apply distributed combat
        print('step1')
        new_site_data = np.array(precombat_features).T 
        dc.distributedCombat_site(new_site_data,
                                  bat, 
                                  new_site_covars[['ages','sex','group']], 
                                  file=os.path.join(site_combat_path,f"{site_code}_{feature}_summary.pickle"), 
                              ref_batch = 'H0', 
                              robust=True,)
        print('step2')
        dc_out = dc.distributedCombat_central(
            [os.path.join(aidhs_combat_path,f'combat_{feature}.pickle'),
             os.path.join(site_combat_path,f"{site_code}_{feature}_summary.pickle")], ref_batch = 'H0'
        )
        # third, use variance estimates from full AIDHS cohort
        dc_out['var_pooled'] = pd.read_pickle(os.path.join(aidhs_combat_path,f'combat_{feature}_var.pickle')).ravel()
        for c in ['ages','sex','group']:
            new_site_covars[c]=new_site_covars[c].astype(np.float64)      
        print('step3')
        pickle_file = os.path.join(site_combat_path,f"{site_code}_{feature}_harmonisation_params_test.pickle")
        _=dc.distributedCombat_site(
            pd.DataFrame(new_site_data), bat, new_site_covars[['ages','sex','group']], 
            file=pickle_file,
             central_out=dc_out, 
            ref_batch = 'H0', 
            robust=True,
        )
        #open pickle, shrink estimates and save in hdf5 and delete pickle
        with open(pickle_file, 'rb') as f:
            params = pickle.load(f)
        #filter name keys
        target_dict = {'batch':'batches', 'delta_star':'delta.star', 'var_pooled':'var.pooled',
           'gamma_star':'gamma.star', 'stand_mean':'stand.mean', 'mod_mean': 'mod.mean', 
           'parametric': 'del', 'eb':'del', 'mean_only':'del', 'mod':'del', 'ref_batch':'del', 'beta_hat':'del', 
          }
        estimates = params['estimates'].copy()
        for key in target_dict.keys():  
            if target_dict[key]=='del':
                estimates.pop(key)
            else:
                estimates[target_dict[key]] = estimates.pop(key)
        for key in estimates.keys():
            if key in ['a_prior', 'b_prior', 't2', 'gamma_bar']:
                estimates[key]=[estimates[key]]
            if key == 'batches':
                estimates[key]=np.array([estimates[key][0]]).astype('object')
            if key=='var.pooled':
                estimates[key]=estimates[key][:,np.newaxis]
            if key in ['gamma.star', 'delta.star']:
                estimates[key]=estimates[key][np.newaxis,:]
            estimates[key] = np.array(estimates[key])
        #shrink estimates
        shrink_estimates = self.shrink_combat_estimates(estimates)
        #save estimates and delete pickle file
        combat_params_file=os.path.join(self.data_dir, self.write_hdf5_file_root.format(site_code=site_code))
        self.save_norm_combat_parameters(feature, shrink_estimates, combat_params_file)
        os.remove(pickle_file)
        pickle_file = os.path.join(site_combat_path,f"{site_code}_{feature}_summary.pickle")
        os.remove(pickle_file)
        return estimates, shrink_estimates

    def combat_new_subject(self, feature_name, combat_params_file):
        """Harmonise new subject data with Combat parameters from whole cohort
            and save in new hdf5 file
        Args:
            subjects (list of str): list of subjects ID to harmonise
            feature_name (str): name of the feature, usually smoothed data.
            combat_estimates (arrays): combat parameters used for the harmonisation
        """
        # find adapted mask
        if 'label-dentate' in feature_name:
            mask = self.cohort.dentate_mask
        elif 'label-avg' in feature_name:
            mask = self.cohort.avg_mask
        else:
            mask = self.cohort.hippo_mask
        # load combat parameters        
        precombat_features = []
        site_scanner = []
        subjects_included=[]
        for subject in self.subject_ids:
            subj = AidhsSubject(subject, cohort=self.cohort)
            if subj.has_features(feature_name):
                lh = subj.load_feature_values(feature_name, hemi="lh")[mask]
                rh = subj.load_feature_values(feature_name, hemi="rh")[mask]
                combined_hemis = np.hstack([lh, rh])
                precombat_features.append(combined_hemis)
                site_scanner.append(subj.site_code + "_" + subj.scanner)
                subjects_included.append(subject)
        #if matrix empty, pass
        if precombat_features:
            combat_estimates = self.read_norm_combat_parameters(feature_name, combat_params_file)
            combat_estimates = self.unshrink_combat_estimates(combat_estimates)
            precombat_features = np.array(precombat_features)
            site_scanner = np.array(site_scanner)
            dict_combat = neuroCombatFromTraining(dat=precombat_features.T, batch=site_scanner, estimates=combat_estimates)
            post_combat_feature_name = self.feat.combat_feat(feature_name)
            print("Combat finished \n Saving data")
            self.save_cohort_features(post_combat_feature_name, dict_combat["data"].T, np.array(subjects_included))
        else:
            print('No data to combat harmonised')
            pass


    def compute_mean_std_controls(self, feature, cohort, asym=False, params_norm=None):
        """retrieve controls from given cohort, intra-normalise feature and return mean and std for inter-normalisation"""
        # find adapted mask
        if 'label-dentate' in feature:
            mask = cohort.dentate_mask 
        elif 'label-avg' in feature:
            mask = cohort.avg_mask
        else:
            mask = cohort.hippo_mask
        controls_ids = cohort.get_subject_ids(group="control")
        # Give warning if list of controls empty
        if len(controls_ids) == 0:
            print("WARNING: there is no controls in this cohort to do inter-normalisation")
        vals_array = []
        included_subj = []
        for id_sub in controls_ids:
            # create subject object
            subj = AidhsSubject(id_sub, cohort=cohort)
            # append data to compute mean and std if feature exist
            if subj.has_features(feature):
                # load feature's value for this subject
                vals_lh = subj.load_feature_values(feature, hemi="lh")
                vals_rh = subj.load_feature_values(feature, hemi="rh")
                vals = np.array(np.hstack([vals_lh[mask], vals_rh[mask]]))
                # # intra subject normalisation asym
                # intra_norm = np.array(self.normalise(vals))
                intra_norm=np.array(vals) # no intranormalisation
                # Calculate asymmetry
                if asym == True:
                    intra_norm = self.compute_asym(intra_norm)
                    names_save = [f'mean.asym',f'std.asym']
                else:
                    names_save = [f'mean',f'std']   
                vals_array.append(intra_norm)
                included_subj.append(id_sub)
            else:
                pass
        print("Compute mean and std from {} controls".format(len(included_subj)))
        # get mean and std from controls
        params = {}
        params[names_save[0]] = np.mean(vals_array, axis=0)
        params[names_save[1]] = np.std(vals_array, axis=0)
        # save parameters in hdf5
        if params_norm!=None:
            self.save_norm_combat_parameters(feature, params, params_norm)
        return params[names_save[0]], params[names_save[1]]
    

    def normalise(self, data):
        if len(data.shape) == 1:
            data[:, np.newaxis]
        mean_intra = np.mean(data, axis=0)
        std_intra = np.std(data, axis=0)
        intra_norm = (data - mean_intra) / std_intra
        return intra_norm


    def compute_asym(self, intra_norm):
        intra_lh = intra_norm[: int(len(intra_norm) / 2)]
        intra_rh = intra_norm[int(len(intra_norm) / 2) :]
        lh_asym = 2*(intra_lh - intra_rh)/(intra_lh + intra_rh)
        rh_asym = 2*(intra_rh - intra_lh)/(intra_lh + intra_rh)
        asym = np.hstack([lh_asym, rh_asym])
        if np.isinf(asym).any():
            print('inf here')
        return asym


    def asymmetry_internorm_subject(self, feature, cohort_for_norm=None, params_norm=None):
        """ inter-normalisation (between subjects relative to controls) and asymetry between hemispheres"""
        # find adapted mask
        if 'label-dentate' in feature:
            mask = self.cohort.dentate_mask
        elif 'label-avg' in feature:
            mask = self.cohort.avg_mask
        else:
            mask = self.cohort.hippo_mask
        feature_asym=self.feat.asym_inter_feat(feature)
        # loop over subjects
        vals_asym_array = []
        included_subjects=np.zeros(len(self.subject_ids),dtype=bool)
        for k,id_sub in enumerate(self.subject_ids):
            # create subject object
            subj = AidhsSubject(id_sub, cohort=self.cohort)
            if subj.has_features(feature):
                included_subjects[k] = True             
                # load feature's value for this subject
                vals_lh = subj.load_feature_values(feature, hemi="lh")
                vals_rh = subj.load_feature_values(feature, hemi="rh")
                vals = np.array(np.hstack([vals_lh[mask], vals_rh[mask]]))
                # intra subject normalisation asym
                intra_norm=vals                                # no intra normalisation
                # Calculate asymmetry
                vals_asym = self.compute_asym(intra_norm)
                vals_asym_array.append(vals_asym)
            else:
                print('exlude subject {}'.format(id_sub))
                included_subjects[k] = False       
        vals_asym_array=np.array(vals_asym_array)
        # remove exclude subjects
        included_subjects = np.array(self.subject_ids)[included_subjects]
        #normalise by controls
        if cohort_for_norm is not None:
            print('Use other cohort for normalisation')
            mean_c, std_c = self.compute_mean_std_controls(feature, cohort=cohort_for_norm, asym=True)
        else:
            if params_norm is not None:
                print(f'Use normalisation parameter from {params_norm}')
                params = self.read_norm_combat_parameters(feature, params_norm)
                mean_c = params['mean.asym']
                std_c = params['std.asym']
            else:
                print("Need to compute mean and std from control first")
                sys.exit()
        asym_combat = (vals_asym_array - mean_c) / std_c
        # save subject
        print('Asym finished \n Saving data')
        self.save_cohort_features(feature_asym, asym_combat, included_subjects)

    
def extract_volume_hippunfold(path,hemi="lh", DGexcluded=False):
    # extract number of voxels in subfields of hippocampus from hippunfold segmentation 
    data = pd.read_csv(path+'space-cropT1w_desc-subfields_atlas-bigbrain_volumes.tsv',sep='\t')    
    if hemi=='lh':
        hemi='L'
    elif hemi=='rh':
        hemi='R'
    else:
        hemi=np.nan
    if DGexcluded: 
        return data[data['hemi']==hemi][['Sub','CA1','CA2','CA3','CA4']].values[0].sum()
    else:
        return data[data['hemi']==hemi][['Sub','CA1','CA2','CA3','CA4','DG']].values[0].sum()


def extract_volume_freesurfer(path, hemi="lh",):
    # extract hippocampal volume in mm3 from freesurfer segmentation 
    if hemi=='lh':
        hemi='Left'
    elif hemi=='rh':
        hemi='Right'
    else:
        hemi=np.nan
    volume_file = os.path.join(path, FS_STATS_FILE)
    with open(volume_file,"r") as fp:
        for line in fp:
            if line[0] == '#':
                pass
            elif hemi+'-Hippocampus' in line:
                split = line.split(' ')
                split = list(filter(None, split))
            else:
                pass   
    return float(split[3])

def extract_totalbrainvolume_freesurfer(path,):
    # extract intracranial volume from freesurfer segmentation 
    volume_file = os.path.join(path, FS_STATS_FILE)
    with open(volume_file,"r") as fp:
        for line in fp:
            if  'Measure EstimatedTotalIntraCranialVol' in line:
                split = line.split(',')
                split = list(filter(None, split))
            else:
                pass   
    return int(float(split[3]))


class Feature:
    def __init__(self):
        """ Class to define feature name """
        pass

    def raw_feat(self, feature):
        self._raw_feat = feature
        return self._raw_feat

    def smooth_feat(self, feature, smoother=None):
        if smoother != None:
            smooth_part = "sm" + str(int(smoother))
            self._smooth_feat = ".".join([feature, smooth_part])
        else:
            self._smooth_feat = feature
        return self._smooth_feat
    
    def icvcorr_feat(self, feature):
        feat_split = feature.split('.')
        if "sm" in feat_split[-1]:
            feat_split[-2] = feat_split[-2]+'_icvcorr'
            self._icvcorr_feat = ".".join(feat_split)
        else:
            feat_split[-1] = feat_split[-1]+'_icvcorr'
            self._icvcorr_feat = ".".join(feat_split)
        return self._icvcorr_feat

    def combat_feat(self, feature):
        self._combat_feat = "".join([".combat", feature])
        return self._combat_feat 

    def norm_inter_feat(self, feature):
        self._norm_feat = "".join([".inter_z", feature])
        return self._norm_feat
    
    def norm_intra_feat(self, feature):
        self._norm_feat = "".join([".intra_z", feature])
        return self._norm_feat
    
    def norm_intra_inter_feat(self, feature):
        self._norm_feat = "".join([".inter_z.intra_z", feature])
        return self._norm_feat

    def asym_feat(self, feature):
        self._asym_feat = "".join([".asym", feature])
        return self._asym_feat
    
    def asym_inter_feat(self, feature):
        self._asym_feat = "".join([".inter_z.asym", feature])
        return self._asym_feat

    def list_feat(self):
        self._list_feat = [self.smooth, self.combat, self.norm, self.asym]
        return self._list_feat