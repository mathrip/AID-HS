import os
import sys
import argparse
import numpy as np
from glob import glob
from os.path import join as opj
import pandas as pd

from run_script_segmentation import run_hippunfold_parallel, run_hippunfold, run_segmentation_parallel, run_segmentation
from run_script_dataprep import prepare_T1_parallel ,prepare_T1, bidsify_results, extract_surface_features


def init(lock):
    global starting
    starting = lock

class SubjectSeg:
    def __init__(self, subject_id, input_dir=None, fs_dir=None, bids_dir=None, hippo_dir=None, bids_id=None):
        
        #initialise
        self.id=subject_id
        
        #update bids id
        self.bids_id = self.convert_bids_id(bids_id=bids_id)
        #update directories
        self.get_input_dir(input_dir)
        self.get_fs_dir(fs_dir)
        self.get_bids_dir(bids_dir)
        self.get_hippo_dir(hippo_dir)
   
        #update inputs path 
        if self.input_dir!=None:
           self.t1_input = self.get_inputs_volumes_path(modality='T1', input_dir=input_dir)
           
        #update bids path
        if self.bids_dir!=None:
           self.t1_bids = self.get_bids_volumes_path(modality='T1w', bids_dir=bids_dir)


    #functions
    # update id with bids compatible
    def convert_bids_id(self, bids_id=None):
        if bids_id == None:
            bids_id = self.id
        #clean id
        list_exclude = ['{','}','_']
        for l in list_exclude:
            if l in bids_id:
                bids_id = bids_id.replace(l, '')
        #add 'sub' if needed  
        if not 'sub-' in bids_id:
            bids_id = 'sub-'+bids_id
        print(f'INFO: subject {self.id} converted in {bids_id}')
        #TODO: write in a csv new id and old id?
        self.bids_id = bids_id
        return self.bids_id
    
    #update with input directory
    def get_input_dir(self, input_dir):
        if input_dir != None:
            self.input_dir = opj(input_dir, self.id)
        else:
            print('WARNING: no input directory provided or existing')
            self.input_dir = None
        return self.input_dir

    #update with FS directory
    def get_fs_dir(self, fs_dir):
        if fs_dir != None:
            self.fs_dir = opj(fs_dir, self.id)
        else:
            print('WARNING: no FS directory provided or existing')
            self.fs_dir = None
        return self.fs_dir

    #update with BIDS directory
    def get_bids_dir(self, bids_dir, bids_id=None):
        if bids_id ==  None:
            bids_id = self.id
        if bids_dir != None:
            self.bids_dir = opj(bids_dir, bids_id)
        else:
            print('WARNING: no BIDS directory provided or existing')
            self.bids_dir = None
        return self.bids_dir

    #update with hippunfold directory
    def get_hippo_dir(self, hippo_dir):
        if hippo_dir != None:
            self.hippo_dir = opj(hippo_dir, 'hippunfold', self.bids_id)   
        else:
            print('WARNING: no hippunfold directory provided or existing')
            self.hippo_dir = None
        return self.hippo_dir

    #find inputs volumes paths
    def get_inputs_volumes_path(self, modality='T1',input_dir=None):
        if input_dir == None :
            input_dir= self.input_dir
        if input_dir == None:
            print(f'ERROR: No input directory provided to find {modality} volume')
            return
        if not os.path.isdir(input_dir):
            print(f'ERROR: Input directory provided does not exist at {input_dir}')
        else:    
            subject_input_dir= self.get_input_dir(input_dir)  
            print(subject_input_dir)
            print(subject_input_dir)
            #find modality
            try:
                glob(opj(subject_input_dir, 'anat', f'*preop_{modality}*.nii*'))[0]
                subject_path = glob(opj(subject_input_dir, 'anat', f'*preop_{modality}*.nii*'))
                print(subject_path)
            except:
                subject_path = glob(opj(subject_input_dir, 'anat', f'*preop_{modality}*.nii*'))
            if len(subject_path) > 1:
                raise FileNotFoundError(
                    f"Find too much volumes for {modality}. Check and remove the additional volumes with same key name"
                )
            elif not subject_path:
                print(f"No {modality} file has been found for {self.id}")
                subject_path = None
            else:
                subject_path = subject_path[0]
            return subject_path

    #find bids volumes paths
    def get_bids_volumes_path(self, modality='T1w',bids_dir=None):
        if bids_dir == None :
            bids_dir= self.bids_dir
        if bids_dir == None:
            print(f'ERROR: No input directory provided to find {modality} volume')
            return
        bids_id = self.bids_id
        subject_bids_dir= self.get_bids_dir(bids_dir, bids_id)
        #define file
        if modality=='DTI':
            mod_type='dwi'
        else:
            mod_type='anat'
        subject_path = opj(subject_bids_dir, mod_type, f'{bids_id}_{modality}.nii.gz')
        return subject_path

def run_pipeline(list_ids=None, sub_id=None, input_dir=None, fs_dir=None, bids_dir=None, hippo_dir=None, 
                use_parallel=False, use_fastsurfer=False, pial_flair=True, skip_segmentation=False):
    subject_id=None
    subject_ids=None
    if list_ids != None:
        try:
            sub_list_df=pd.read_csv(list_ids)
            subject_ids=np.array(sub_list_df.ID.values)
            try: 
                subject_bids_ids=np.array(sub_list_df.bids_ID.values)
                print(subject_bids_ids)
            except:
                subject_bids_ids=np.full(len(subject_ids), None)
        except:
            subject_ids=np.array(np.loadtxt(list_ids, dtype='str', ndmin=1)) 
            subject_bids_ids=np.full(len(subject_ids), None)            
    elif sub_id != None:
        subject_id=sub_id
        subject_ids=np.array([sub_id])
        subject_bids_ids=np.full(len(subject_ids), None) 
    else:
        print('ERROR: No ids were provided')
        print("ERROR: Please specify subject(s) ...")
        sys.exit(-1) 
        
    subjects=[]
    print('here')
    for i, subject_id in enumerate(np.array(subject_ids)):
        print(subject_id)
        print(subject_bids_ids[i])
        subjects.append(SubjectSeg(subject_id, input_dir=input_dir, fs_dir=fs_dir, bids_dir=bids_dir, hippo_dir=hippo_dir, bids_id=subject_bids_ids[i]))
    if use_parallel:
        #launch segmentation and feature extraction in parallel
        print('INFO: Run subjects in parallel') 
        if not skip_segmentation:
            print('STEP 1: Neocortical segmentation')
            run_segmentation_parallel(subjects, fs_dir=fs_dir, use_fastsurfer = use_fastsurfer, pial_flair=pial_flair)
        else:
            print('STEP 1: Skip neocortical segmentation')
        # #prepare T1
        # print('STEP 2: Prepare T1')
        # prepare_T1_parallel(subjects)
        # #extract surface based features
        print('STEP 3: Run hippunfold segmentation')
        run_hippunfold_parallel(subjects, bids_dir=bids_dir, hippo_dir=hippo_dir)
        print('STEP 4: Extract hippocampal surface features')
        for subject in subjects:
           extract_surface_features(subject, output_dir=bids_dir)
    else:
        #launch segmentation and feature extraction for each subject one after another
        print('INFO: No parralelisation. Run subjects one after another')
        for subject in subjects:
            #run FS segmentation
            if not skip_segmentation:
                print('STEP 1: Neocortical segmentation')
                run_segmentation(subject, fs_dir=fs_dir, use_fastsurfer = use_fastsurfer, pial_flair=pial_flair)
            else:
                print('STEP 1: Skip neocortical segmentation')
            # #prepare T1
            # print('STEP 2a: Prepare T1')
            # prepare_T1(subject)
            # #run hippunfold segmentation
            # print('STEP 3: Run hippunfold segmentation')
            # run_hippunfold(subject, bids_dir=bids_dir, hippo_dir=hippo_dir)
            # #extract surface based features
            # print('STEP 4: Extract hippocampal surface features')
            # extract_surface_features(subject, output_dir=bids_dir)
        

if __name__ == "__main__":
    # parse commandline arguments
    parser = argparse.ArgumentParser(description="perform cortical parcellation using recon-all from freesurfer")
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
    parser.add_argument("-in_dir","--input_dir",
                        default=None,
                        help="Directory containing inputs file such as T1 and FLAIR",
                        required=False,
                        )
    parser.add_argument("-fs_dir","--fs_dir",
                        default=None,
                        help="Folder to store FS outputs",
                        required=False,
                        )
    parser.add_argument("-bids_dir","--bids_dir",
                        default=None,
                        help="Directory to save outputs files in bids format for hippunfold",
                        required=False,
                        )
    parser.add_argument("-hip_dir","--hippo_dir",
                        default=None,
                        help="Directory to save hippunfold outputs files in bids format for hippunfold",
                        required=False,
                        )
    parser.add_argument("--fastsurfer", 
                        help="use fastsurfer instead of freesurfer", 
                        required=False, 
                        default=False,
                        action="store_true",
                        )
    parser.add_argument("--parallelise", 
                        help="parallelise segmentation", 
                        required=False,
                        default=False,
                        action="store_true",
                        )
    parser.add_argument("--pialFLAIR", 
                        help="improve pial surface with FLAIR", 
                        required=False, 
                        default=False,
                        action="store_true",
                        )
    parser.add_argument("--skip_fs", 
                        help="skip the segmentation with freesurfer if already done", 
                        required=False, 
                        default=False,
                        action="store_true",
                        )
    
    args = parser.parse_args()
    print(args)

    run_pipeline(list_ids=args.list_ids,
                sub_id=args.id, 
                input_dir=args.input_dir,
                fs_dir=args.fs_dir,
                bids_dir=args.bids_dir,
                hippo_dir=args.hippo_dir, 
                use_parallel=args.parallelise, 
                use_fastsurfer=args.fastsurfer,
                pial_flair=args.pialFLAIR,
                skip_segmentation=args.skip_fs,
                        )
    



    
