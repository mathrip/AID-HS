import os
import sys
from glob import glob
from os.path import join as opj
import subprocess
from subprocess import Popen, DEVNULL, STDOUT, check_call
import threading
import multiprocessing
from functools import partial
from aidhs.tools_print import get_m


def init(lock):
    global starting
    starting = lock

def check_FS_outputs(folder):
    fname = opj(folder,'stats',f'aparc.DKTatlas+aseg.deep.volume.stats')
    if not os.path.isfile(fname):
        return False
    else:
        return True

def fastsurfer_subject(subject, fs_dir, verbose=False):
    # run fastsurfer segmentation on 1 subject

    subject_id = subject.bids_id
    subject_t1_input = subject.t1_input
    fs_s = os.path.join(fs_dir, subject_id)
    
    # if freesurfer outputs already exist for this subject, skip segmentation
    if os.path.isdir(fs_s):
        if check_FS_outputs(fs_s):
            print(get_m(f'Fastsurfer outputs already exists. Fastsurfer will be skipped', subject_id, 'STEP 1'))
            return True
        else:
            print(get_m(f'Fastsurfer outputs already exists but is incomplete. Delete folder {fs_s} and reran', subject_id, 'ERROR'))
            return False
    else:
        pass 

    # select inputs files T1 
    if subject_t1_input != None:
        print(get_m(f'Start segmentation using T1 only with FastSurfer (up to 10min). Please wait', subject_id, 'INFO'))
        
        # setup cortical segmentation command
        command = format(
            "$FASTSURFER_HOME/run_fastsurfer.sh --sd {} --sid {} --t1 {} --seg_only --vol_segstats --parallel --batch 1 --run_viewagg_on gpu".format(fs_dir, subject_id, subject_t1_input)
        )
        if verbose:
            print(command)
        # call fastsurfer
        print(f"INFO: Results will be stored in {fs_dir}")
        starting.acquire()  # no other process can get it until it is released
        proc = Popen(command, shell=True, stdout = subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')  
        threading.Timer(120, starting.release).start()  # release in two minutes
        stdout, stderr= proc.communicate()
        if verbose:
            print(stdout)
        if proc.returncode==0:
            print(get_m(f'Finished cortical segmenttaion', subject_id, 'INFO'))
            return True
        else:
            print(get_m(f'Cortical segmentation using fastsurfer failed. Please check the log at {fs_s}/scripts/recon-surf.log', subject_id, 'ERROR'))
            print(get_m(f'COMMAND failing : {command} with error {stderr}', subject_id, 'ERROR'))
            return False
    else:
        print(get_m(f'T1 does not exist. Segmentation cancelled for that subject', subject_id, 'ERROR'))
        return False

def run_segmentation_parallel(subjects, fs_dir, num_procs=10, verbose=False):
    # parallel version of the pipeline, finish each stage for all subjects first
    ini_freesurfer = format("$FREESURFER_HOME/SetUpFreeSurfer.sh")
    check_call(ini_freesurfer, shell=True, stdout = DEVNULL, stderr=STDOUT)

    ## Make a directory for the outputs
    os.makedirs(fs_dir, exist_ok=True)

    subject_ids=[subject.id for subject in subjects]

    #launch segmentation multiprocessing
    pool = multiprocessing.Pool(processes=num_procs, initializer=init, initargs=[multiprocessing.Lock()])
    subject_ids_failed=[]
    for i,result in enumerate(pool.imap(partial(fastsurfer_subject, fs_dir=fs_dir, verbose=verbose), subjects)):
        if result==False:
            print(get_m(f'Subject removed from futur process because a step in the pipeline failed', subject_ids[i], 'ERROR'))
            subject_ids_failed.append(subject_ids[i])
        else:
            pass
    #return list of subjects that did not fail
    subject_ids = list(set(subject_ids).difference(subject_ids_failed))
    return subject_ids

def run_segmentation(subject, fs_dir, verbose=False):
    # pipeline to segment the brain, exract surface-based features and smooth features for 1 subject
    ini_freesurfer = format("$FREESURFER_HOME/SetUpFreeSurfer.sh")
    check_call(ini_freesurfer, shell=True, stdout = DEVNULL, stderr=STDOUT)

    ## Make a directory for the outputs
    os.makedirs(fs_dir, exist_ok=True)

    ## first processing stage with fastsurfer: segmentation
    init(multiprocessing.Lock())
    result = fastsurfer_subject(subject,fs_dir, verbose=verbose)
    if result == False:
        return False
        

def run_hippunfold_parallel(subjects, bids_dir=None, hippo_dir=None, num_procs=10, verbose=False):
    # parallel version of Hippunfold

    #make a directory for the outputs
    os.makedirs(hippo_dir, exist_ok=True)

    subjects_to_run = []
    for subject in subjects:
        hippo_s = subject.hippo_dir
        subject_bids_id = subject.bids_id

        if subject_bids_id != None:
            subject_id = subject_bids_id
        else:
            subject_id = subject.convert_bids_id()
        subject_id = subject_id.split('sub-')[-1]

        #check if outputs already exists
        files_surf = glob(f'{hippo_s}/surf/*_den-0p5mm_label-hipp_*.surf.gii')

        if files_surf==[]:
            subjects_to_run.append(subject_id)
        else:
            print(get_m(f'Hippunfold outputs already exists. Hippunfold will be skipped', subject_id, 'INFO'))
    
    if subjects_to_run!=[]:
        print(get_m(f'Start Hippunfold segmentation in parallel for {subjects_to_run}', None, 'INFO'))
        command =  format(f"$HIPPUNFOLD_PATH/khanlab_hippunfold_latest.sif {bids_dir} {hippo_dir} participant --participant-label {' '.join(subjects_to_run)} --core {num_procs} --modality T1w")
        if verbose:
            print(command)
        proc = Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
        stdout, stderr= proc.communicate()
        if verbose:
            print(stdout)
        if proc.returncode==0:
            print(get_m(f'Finished hippunfold segmentation for {subjects_to_run}', None, 'INFO'))
            return True
        else:
            print(get_m(f'Hippunfold segmentation failed for 1 of the subject. Please check the logs at {hippo_dir}/logs/<subject_id>', None, 'ERROR'))
            print(get_m(f'COMMAND failing : {command} with error {stderr}', None, 'ERROR'))
            return False

def run_hippunfold(subject, bids_dir=None, hippo_dir=None, verbose=False):

    hippo_s = subject.hippo_dir
    subject_bids_id = subject.bids_id

    if subject_bids_id != None:
        subject_id = subject_bids_id
    else:
        subject_id = subject.convert_bids_id()
    subject_id = subject_id.split('sub-')[-1]

    #make a directory for the outputs
    os.makedirs(hippo_dir, exist_ok=True)

    #check if outputs already exists
    files_surf = glob(f'{hippo_s}/surf/*_den-0p5mm_label-hipp_*.surf.gii')
    if files_surf==[]:
        print(get_m(f'Start Hippunfold segmentation', subject_id, 'INFO'))
        command =  format(f"$HIPPUNFOLD_PATH/khanlab_hippunfold_latest.sif {bids_dir} {hippo_dir} participant --participant-label {subject_id} --core 3 --modality T1w")
        if verbose:
            print(command)
        proc = Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
        stdout, stderr= proc.communicate()
        if verbose:
            print(stdout)
        if proc.returncode==0:
            print(get_m(f'Finished hippunfold segmentation', subject_id, 'INFO'))
            return True
        else:
            print(get_m(f'Hippunfold segmentation failed. Please check the log at {hippo_dir}/logs/{subject_id}', subject_id, 'ERROR'))
            print(get_m(f'COMMAND failing : {command} with error {stderr}', subject_id, 'ERROR'))
            return False
    else:
        print(get_m(f'Hippunfold outputs already exists. Hippunfold will be skipped', subject_id, 'INFO'))
    

    

    
