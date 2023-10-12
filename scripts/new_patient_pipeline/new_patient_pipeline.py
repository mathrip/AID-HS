import os
import argparse
import sys
import time

from scripts.new_patient_pipeline.run_pipeline_segmentation import run_pipeline_segmentation
from scripts.preprocess.extract_features_hdf5 import extract_features_hdf5
from scripts.new_patient_pipeline.run_pipeline_preprocessing import run_pipeline_preprocessing
from scripts.new_patient_pipeline.run_pipeline_prediction import run_pipeline_prediction
from aidhs.tools_print import get_m
from aidhs.paths import DATA_PATH, BASE_PATH, FS_SUBJECTS_PATH, HIPPUNFOLD_SUBJECTS_PATH, BIDS_SUBJECTS_PATH

class Logger(object):
    def __init__(self, sys_type=sys.stdout, filename='AIDHS_output.log'):
        self.terminal = sys_type
        self.filename = filename
        self.log = open(self.filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        pass

if __name__ == "__main__":
    # parse commandline arguments
    parser = argparse.ArgumentParser(description="Main pipeline to predict on subject with AIDHS classifier")
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
    parser.add_argument("--parallelise", 
                        help="parallelise segmentation", 
                        required=False,
                        default=False,
                        action="store_true",
                        )
    parser.add_argument('-demos', '--demographic_file', 
                        type=str, 
                        help='provide the demographic files for the harmonisation',
                        required=False,
                        default=None,
                        )
    parser.add_argument('--harmo_only', 
                        action="store_true", 
                        help='only compute the harmonisation combat parameters, no further process',
                        required=False,
                        default=False,
                        )
    parser.add_argument('--skip_segmentation',
                        action="store_true",
                        help='Skip the segmentation and extraction of the AIDHS features',
                        )
    parser.add_argument("--debug_mode", 
                        help="mode to debug error", 
                        required=False,
                        default=False,
                        action="store_true",
                        )
    

     
    #write terminal output in a log
    file_path=os.path.join(os.path.abspath(os.getcwd()), 'AIDHS_pipeline_'+time.strftime('%Y-%m-%d-%H-%M-%S') + '.log')
    sys.stdout = Logger(sys.stdout,file_path)
    sys.stderr = Logger(sys.stderr, file_path)
    
    args = parser.parse_args()
    print(args)
    
    #---------------------------------------------------------------------------------
    ### CHECKS
    if (args.harmo_only) & (args.list_ids == None):
        print('ERROR: Please provide a list of subjects for the harmonisation. We recommend 20 subjects')
        os.sys.exit(-1)

    #---------------------------------------------------------------------------------
    ### SEGMENTATION ###

    if not args.skip_segmentation:
        print(get_m(f'Call script segmentation', None, 'SCRIPT 1'))
        result = run_pipeline_segmentation(
                        list_ids=args.list_ids,
                        sub_id=args.id, 
                        input_dir= os.path.join(DATA_PATH, 'input'),
                        fs_dir=FS_SUBJECTS_PATH,
                        bids_dir=BIDS_SUBJECTS_PATH,
                        hippo_dir=HIPPUNFOLD_SUBJECTS_PATH, 
                        use_parallel=args.parallelise, 
                        skip_fs=False,
                        verbose=args.debug_mode
                        )

        if result == False:
            print(get_m(f'Segmentation and feature extraction has failed at least for one subject. See log at {file_path}. Consider fixing errors or excluding these subjects before re-running the pipeline. Segmentation will be skipped for subjects already processed', None, 'SCRIPT 1'))    
            sys.exit()
    else:
        print(get_m(f'Skip script segmentation', None, 'SCRIPT 1'))

    
    #---------------------------------------------------------------------------------
    ### EXTRACT FEATURES ###
    
    print(get_m(f'Call script preprocessing', None, 'SCRIPT 2'))
    extract_features_hdf5(list_ids=args.list_ids, 
                            sub_id=args.id, 
                            data_dir=DATA_PATH, 
                            output_dir=BASE_PATH)

    #---------------------------------------------------------------------------------
    ### PREPROCESSING ###
    
    print(get_m(f'Call script preprocessing', None, 'SCRIPT 2'))
    run_pipeline_preprocessing(
                    site_code=args.site_code,
                    list_ids=args.list_ids,
                    sub_id=args.id,
                    harmonisation_only = args.harmo_only,
                    verbose = args.debug_mode,
                    )
            
    #---------------------------------------------------------------------------------
    ### PREDICTION ###
    
    if not args.harmo_only:
        print(get_m(f'Call script prediction', None, 'SCRIPT 3'))
        result = run_pipeline_prediction(
                    list_ids=args.list_ids,
                    sub_id=args.id,
                    verbose = args.debug_mode,
                    )
        if result == False:
            print(get_m(f'Prediction and creating report has failed at least for one subject. See log at {file_path}. Consider fixing errors or excluding these subjects before re-running the pipeline. Segmentation will be skipped for subjects already processed', None, 'SCRIPT 3'))    
            sys.exit()
    else:
        print(get_m(f'Skip script predition', None, 'SCRIPT 3'))
                
    print(f'You can find a log of the pipeline at {file_path}')