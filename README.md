# AID-HS

Automated and Interpretable Detection of Hippocampal Sclerosis

AID-HS 
1) Extracts hippocampal volume- and surface-based features from T1w 3T MRI scans
2) Compares patient hippocampal morphology to normative growth charts generated from healthy controls
3) Compares patient's left and right hippocampi
4) Runs a logistic regression classifier to automatically detect and lateralise hippocampal sclerosis (HS)
5) Outputs an interpretable report 
For more details please read our [preprint](https://www.medrxiv.org/content/10.1101/2023.10.13.23296991v1)

Note: 
- AID-HS works on T1w scans, and has only been tested at 3T
- You will need demographic information (age at scan, sex) to run AID-HS on your patients. 

Pipeline overview:\
<img src="images/overview_pipeline.jpg " height="500" />

## Disclaimer

The AID-HS software is intended for research purposes only and has not been reviewed or approved by the Medicines and Healthcare products Regulatory Agency (MHRA), European Medicine Agency (EMA) or by any other agency. Any clinical application of the software is at the sole risk of the party engaged in such application. There is no warranty of any kind that the software will produce useful results in any way. Use of the software is at the recipient's own risk.

## Installation

To use AID-HS you will need to install the prerequesites below and create the AID-HS environment.

Notes:
- The current installation has been tested on Ubuntu 18.04
- **COMING SOON**: AID-HS will be packaged as a Docker/Singularity package. This will be the recommended installation method.

### Prerequisites

- We use **Anaconda** to manage the environment and dependencies. Please follow instructions to [install Anaconda](https://docs.anaconda.com/anaconda/install).
- AID-HS extracts volume- and surface-based features of the hippocampus using **HippUnfold**. Please follow instructions to [install HippUnfold Singularity container](https://hippunfold.readthedocs.io/en/latest/getting_started/installation.html).
- AID-HS uses **Workbench Connectom** to create additional surface-based features. Please follow instructions to [install Workbench Connectom](https://www.humanconnectome.org/software/get-connectome-workbench).
- AID-HS extracts total intracranial volume using **FastSurfer**. Please follow instructions for [native installation of Fastsurfer](https://github.com/Deep-MI/FastSurfer.git). Note that Fastsurfer requires to install **Freesurfer V7.2**. Please follow instructions to [install Freesurfer](https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall)

You will need to ensure that Freesurfer, FastSurfer and Hippunfold are activated in your terminal by running : 
```bash
export FREESURFER_HOME=<freesurfer_installation_directory>
source $FREESURFER_HOME/SetUpFreeSurfer.sh
export FASTSURFER_HOME=<fastsurfer_installation_directory>
export SINGULARITY_BINDPATH=/home:/home
export HIPPUNFOLD_PATH=<hippunfold_installation_directory>

```
with `<freesurfer_installation_directory>`, `<fastsurfer_installation_directory>` and `<hippunfold_installation_directory>` being the paths to where your Freesurfer, FastSurfer and HippUnfold softwares have been installed.

### Software & environment installation

Run the commands below in your terminal to download the AID-HS code and create the conda environment

``` bash
# get the AID-HS software from GitHub
git clone https://github.com/mathrip/AID-HS.git 
# enter the aid_hs directory
cd aidhs
# create the aidhs_env environment with all the dependencies 
conda env create -f environment.yml
# activate the environment
conda activate aidhs_env
# install aid_hs package with pip (with `-e`, the development mode, to allow changes in the code to be immediately visible in the installation)
pip install -e .
```

Install the additional **hippunfold_toolbox** package on the aidhs_env environment:
``` bash
# get the hippunfold_toolbox from GitHub
git clone https://github.com/jordandekraker/hippunfold_toolbox.git
# enter the hippunfold_toolbox directory
cd hippunfold_toolbox
# install hippunfold_toolbox package with pip
pip install -e .
```

## Usage
With this software you can detect and lateralise HS from T1w MRI scans

### Prior to running the pipeline: get your site code
AID-HS uses [DistributedCombat](https://www.sciencedirect.com/science/article/pii/S1053811921010934?via%3Dihub) to remove site-scanner bias differences. Thus, prior to running AID-HS on a patient's MRI scan, you will need to do a first step of *harmonisation* to compute the parameters for the scanner used to acquire the MRI data. Each scanner will have a *site_code* that will be needed to organise your data and run the code as detailled below.

To get a *site_code* please contact *m.ripart@ucl.ac.uk* and don't forget to mention your institution and provide us with an email address. 

### First step: prepare your data

You will need to prepare your data following a specific architecture:
- 1. Download the aidhs_data_folder at https://figshare.com/s/16011ee4d6b5723b14b6
- 2. Unzip the folder where you want to store the aidhs_data_folder
- 3. Follow the guidelines below to prepare your MRI data and demographic information

#### MRI data
AID-HS runs on 3D T1w MRI scans **acquired at 3T**. The T1w scans will need to be saved in the `input` folder as a BIDS format detailed as below:
- Each subject should have a folder `sub-<subject_ID>`
- In this subject folder should be an `anat` folder
- In the anat folder should be a nifti T1w scan with the name `sub-<subject_ID>_ses-preop_T1.nii.gz`

You will find an example of the folder architecture for subject *H1P0003* on the `aidsh_data_folder\input` folder and as illustrated below:\
<img src="images/example_input_folder_subject.png " height="80" />

Notes: 
- Please ensure that the nifti scan is following the **BIDS standard** and is a compressed format (**gzip**).
- AID-HS has been developped on **3D T1w scans acquired at 3T**. We cannot guarantee robustness and accuracy of the results on 2D scans nor scans acquired at lower (1.5T) or higher (7T) magnetic strenghs. 

#### demographic data

AID-HS provides individualised results, which are adapted for the age and sex of the patients. Thus, you will need to fill the `demographics_file.csv`  file with:
- **ID**: subject ID
- **Site**: site_code
- **Scanner**:  '3T' (mandatory as AID-HS does not work on other scanners)
- **Patient or Control**: patient = 1, control = 0
- **Age at preoperative**: age at time of acquisition , in years
- **Sex**: male = 0, female = 1

You will find an example of the demographics_file.csv for subject *H1P0003* in the `aidhs_data_folder` and illustrated below:\
<img src="images/example_demographic_file_subject.png " height="50" />

### Intermediate step: Harmonisation (to do only once)

If this is the first time you are using AID-HS or if your data come from a new scanner, you will need to compute the *harmonisation* parameters for the scanner used to acquire your patient's data. This step only needs to be done once, for each new scanner you might use for prediction. 

To do so you will need:
- T1w scans from at least 20 subjects (controls and/or patients) for that scanner
- A 'site_code'. Please contact `m.ripart@ucl.ac.uk` to get a site code 
- Have organised your MRI data and your demographics information (`demographics_file.csv`) following the instructions above for these 20 subjects
- Have prepared a csv file (`list_subjects.csv`) containing the list of subjects ID used for the harmonisation

You will find an example of the `list_subjects.csv` on the `aidsh_data_folder`

Before running the harmonisation command,  please ensure you are in the folder containing the AID-HS scripts and that the *aidhs_env* environment is activated:
```bash
cd <path_to_aidhs_folder>
conda activate aidhs_env
```

To harmonise run the command:
```bash
python scripts/preprocess/new_patient_pipeline.py -site <site_code> -ids <path_to_list_subject_ids> --parallelise --harmo_only
```

This will compute the harmonisation parameters and store them so that they can be used for new prediction. As this process relies on segmenting the brain and the hippocampus, it can take up to 1h per subject. 

### Final step: Prediction

To predict on a subject you will need 
- The T1w scan of the subject you want to predict on
- To have organised your the MRI data and demographics following the instructions above
- The ID of the subject (`subject_id`)
- To have run the harmonisation for the `site_code` that corresponds to the subject (see above)

Before running the prediction command, ensure you are in the folder containing the AID-HS scripts and that the aidhs environment is activated:
```bash
cd <path_to_aidhs_folder>
conda activate aidhs_env
```

To predict run the command:
```bash
python scripts/preprocess/new_patient_pipeline.py -site <site_code> -id <subject_id>
```

## Looking at the outputs

AID-HS outputs individualised and interpretable reports that can be found at: 
`<aidhs_data_folder>/output/prediction_reports/<subject_id>/Report_<subject_id>.pdf`

These reports contain:
- **Hippocampal segmentation** & **Hippocampal pial surfaces**: HippUnfold segmentations and surface reconstructions for left and right hippocampi, alongside automated quality control scores to highlight subjects in which the segmentation might have failed. We recommend manually checking segmentations with dice scores below 0.70.
- **Individual hippocampal features vs normative trajectories**: Left and right hippocampal features mapped against normative growth charts.
- **Asymmetries**: Feature asymmetries that indicate the magnitude and direction of asymmetries, and compared to abnormality thresholds. 
- **Automated detection & lateralisation**: Detection and lateralisation scores from the AID-HS classifier.

An example of the report for patient *H1P0003* can be found [here](images/Report_H1P0003.pdf) with interpretation below.

In this example, the automated quality control scores of 0.79 and 0.81 for both left and right hippocampi indicate good quality hippocampal segmentations. Compared with the normative growth charts, the left hippocampus features fall within the normal range of the healthy population, while the right hippocampus had features that fell outside the 5th and 95th percentiles. In the asymmetry analysis, abnormalities are lateralised to the right hippocampus, with significant reductions in volume, thickness and gyrification, alongside increased curvature and intrinsic curvature. These findings are further supported by the automated classifier results, which indicate right hippocampal sclerosis with a predicted probability of 88.2%. 


## Manuscript
Please check out our preprint [manuscript](https://www.medrxiv.org/content/10.1101/2023.10.13.23296991v1) to learn more.

An overview of the notebooks that we used to create the figures can be found [here](figure_notebooks.md).

## Contacts

Mathilde Ripart, PhD student, UCL  \
`m.ripart@ucl.ac.uk` 








