
import os
import sys
import argparse
import numpy as np
import nibabel as nb
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
from datetime import date
from fpdf import FPDF
import subprocess
import pickle
from PIL import Image
import tempfile
from sklearn.utils.extmath import softmax

from aidhs.aidhs_cohort_hip import AidhsCohort, AidhsSubject
from aidhs.paths import BASE_PATH
from aidhs.train_evaluate import create_dataset_file, predict_subjects


class PDF(FPDF):    
    def lines(self):
        self.set_line_width(0.0)
        self.line(5.0,5.0,205.0,5.0) # top one
        self.line(5.0,292.0,205.0,292.0) # bottom one
        self.line(5.0,5.0,5.0,292.0) # left one
        self.line(205.0,5.0,205.0,292.0) # right one
    
    def custom_header(self, txt1, txt2=None):
        # Arial bold 
        self.set_font('Arial', 'B', 30)
        # Move to the right
        self.cell(80)
        # Title
        self.cell(w=30, h=10, txt=txt1, border=0, ln=0, align='C')
        if txt2 != None:
            # Arial bold 15
            self.ln(20)
            self.cell(80)
            self.set_font('Arial', 'B', 20)
            self.cell(w=30, h=5, txt=txt2, border=0, ln=0, align='C')
        # Line break
        self.ln(20)
    
    def custom_footer(self, txt):
        # Arial italic 8
        self.set_font('Arial', 'I', 8)
        # Position at 1.5 cm from bottom
        self.set_y(-30)
        # add text
        self.cell(w=0, h=0, txt=txt, border=0, ln=2, align='C') 
        # Date
        today = date.today()
        today = today.strftime("%d/%m/%Y")
        self.cell(w=5, h=0, txt=str(today) , border=0, ln=0, align='L')
        # Page number
        self.cell(w=180, h=0, txt='Page ' + str(self.page_no()), border=0, ln=2, align='R')
                
    
    def info_box(self, txt):
        # set font
        self.set_font('Arial', 'I', 10)
        #set box color
        self.set_fill_color(160,214,190)
        # add texte box info
        self.multi_cell(w=190, h=5, txt=txt , border=1, align='L', fill=True)
        
    def info_box_chart(self, txt):
        # set font
        self.set_font('Arial', 'I', 6)
        #set box color
        self.set_fill_color(160,214,190)
        # add text box info
        self.multi_cell(w=80, h=5, txt=txt , border=1, align='L', fill=True)
    
    def disclaimer_box(self, txt):
        # set font
        self.set_font('Arial', 'I', 8)
        #set box color
        self.set_fill_color(240,128,128)
        # add texte box info
        self.multi_cell(w=190, h=5, txt=txt , border=1, align='L', fill=True)

    def info_box_prediction(self, txt):
        self.ln(140)
        # set font
        self.set_font('Arial', 'I', 6)
        #set box color
        self.set_fill_color(160,214,190)
        # add text box info
        self.multi_cell(w=80, h=5, txt=txt , border=1, align='L', fill=True)

    def subtitles(self, txt):
        # Arial bold 15
        self.set_font('Arial', 'B', 20)
        # Title
        self.cell(w=80, h=80, txt=txt, border=0, ln=2, align='L')
         
    def imagey(self,im, y, w=190):
        self.image(im, 5, y, link='', type='', w=w, h=297/3)

def return_features_title(feature,features_title):
    for feat in set(features_title):
        if feat in feature:
            return features_title[feat]        

def plot_controls_chart(ax, data_c, feature, color = 'green', cmap=False, fill_color=True, label=None):
    '''
    Plot normative trajectories (GAM curves) of healthy population 
    '''
    if cmap != False:
        cmap = matplotlib.cm.get_cmap(cmap)

    features_title = {
                '.curvature' : 'mean curvature',
                '.gauss-curv_filtered': 'intrinsic curvature ',
                '.gyrification' : 'gyrification ',
                '.thickness' : 'thickness (mm)',
                '.hippunfold_volume' :'volume (mm$^3$)',
    }

    #plot percentiles 
    percentiles = np.sort(list(set(data_c['predict_vals_intervals'])))
    for p, percentile in enumerate(percentiles):
        if cmap != False:
            color = cmap(abs(0.5-percentile))
        if percentile == 0.5:
            ax.plot(data_c['age_range'],data_c['predict_vals_intervals'][percentile], color=color, 
                        ls='-', linewidth= 1, alpha=1, label = label)
        else:
            ax.plot(data_c['age_range'],data_c['predict_vals_intervals'][percentile], color=color, 
                        ls='--',linewidth= 1,  alpha=1)
        ax.text(data_c['age_range'][-1], data_c['predict_vals_intervals'][percentile][-1], f'{int(percentile*100)}th',
                    fontsize = 12, color=color)   
    
    #fill between line
    if fill_color:
        if len(percentiles)>1:
            bands = np.delete(percentiles, np.where(percentiles == 0.5))
            colors = np.linspace(0,1,len(bands)-1)-0.5
            for b, band in enumerate(bands[0:-1]):
                color = cmap(abs(colors[b]))
                ax.fill_between(data_c['age_range'], data_c['predict_vals_intervals'][band], data_c['predict_vals_intervals'][bands[b+1]], 
                                        facecolor=color, alpha=0.5, interpolate=True)

    #details plot 
    ax.set_xlim([data_c['age_range'][0], data_c['age_range'][-1]])
    ax.tick_params(which='both', width=2, labelsize=13)
    ax.set_ylabel(return_features_title(feature, features_title), fontsize=18)
    ax.set_xlabel('age (years)', fontsize=18)

def plot_patient_on_controls_charts(subj, features, controls_GAM, filename):
    '''
    Plot patient left and right features on normative trajectories
    '''
    #get subject info
    age_scan_p, sex_p= subj.get_demographic_features(["Age at preoperative", "Sex"])

    fig = plt.figure(figsize=(15,9))
    gs1 = GridSpec(2, 3, wspace=0.5, hspace=0.2)
    axs = []

    i=0
    for feature in features:
        axs.append(fig.add_subplot(gs1[i]))
            
        #plot controls chart
        data_c = controls_GAM[feature][sex_p]
        plot_controls_chart(axs[i], data_c, feature,  cmap='Greens_r', fill_color=True)
            
        #plot patient in chart
        for (hemi,color,label) in zip(['lh','rh'], ['navy','deeppink'],['Left hippocampus','Right hippocampus']):
            vals = subj.load_feature_values(feature, hemi)[0]
            axs[i].scatter(age_scan_p, vals, c=color,s=10, zorder=3, label=label)
            axs[i].text(age_scan_p+1, vals, f'{label[0:5]}',fontsize = 12, color=color)
        i=i+1
        
    axs.append(fig.add_subplot(gs1[i]))  
    h, l = axs[0].get_legend_handles_labels() # get labels and handles from ax1
    axs[i].legend(h, l, loc='upper left', fontsize='18')
    axs[i].axis('off')
    plt.tight_layout()
    #save plot
    fig.savefig(filename, bbox_inches="tight")


def plot_abnormalities_direction_subject(subj, features, filename):
    '''
    Plot magnitude and direction of asymmetries in subject
    '''
    
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot()
    
    abnormal_AI_direction_feat={
    '.inter_z.asym.combat.label-avg.curvature.sm1':1,
    '.inter_z.asym.combat.label-avg.gauss-curv_filtered_sm1':1,
    '.inter_z.asym.combat.label-avg.gyrification.sm1':-1,
    '.inter_z.asym.combat.label-avg.thickness.sm1':-1,
    '.inter_z.asym.combat.label-avg.hippunfold_volume_icvcorr':-1
    }

    features_AI_title = {
    '.inter_z.asym.combat.label-avg.curvature.sm1' : 'Increased mean\ncurvature',
    '.inter_z.asym.combat.label-avg.gauss-curv_filtered_sm1': 'Increased intrinsic\ncurvature ',
    '.inter_z.asym.combat.label-avg.gyrification.sm1' : 'Decreased gyrification ',
    '.inter_z.asym.combat.label-avg.thickness.sm1' : 'Decreased thickness ',
    '.inter_z.asym.combat.label-avg.hippunfold_volume_icvcorr' :'Decreased volume',
    }

    #load abnormality thresholds
    file = os.path.join(BASE_PATH, 'data_saved','asymmetries_abnormality_thresholds.pkl')   
    with open(file, 'rb') as handle:
        abnormal_threshold = pickle.load(handle)

    #extract subject features
    data_p={}
    for feature in features:
        data_p[feature]={}
        for hemi in ['lh','rh']:
            vals = subj.load_feature_values(feature, hemi)
            data_p[feature][hemi]=vals[0]
        else:
            pass

    #normalise by direction
    data_lh = np.array([data_p[feature]['rh']/abnormal_AI_direction_feat[feature] for feature in features])
    
    #bar plot
    ax.barh(y=np.array(range(len(features))),
        width=data_lh,
        edgecolor="k",
        height=0.2,
        tick_label = [features_AI_title[feature] for feature in features],
        color='yellow',
        )

    # Add threshold lines for each bar
    for i, feature in enumerate(features):
        for (hemi,color,direc) in zip(['lh','rh'], ['navy','deeppink'],[-1,1]):
            thresh = abnormal_threshold[feature][hemi]
            #normalise threshold by abnormality direction and hemisphere
            thresh = thresh/abnormal_AI_direction_feat[feature]*direc 
            ax.plot([thresh, thresh], [i-0.2,i+0.2], color=color, linestyle='--', alpha=0.7)

    #removing the default axis on all sides:
    for side in ['right','top','left']:
        ax.spines[side].set_visible(False)
    plt.axvline(x = 0, color = 'green', linestyle = '-') #add line center
    ax.set_xlim([-7, 7])
    ax.tick_params(axis=u'both', which=u'both',length=0)
    ax.set_xticks([-6.5,-5, -2, 0, 2, 5, 6.5])
    ax.set_yticklabels(ax.get_yticklabels(),fontsize=20)
    ax.set_xticklabels(['Left', '5',  '2', '0', '2', '5', 'Right'], fontsize=20)
    ax.set_xlabel('absolute z-scored asymmetry index', fontsize=20)

    #add legend
    fig_tmp, ax_tmp = plt.subplots(1,1, figsize=(3,1))
    ax_tmp.plot([0,0],[0,0], c= 'navy', ls='--', linewidth= 1, alpha=1,  label=f'Left abnormality threshold')
    ax_tmp.plot([0,0],[0,0], c= 'deeppink', ls='--', linewidth= 1, alpha=1,  label=f'Right abnormality threshold')
    h, l =  ax_tmp.get_legend_handles_labels() # get labels and handles from ax1
    ax.legend(h, l, markerscale=2., scatterpoints=1, loc='best', bbox_to_anchor=(0.3, 0.75, 0.5, 0.5), fontsize=14)

    #save plot
    fig.savefig(filename, bbox_inches="tight")

    

def coords_seg_extract(seg_array):
    """
    Automatically searches through the provided rgb seg array along the x,y, and z axes.

    """
    x_coord = None
    x_size = 0
    y_coord = None
    y_size = 0
    z_coord = None
    z_size = 0

    for i in range(0, seg_array.shape[0]):
        if len(seg_array[i,:,:].nonzero()[0]) > x_size:
            x_coord = i
            x_size = len(seg_array[i,:,:].nonzero()[0])

    for i in range(0, seg_array.shape[1]):
        if len(seg_array[:,i,:].nonzero()[0]) > y_size:
            y_coord = i
            y_size = len(seg_array[:,i,:].nonzero()[0])

    for i in range(0, seg_array.shape[2]):
        if len(seg_array[:,:,i].nonzero()[0]) > z_size:
            z_coord = i
            z_size = len(seg_array[:,:,i].nonzero()[0])

    return x_coord, y_coord, z_coord

def return_labels_cmap(output_file=None):
    from matplotlib.colors import ListedColormap
    
    subfields_names = {'Subiculum':1, 'CA1':2, 'CA2':3, 'CA3':4, 'CA4':5, 'Dentate-Gyrus':6, 'SRLM':7, 'Cyst':8}
    colors =  np.array([[0,0,0], [0,0,128/255], [0,79/255,1], [0,200/255,1],  [0,1,108/255], [1,188/255,0], [128/255,128/255,128/255], [128/256,0,0 ], [255/256,211/256,155/256 ]])
    subfields_atlas={}
    for key in subfields_names.keys():
        subfields_atlas[key]=colors[subfields_names[key]]
    cmap = ListedColormap(colors)

    #save figure
    if output_file!=None:
        import matplotlib.patches as mpatches
        fig, ax = plt.subplots(1,1, figsize=(3,2))
        patches = []
        for key in subfields_atlas.keys():
            patches.append(mpatches.Patch(color=subfields_atlas[key], label=key))
        plt.axis('off')
        fig.legend(handles=patches)
        fig.savefig(output_file, dpi=96, transparent =True)
    
    return subfields_atlas, cmap

def plot_segmentations_subject(subject, hippunfold_folder, output_file, hemis=['L','R']):
    '''
    Plot HippUnfold segmentations for left and right hippocampi
    '''
    import SimpleITK as sitk

    fig, axs = plt.subplots(3, 2, figsize=(4,6), layout=None)

    for i,hemi in enumerate(hemis):
        file_hipo = os.path.join(hippunfold_folder, f'sub-{subject}', 'anat', f'sub-{subject}_hemi-{hemi}_space-cropT1w_desc-preproc_T1w.nii.gz')
        file_hipo_seg = os.path.join(hippunfold_folder, f'sub-{subject}', 'anat', f'sub-{subject}_hemi-{hemi}_space-cropT1w_desc-subfields_atlas-bigbrain_dseg.nii.gz')

        #load images
        img_array = sitk.GetArrayFromImage(sitk.ReadImage(file_hipo, sitk.sitkFloat32))  # convert to sitk object
        seg_array = sitk.GetArrayFromImage(sitk.ReadImage(file_hipo_seg, sitk.sitkInt64))

        # get cmap
        _ , cmap_labels = return_labels_cmap()

        # plot on conventional coronal view
        img_array = np.rot90(img_array, 2)
        seg_array = np.rot90(seg_array, 2)
        x,y,z = coords_seg_extract(seg_array)
        axs[0,i].imshow(img_array[x,:,:], cmap='gray')
        axs[0,i].imshow(seg_array[x,:,:], origin='lower', cmap=cmap_labels, alpha=0.4)
        axs[0,i].axis('off')
        axs[0,i].text(1, 125, 'R', fontsize = 8, color='red')
        axs[0,i].text(240, 125, 'L', fontsize = 8, color='red')
        axs[0,i].text(125, 240, 'S', fontsize = 8, color='red')
        axs[0,i].text(125, 5, 'I', fontsize = 8, color='red')
        # plot on conventional axial view
        img_array = np.flipud(img_array)
        seg_array = np.flipud(seg_array)
        x,y,z = coords_seg_extract(seg_array)
        axs[1,i].imshow(img_array[:,y,:], origin='lower', cmap='gray')
        axs[1,i].imshow(seg_array[:,y,:], origin='lower', cmap=cmap_labels, alpha=0.4)
        axs[1,i].axis('off')
        axs[1,i].text(1, 125, 'R', fontsize = 8, color='red')
        axs[1,i].text(240, 125, 'L', fontsize = 8, color='red')
        axs[1,i].text(125, 240, 'A', fontsize = 8, color='red')
        axs[1,i].text(125, 5, 'P', fontsize = 8, color='red')
        # plot on conventional sagital view
        x,y,z = coords_seg_extract(seg_array)
        img_array = np.rot90(img_array, -1)
        seg_array = np.rot90(seg_array, -1)
        axs[2,i].imshow(img_array[:,:,z], origin='lower', cmap='gray')
        axs[2,i].imshow(seg_array[:,:,z], origin='lower', cmap=cmap_labels, alpha=0.4)
        axs[2,i].axis('off')
        axs[2,i].text(1, 125, 'A', fontsize = 8, color='red')
        axs[2,i].text(240, 125, 'P', fontsize = 8, color='red')
        axs[2,i].text(125, 240, 'S', fontsize = 8, color='red')
        axs[2,i].text(125, 5, 'I', fontsize = 8, color='red')
        
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0)
        axs[0,i].set_title(f'{hemi} hippo', fontsize=12)

        fig.savefig(output_file, dpi=96, transparent =True)

def create_surf_plot(borders, faces, vertices, cmap=False, file='./tmp.png'):
    """plot and reload surface images"""
    from hippunfold_toolbox import plotting
    fig, ax_temp = plt.subplots(nrows=1, ncols=1, figsize=(8,8), subplot_kw={'projection': "3d"})
    plotting.surfplot_cdata(ax_temp,borders,faces,vertices, cmap=cmap)
    ax_temp.view_init(elev=90, azim=-90)
    plt.savefig(file, dpi=96, transparent =True)
    subprocess.call(f"convert {file} -trim {file}", shell=True)
    im = Image.open(file)
    im = im.convert("RGBA")
    im1 = np.array(im)
    plt.close('all')
    return im1

def plot_surfaces_subject(subject, hippunfold_folder, labels_file, output_file):
    '''
    Plot hippocampal surface folded extracted from HippUnfold
    '''
    #get labels
    labels = nb.load(labels_file).darrays[0].data
    subfields=list(set(labels))
    subfields_name={1:'Sub', 2:'CA1', 3:'CA2', 4:'CA3', 5:'CA4', 6:'DG'} 
    colors =  np.array([[0,0,128/255], [0,79/255,1], [0,200/255,1],  [0,1,108/255], [1,188/255,0], [128/255,128/255,128/255] ])
    
    #get surfaces and plot
    fig, ax = plt.subplots(1,1,figsize=(8,8) )
    vertices={}
    faces={}
    borders={}
    for i,hemi in enumerate(['L','R']):
        file_fold = os.path.join(hippunfold_folder, f'sub-{subject}', 'surf', f'sub-{subject}_hemi-{hemi}_space-T1w_den-0p5mm_label-hipp_midthickness.surf.gii')
        gii = nb.load(file_fold)
        vertices[hemi] = gii.get_arrays_from_intent('NIFTI_INTENT_POINTSET')[0].data
        faces[hemi] = gii.get_arrays_from_intent('NIFTI_INTENT_TRIANGLE')[0].data
        borders[hemi] = labels
    
    #shift vertices of right hippo to be closer to left
    shift = vertices['L'][:,0].max()-vertices['R'][:,0].min() + 3
    vertices['R'][:,0] = vertices['R'][:,0]+shift

    #concatenate surfaces on same space
    vertices_both = np.concatenate((vertices['L'], vertices['R']))
    faces_both = np.concatenate((faces['L'], faces['R'] + len(vertices['L'])))
    borders_both = np.concatenate((borders['L'], borders['R']))

    #plot
    im = create_surf_plot(borders_both, faces_both, vertices_both, cmap=colors)
    ax.imshow(im)
    ax.axis('off')

    fig.savefig(output_file, dpi=96, transparent =True)

def write_prediction_subject(subj, output_file, cohort, filename_model):
    '''
    Predict subject with a given trained model
    Plot the predicted probabilities of being left HS, right HS or no HS
    '''
    #TODO: get features from json saved with model
    features = [ 
    '.inter_z.asym.combat.label-avg.curvature.sm1',
    '.inter_z.asym.combat.label-avg.gauss-curv_filtered_sm1',
    '.inter_z.asym.combat.label-avg.gyrification.sm1',
    '.inter_z.asym.combat.label-avg.thickness.sm1',
    '.inter_z.asym.combat.label-avg.hippunfold_volume_icvcorr',
    ]

    #predict 
    y_pred, scores = predict_subjects(features, cohort, filename_model)

    #scores global
    smax = scores 
    max_ind = np.argmax(smax)

    #output predictions scores lateralisation
    x1 = np.array([scores[0][1:]])
    smax1 = x1 
    max_ind1 = np.argmax(smax1)
    sides = ['left', 'right']
    side_ipsi = sides[max_ind1]
    sides.remove(side_ipsi)
    side_contra = sides[0]

    #output predictions scores detection
    x2 = np.array([scores][0])
    smax2 = x2
    max_ind2 = np.argmax(smax2)
    
    #plot scores
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(7, 7))
    scores_name = ['left HS', 'no asymmetry', 'right HS',]
    scores_number = [smax[0][1], smax[0][0], smax[0][2]]
    graph = ax1.bar(scores_name, scores_number, color=['navy','green','deeppink'])
    ax1.set_xticklabels(scores_name, fontsize=15)
    ax1.get_yaxis().set_visible(False)
    ax1.spines[['left', 'right', 'top']].set_visible(False)
    for i, p in enumerate(graph):
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy()
        ax1.text(x+width/2,
                y+height*1.05,
                f'{round(scores_number[i]*100,1)}%',
                ha='center', fontsize=18)

    # plot info in text box in upper left in axes coords
    group = ['control', 'HS patient', 'HS patient'][max_ind]
    if group=='control':
        textstr = "\n".join(
            (
                f"AID-HS classifier indicates \n features consistent with normal hippocampi \n (predicted probability = {round(smax[0][0]*100,1)}%)",
                "",
                f"Note: the predcited probability of {side_ipsi} HS \n is relatively higher than \n of {side_contra} HS",
                )
        )
    else:
        textstr = "\n".join(
            (
                f"AID-HS classifier indicates \n features consistent with \n {side_ipsi} hippocampal sclerosis \n (predicted probability={round(smax[0][max_ind]*100,1)}%)",
                )
        )


    props = dict(boxstyle="round", facecolor=[240/256,128/256,128/256,1], alpha=0.5)
    ax2.text(0.5, 0.95, textstr, transform=ax2.transAxes, fontsize=20, verticalalignment="top", ha='center',  bbox=props, )
    ax2.axis("off")
        #fig.tight_layout()
    fig.savefig(output_file)
    return textstr

def plot_dice_score_subject(subject, hippunfold_folder, output_file):
    txt=['Quality check of segmentation (scores):']
    for i, hemi in enumerate(['L','R']):
        file_dice = os.path.join(hippunfold_folder, f'sub-{subject}', 'qc', f'sub-{subject}_hemi-{hemi}_desc-unetf3d_dice.tsv')
        df_temp = pd.read_csv(file_dice, sep = '\t', header=None)
        dice = round(df_temp.values[0][0],2)
        
        if dice<0.7:
            txt.append('{} hippocampi: {} - check segmentation'.format(hemi,dice))
        else:
            txt.append('{} hippocampi: {}'.format(hemi,dice))
    # plot info in text box in upper left in axes coords
    textstr = "\n".join(txt)
    fig, ax = plt.subplots(1, 1, figsize=(10, 2))
    props = dict(boxstyle="round", facecolor=[128/256,128/256,128/256,1], alpha=0.2)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=25, verticalalignment="top", bbox=props)
    ax.axis("off")
    #fig.tight_layout()
    fig.savefig(output_file)

def generate_prediction_report(subject_ids, hippunfold_dir, output_dir,):
    ''' Create report of hippocampal abnormalities
    inputs: 
        subject_ids: subjects ID
        output_dir: directory to save final reports
        '''
    # setup parameters
    base_feature_sets = [
        '.label-avg.hippunfold_volume_icvcorr',
        '.label-{}.thickness.sm1',
        '.label-{}.gyrification.sm1',
        '.label-{}.curvature.sm1',
        '.label-{}.gauss-curv_filtered_sm1',
    ]
    
    #-----------------------------------------------------
    # Generate GAM controls 
    features = ['.combat'+feature.format('avg') for feature in base_feature_sets]
    # load GAM controls
    file_gam = os.path.join(BASE_PATH, 'data_saved','GAM_curves_controls.pkl')   
    with open(file_gam, 'rb') as handle:
        controls_GAM = pickle.load(handle)

    for subject in subject_ids:

        # create dataset containing subject
        tmp = tempfile.NamedTemporaryFile(mode="w")
        create_dataset_file(subject, tmp.name)
    
        # create cohort containing subject
        c_combat = AidhsCohort(hdf5_file_root='{site_code}_{group}_featurematrix_combat_avg_newsubjects.hdf5', dataset=tmp.name)
        c_norm =  AidhsCohort(hdf5_file_root='{site_code}_{group}_featurematrix_norm_avg_newsubjects.hdf5', dataset=tmp.name)
    
        # subject bids id
        subj = AidhsSubject(subject, cohort=c_norm)
        subject_bids = subj.get_demographic_features(["Hippunfold_ids"])[0]

        # create output dir subject
        output_dir_subj = os.path.join(output_dir,subject)
        os.makedirs(output_dir_subj,exist_ok=True)
        
        #-----------------------------------------
        # plot segmentation hippocampus 
        filename1 = os.path.join(output_dir_subj, f'{subject}_hippo_segmentation.png')
        plot_segmentations_subject(subject_bids, hippunfold_dir, filename1)

        # add dices scores segmentation
        filename1bis = os.path.join(output_dir_subj, f'{subject}_hippo_segmentation_dices.png')
        plot_dice_score_subject(subject_bids, hippunfold_dir, filename1bis)

        #add legend subfields
        filename1bis2 = os.path.join(output_dir_subj, f'{subject}_hippo_segmentation_legend.png')
        return_labels_cmap(filename1bis2)

        #-----------------------------------------
        # plot surarface hippocampus 
        labels_file = os.path.join(BASE_PATH, 'templates', 'tmp_hemi-L_space-T1w_den-0p5mm_label-hipp_subfields.label.gii')
        filename2 = os.path.join(output_dir_subj, f'{subject}_hippo_surfaces.png')
        plot_surfaces_subject(subject_bids, hippunfold_dir, labels_file, output_file=filename2)

        #-----------------------------------------
        # plot patient on normative charts
        features = ['.combat'+feature.format('avg') for feature in base_feature_sets]
        subj = AidhsSubject(subject, cohort=c_combat)

        filename3 = os.path.join(output_dir_subj,f'{subject}_normative_charts.png')
        plot_patient_on_controls_charts(subj, features, controls_GAM, filename3)
        

        #-----------------------------------------
        # plot abnormalities directions
        features = ['.inter_z.asym.combat'+feature.format('avg') for feature in base_feature_sets]
        subj = AidhsSubject(subject, cohort=c_norm)  

        filename4 = os.path.join(output_dir_subj,f'{subject}_abnormalities_directions.png')
        plot_abnormalities_direction_subject(subj, features[::-1], filename= filename4)
        
        #-----------------------------------------
        # write predictions 
        subj = AidhsSubject(subject, cohort=c_norm)  

        #get model
        site_code = subj.get_demographic_features(["Site"])[0]
        filename_model = os.path.join(BASE_PATH, 'data_saved', f'model_LogReg_Hippunfold features_{site_code}.sav')
        filename5 = os.path.join(output_dir_subj,f'{subject}_predictions_scores.png')
        write_prediction_subject(subj, 
                                 filename5,
                                 cohort = c_norm,
                                 filename_model= filename_model,
                                )
        

        # ----------------------------------------------
        # create PDF overview
        pdf = PDF()  
        pdf = PDF(orientation="P")  
        pdf = PDF(unit="mm")  
        pdf = PDF(format="A4")  

        text_info_1 = "This report presents an in-depth characterisation of hippocampal abnormalities using AID-HS" + \
            "\n Page 1 displays the results of the segmentation and the hippocampal surface reconstructions using Hippunfold. It also displays quality control scores. A dice score <0.70 indicates a possible poor-quality segmentation that would benefit from additional checks " + \
            "\n Page 2 maps the values of the left and right hippocampi to the normative trajectories of healthy population, and presents the magnitude and direction of feature asymmetries. It also output the scores resulting of the automated classification alongside an adapted interpretation"


        disclaimer = "Disclaimer: This algorithm is intended for research purposes only and has not been reviewed or approved by the Medicines and Healthcare products Regulatory Agency (MHRA),European Medicine Agency (EMA) or by any other agency. Any clinical application of the software is at the sole risk of the party engaged in such application. There is no warranty of any kind that the software will produce useful results in any way. Use of the software is at the recipient own risk"

        footer_txt = "This report was created by Mathilde Ripart"

        #### create main page with overview on inflated brain

        # add page
        pdf.add_page()
        # add line contours
        pdf.lines()
        # add header
        pdf.custom_header(txt1="AID-HS report", txt2=f"Patient ID: {subject}")
        # add info box
        pdf.info_box(text_info_1)
        # add disclaimer box
        pdf.disclaimer_box(disclaimer)
        # Arial bold 15
        pdf.set_font('Arial', 'B', 15)
        
        # add title 1st figure left
        pdf.cell(w=20, h=16, txt="Hippocampal segmentation", border=0, ln=0, align='L')
        # add title 2nd figure right
        pdf.cell(90) # Pushes next cell to right by 100 mm
        pdf.cell(w=20, h=16, txt="Hippocampal pial surfaces", border=0, ln=2, align='L')
        
        #add segmentation figure left
        pdf.image(filename1, 5, 105, link='', type='', w=120, h=155)
        
        #add surfaces figure right
        pdf.image(filename2, 110, 110, link='', type='', w=90, h=100)
        
        #add legend subfields figure right below
        pdf.image(filename1bis2, 110, 200, link='', type='', w=70, h=50)

         #add dices scores bellow segmentation 
        pdf.image(filename1bis, 5, 240, link='', type='', w=120, h=30)

        
        #TODO add explanation box ?

        # add footer date
        pdf.custom_footer(footer_txt)

        ### SECOND PAGE
        pdf.add_page()
        pdf.set_font('Arial', 'B', 15)
        #add figure charts
        pdf.cell(w=20, h=40, txt="Individual hippocampal features vs normative trajectories", border=0, ln=2, align='L')
        pdf.image(filename3, 5, 40, link='', type='', w=190, h=110)
        
        #add figure direction abnormalities
        pdf.cell(w=20, h=220, txt="Asymmetries", border=0, ln=0, align='L')
        pdf.cell(80) # Pushes next cell to right by 100 mm
        pdf.cell(w=20, h=220, txt="Automated detection & lateralisation", border=0, ln=2, align='L')
        pdf.image(filename4, 5, 170, link='', type='', w=110, h=80)
        
        #add box predictions
        pdf.image(filename5, 120, 170, link='', type='', w=90, h=90)
        
        # add footer date
        pdf.custom_footer(footer_txt)

        # save pdf
        file_path = os.path.join(output_dir_subj, f"Report_{subject}.pdf")
        pdf.output(file_path, "F")

        print(f'Report ready at {file_path}')


if __name__ == "__main__":
    # Set up experiment
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--hippunfold_dir", default="", help="folder containing the hippunfold data"
    )
    parser.add_argument(
        "--output_dir", default="", help="folder containing the output prediction and reports"
    )
    parser.add_argument('-id','--id',
                    help='Subjects ID',
                    required=False,
                    default=None)
    args = parser.parse_args()
    hippunfold_dir=args.hippunfold_dir
    output_dir=args.output_dir

    subject_ids=np.array([args.id])
    generate_prediction_report(
        subject_ids,
        hippunfold_dir=hippunfold_dir,
        output_dir=output_dir,
    )
