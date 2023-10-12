# import and functions
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
import tempfile
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, auc
from aidhs.aidhs_cohort_hip import AidhsCohort, AidhsSubject

def is_nan(x):
    return (x != x)

def prepare_dataset(features, cohort ):
    hemis=['lh', 'rh']
    #get subjects 
    subjects = cohort.get_subject_ids(lesional_only = False)

    df=pd.DataFrame()
    for subject in subjects:
        # create subject object
        subj = AidhsSubject(subject, cohort=cohort)
        group=subj.get_demographic_features("group")
        if group=='P':
            lesional_hemi = subj.get_demographic_features("Lesional hemi")
        elif group=='DC':
            # lesional_hemi = subj.get_demographic_features("Lesional hemi")
            lesional_hemi = random.choice(['lh','rh'])   #random choice
        else:
            lesional_hemi = random.choice(['lh','rh'])   #random choice
            # lesional_hemi = '666'
    #     if not is_nan(lesional_hemi):
        for hemi in hemis:
            values={}
            #get demographic info 
            values['ID']=subj.subject_id
            values['group']=group
            values['hemi'] = hemi
            values['site'], values['age_scan'], values['sex'], values['mri_neg'], values['lobes']= subj.get_demographic_features(["Site","Age at preoperative", "Sex", "Ever reported MRI negative", "Lobes",])
            if is_nan(lesional_hemi):
                values['lesional'] = np.nan
            elif hemi==lesional_hemi:
                values['lesional'] = 'ipsi'
            else:
                values['lesional'] = 'contra'
            #get structural and intensity features               
            for feature in features:
                vals = subj.load_feature_values(feature, hemi)
                if vals.sum()!=0:
                    values[feature]=vals[0]
                else:
                    pass
                
            # save values for each subject in panda matrix
            df2 = pd.DataFrame([values])
            df = pd.concat([df,df2],ignore_index=True)
    return df

def prepare_dataset_newsubject(subj, features ):
    hemis=['lh', 'rh']

    df=pd.DataFrame()
    for hemi in hemis:
        values={}
        #get demographic info 
        values['ID']=subj.subject_id
        values['hemi'] = hemi
        #get features               
        for feature in features:
            vals = subj.load_feature_values(feature, hemi)
            if vals.sum()!=0:
                values[feature]=vals[0]
            else:
                pass 
        # save values for each hemisphere
        df2 = pd.DataFrame([values])
        df = pd.concat([df,df2],ignore_index=True)
    return df

def split_train_test(row, file):
    df_split=pd.read_csv(file, header=0,encoding='unicode_escape', )
    try:
        split= np.array(df_split[df_split['subject_id']==row['ID']]['split'])[0]
    except:
        split = np.nan
    return split


def prepare_train_test_data(df,  file, categories, mode='train', lateralisation_only = False, differentiation_only=False):
    from sklearn.preprocessing import label_binarize
    
    ## prepare train and test data 
    data=df.copy()
    #separate in train and test
    data['split'] = data.apply(lambda row:split_train_test(row, file), axis=1)
    #remove nan split
    data = data.dropna(subset=['split'])

    if mode == 'train':
        ##PREPARE TRAINING DATASET
        train_dataset={}
        datatrain=data[data['split']=='train']
        if len(datatrain)==0:
            print('no train data')
            return
        #create class
        if lateralisation_only:
            datatrain['classe'] = datatrain.apply(lambda row:define_classe_lateralisation(row), axis=1)
        elif differentiation_only:
            datatrain['classe'] = datatrain.apply(lambda row:define_classe_differentiation(row), axis=1)
        else:
            datatrain['classe'] = datatrain.apply(lambda row:define_classe(row), axis=1)
        #remove nan
        datatrain = datatrain.dropna(subset=['classe'])
        #combine left and right hippo
        datatrain = datatrain.pivot(index='ID', columns='hemi', values= categories + ['site','classe', 'group','mri_neg', 'hemi', 'lesional'])
        # select class & remove nan
        y_class = 'classe'
        Y = datatrain[y_class]['lh'].copy() 
        # Encode data class
        le = LabelEncoder().fit(Y)
        print(f"Classes = {le.classes_}")
        Y = le.transform(Y)
        #Binarize the output
        y_bin = label_binarize(Y, classes=np.arange(len(set(Y))))
        n_classes = y_bin.shape[1]
    
        #select features
        X = datatrain[categories].copy()
        X=np.array(X)
        Y=np.array(Y)
        #print numbers
        print('There is {} subject for training'.format(len(Y)))
        if lateralisation_only :
            print('number ipsi lh: ' + str(len(np.where(Y==0)[0])))
            print('number ipsi rh: ' + str(len(np.where(Y==1)[0])))
        elif differentiation_only:
            print('number control: ' + str(len(np.where(Y==0)[0])))
            print('number patient: ' + str(len(np.where(Y==1)[0])))
        else:
            print('number ipsi lh: ' + str(len(np.where(Y==1)[0])))
            print('number ipsi rh: ' + str(len(np.where(Y==2)[0])))
            print('number control: ' + str(len(np.where(Y==0)[0])))
            #store data
        train_dataset['X']=X
        train_dataset['Y']=Y
        train_dataset['y_bin']=y_bin
        train_dataset['n_classes']=n_classes
        train_dataset['df']=datatrain
        return train_dataset
    
    elif mode=='test':
        ##PREPARE TEST DATASET
        test_dataset = {}
        datatest = data[data['split']=='test']
        if len(datatest)==0:
            print('no test data')
            return
        #combine left and right hippo
        # datatest = datatest.pivot(index='ID', columns='hemi', values= categories + ['site', 'group', 'mri_neg', 'hemi', 'lesional'])
        datatest = datatest.pivot(index='ID', columns='hemi', values= categories)
        
        #select features
        X_test = datatest[categories].copy()
        #store data
        test_dataset['X']=X_test
        test_dataset['df']=datatest
        return test_dataset
    else:
        print('This mode does not exists')

        

def evaluate(y,y_hat,labels):
    ''' Function to compute and display confusion matrix '''
    print('classification report :')
    print(classification_report(y,y_hat))
    cm = confusion_matrix(y,y_hat)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cmat = pd.DataFrame(cm)
    cmat.columns = labels
    cmat.set_index([pd.Index(labels)],inplace=True)
    sns.heatmap(cmat,cmap="YlGnBu", annot=True)
    plt.title("Confusion Matrix")


def define_classe(row):
    ''' function to classify data into left HS, right HS and controls from a DataFrame'''
    if (row['group']=='C')|(row['group']=='DC'):
#         return np.array([0,0])  
        return 'control'
    elif (row['group']=='P')&(row['hemi']=='lh')&(row['lesional']=='ipsi'):
#         return np.array([1,0])
        return 'lh_ipsi'
    elif (row['group']=='P')&(row['hemi']=='rh')&(row['lesional']=='contra'):
#         return np.array([1,0])
        return 'lh_ipsi'
    elif (row['group']=='P')&(row['hemi']=='rh')&(row['lesional']=='ipsi'):
#         return np.array([0,1])
        return 'rh_ipsi'
    elif (row['group']=='P')&(row['hemi']=='lh')&(row['lesional']=='contra'):
#         return np.array([0,1])
        return 'rh_ipsi'
    else:
        subject = row['ID']
        print(f'{subject} does not fit in a category')
        return np.nan
    
def define_classe_lateralisation(row):
    ''' function to classify data into left HS or right HS'''
    if (row['group']=='P')&(row['hemi']=='lh')&(row['lesional']=='ipsi'):
        return 'lh_ipsi'
    elif (row['group']=='P')&(row['hemi']=='rh')&(row['lesional']=='contra'):
        return 'lh_ipsi'
    elif (row['group']=='P')&(row['hemi']=='rh')&(row['lesional']=='ipsi'):
        return 'rh_ipsi'
    elif (row['group']=='P')&(row['hemi']=='lh')&(row['lesional']=='contra'):
        return 'rh_ipsi'
    else:
        return np.nan

def define_classe_differentiation(row):
    ''' function to classify data into patient or control'''
    if (row['group']=='P'):
        return 'patient'
    elif (row['group']!='P'):
        return 'control'
    else:
        return np.nan
    




class ModelMLP():
    def __init__(self, layers=(5,10), activation='relu',max_iter=300, seed_model=0, early_stopping=False):
        self.layers = layers
        self.activation = activation
        self.max_iter = max_iter
        self.seed_model = seed_model
        self.early_stopping = early_stopping
        self.imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        self.scalor = StandardScaler()

        return

    def initialise_model(self,):
        self.model_ini = MLPClassifier(hidden_layer_sizes= self.layers, 
                                        activation = self.activation, 
                                        max_iter=self.max_iter, 
                                        random_state=self.seed_model, 
                                        early_stopping=self.early_stopping,
                                        n_iter_no_change=50)
        return self.model_ini
    
class ModelLogR():
    def __init__(self, seed_model=0):
        self.seed_model = seed_model
        self.imputer = SimpleImputer(missing_values=np.nan, strategy='mean',)
        self.scalor = StandardScaler()

        return

    def initialise_model(self,):
        self.model_ini = LogisticRegression(random_state=self.seed_model, class_weight='balanced', multi_class='multinomial', solver='lbfgs')
        return self.model_ini

def cross_validation_training(X, Y, model, splits):
    #initialise results vectors
    scores=np.zeros((len(X), len(set(Y))))
    y_pred=np.zeros(len(Y))
    clfs=np.zeros(len(splits),  dtype=object)
    for i, (train_index, test_index) in enumerate(splits):
        #initialise model
        clfs[i] = model.initialise_model() 
        clfs[i] = make_pipeline(model.imputer, model.scalor, clfs[i])
        #split data in stratified fold
        x_train_fold, x_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = Y[train_index], Y[test_index]
        #train on the fold
        clfs[i].fit(x_train_fold, y_train_fold)
        scores[test_index]=clfs[i].predict_proba(x_test_fold)
        y_pred[test_index]=clfs[i].predict(x_test_fold)
    return scores, y_pred, clfs

def skf_cross_validation_training(X, Y, model, skf):
    #initialise results vectors
    scores=np.zeros((len(X), len(set(Y))))
    y_pred=np.zeros(len(Y))
    clfs=np.zeros(skf.get_n_splits(X, Y),  dtype=object)
    for i, (train_index, test_index) in enumerate(skf.split(X, Y)):
        #initialise model
        clfs[i] = model.initialise_MLP() 
        clfs[i] = make_pipeline(model.imputer, model.scalor, clfs[i])
        #split data in stratified fold
        x_train_fold, x_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = Y[train_index], Y[test_index]
        #train on the fold
        clfs[i].fit(x_train_fold, y_train_fold)
        scores[test_index]=clfs[i].predict_proba(x_test_fold)
        y_pred[test_index]=clfs[i].predict(x_test_fold)
    return scores, y_pred, clfs

def training(X, Y, model, ):
    #initialise results vectors
    scores=np.zeros((len(X), len(set(Y))))
    y_pred=np.zeros(len(Y))
    #initialise model
    clf = model.initialise_MLP() 
    clf = make_pipeline(model.imputer, model.scalor, clf)
    #train on the fold
    clf.fit(X, Y)
    scores=clf.predict_proba(X)
    y_pred=clf.predict(X)
    return scores, y_pred, clf



def compute_performances(Y, y_bin, y_pred, scores):
    performances={}
    #initialise results array
    performances['roc_auc']=np.zeros( y_bin.shape[1])
    performances['accurately_classified']=np.zeros(len(Y))
    performances['accuracy_classes']=np.zeros(len(set(Y)))
    

    if y_bin.shape[1]<2:
        fpr, tpr, _ = roc_curve(y_bin[:], scores[:, 0])
        performances['roc_auc'][0] = auc(fpr, tpr)
    else: 
        for c in range(0,y_bin.shape[1]):
            fpr, tpr, _ = roc_curve(y_bin[:, c], scores[:, c])
            performances['roc_auc'][c] = auc(fpr, tpr)


    cm = confusion_matrix(Y,y_pred)
    performances['accuracy'] = cm.diagonal().sum()/cm.sum()
    performances['accuracy_classes'][:] = cm.diagonal()/cm.sum(axis=1)
    performances['accurately_classified'] = (Y==y_pred)
    performances['predictions'] = y_pred
    performances['scores'] = scores

    return performances

def get_lateralisation_performances(Y, scores):
    #add prediction of lateralisation only
    from sklearn.utils.extmath import softmax
    predict_side = []
    accurately_lateralised = []
    new_scores = []
    for i, _ in enumerate(scores):
        if Y[i]==0:
            predict_side.append(np.nan)
            accurately_lateralised.append(np.nan)
            new_scores.append(np.array([np.nan,np.nan]))
        else:
            x = np.array([[scores[i,1], scores[i,2]]])
            # smax = softmax(x)
            smax = x # don't apply softmax
            pred = [1,2][np.argmax(smax)]
            predict_side.append(pred)
            accurately_lateralised.append(True if pred == Y[i] else False)
            new_scores.append(smax[0])
    return predict_side, accurately_lateralised, new_scores

def get_differentiation_performances(Y, scores):
    #add prediction of group only
    from sklearn.utils.extmath import softmax
    predict_group = []
    accurately_differentiate = []
    new_scores = []
    for i, _ in enumerate(scores):
        x = np.array([[scores[i,0], max(scores[i,1],scores[i,2])]])
        # smax = softmax(x)
        smax = x # don't apply sofmax
        pred = [0,1][np.argmax(smax)] # predict 0 if controls, 1 if patient
        new_Y = 0 if Y[i]==0 else 1  # class 0 is control and class >0 is patient
        predict_group.append(pred)
        accurately_differentiate.append(True if pred == new_Y else False) 
        new_scores.append(smax[0])
    return predict_group, accurately_differentiate, new_scores

def create_dataset_file(subjects_ids, save_file):
    df=pd.DataFrame()
    if  isinstance(subjects_ids, str):
        subjects_ids=[subjects_ids]
    df['subject_id']=subjects_ids
    df['split']=['test' for subject in subjects_ids]
    df.to_csv(save_file)

def predict_subject(subj, features, dataset, filename_model):
    # load model 
    loaded_model = joblib.load(open(filename_model, 'rb'))

    #create dataset
    df_sub = prepare_dataset_newsubject(subj, features)

    #prepare data
    test_dataset = prepare_train_test_data(df_sub, dataset, features, mode='test')

    #predict
    scores=loaded_model.predict_proba(test_dataset['X'])
    y_pred=loaded_model.predict(test_dataset['X'])

    return y_pred, scores

