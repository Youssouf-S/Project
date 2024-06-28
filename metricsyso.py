# -*- coding: utf-8 -*-
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import laspy
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score,jaccard_score
import matplotlib.pyplot as plt 
import json
from tqdm import tqdm

"""# Here, we're going to set, useful fonctions or potentialy useful 

def fIoU (tp,fp,fn) :
    return tp / (tp + fp + fn)
#intersection over union


# Foncton to calculate the precision of our model prediction
def Metricsecall (tp,fn) :
    
    #nombre : nombre de points classifiés dans une certaine classe par le nuage de référence
    if (tp+fn) == 0 :
        return 0
    else :
        return tp/(tp+fn)
    
def faccuracy(tp,fp,fn):
    return tp/(tp+fp+fn)
    
#nombre de points dans la même classification (les true points en cas de nuage de référence) / nombre de point de la classe 
#spécifique dans le nuage 1
#qualitativement c'est la proportion de points correctements classifiés par rapport aux points dans la classif
# recall == 1 signifie que l'algo n'omet pas de points


# Foncton to calculate the precision of our model prediction

def fprecision (tp,fp) :
    if (tp+fp) == 0 :
        return 0
    else :
        return tp/(tp+fp)
# precision == 1 signifie que que l'algo ne prend pas de points en trop

# here, is for a specific comparison between precision and the recall 
def ff1score (precision,rappel) :
    if (precision+rappel) < 10**-5 :
        return 0
    return 2*(precision*rappel)/(precision+rappel)"""


# =======This fonction retutrns general inforrmation about the file=====================#
#========possible to edit the script in oder to define your own output willing==========#

def infos_class(file_path):
    with laspy.open(file_path) as fh:
        print('Points Metricsom Header:', fh.header.point_count)
        las = fh.read()
        print(las.xyz.shape)
        print('Points Metricsom data:', len(las.points))
        unique_classes,class_counts= np.unique(las.classification, return_counts=True)
        class_counts = dict(zip(unique_classes, class_counts))
        print(class_counts)
        #print(las.classification)
        #  Display number of return
        
        """ground_pts= las.classification
        bins_g, counts_g = np.unique(las.return_number[ground_pts], return_counts=True)
        print('Ground Point Return Number distribution:')
        for r,c in zip(bins_g,counts_g):
            print('    {}:{}'.format(r,c))"""


# =============This part is only for reading las files====================================#
# ==============It returns something like this: with the number of points=================#
#=<LasData(1.4, point fmt: <PointFormat(6, 8 bytes of extra dims)>, 1258912 points, 1 vlrs)>=#

def read_file(file_path):
    with laspy.open(file_path) as fh:
        data=fh.read()
    return data
#
# ===================reading data and  extracting specific information,===================# 
# ====================like the occurences of classes in the file==========================#

def read_file_class(file_path):
    with laspy.open(file_path) as fh:
        data=fh.read()
    return data.classification
#
#====== Textract data label (different existing classes inside las files)=================#

def data_label(las):
    unique_classes,class_counts= np.unique(las.classification, return_counts=True)
    class_counts = dict(zip(unique_classes, class_counts))    
    return unique_classes

# this fonction is for Extracting Data 3D cordonnates Metricsom laZ files
def label_xyz(las):
    label_xyz = las.xyz
    return label_xyz


# ==============this fonction allow to count number of points in a class===============#

def count_points_per_class(file_path):
    class_counts = {}
    with laspy.open(file_path) as fh:
        las = fh.read()
        unique_classes, class_counts = np.unique(las.classification, return_counts=True)
        class_counts = dict(zip(unique_classes, class_counts))
    return class_counts


#
# ==========Extract each label of class in its occurence in the global point cloud======#

def num_points(classe):
    num_point_clas=[]
    for el in tqdm(range(len(classe))):
        num_point_clas.append((classe[el]))
    return num_point_clas

# ====================Compute matrix of confusion======================================#


def compute_confusion_matrix(ground_truth, predictions):
    
    # Identifier les classes uniques dans ground_truth et predictions
    unique_classes = np.unique(np.concatenate((predictions, ground_truth)))
    
    # Créer une matrice de confusion basée uniquement sur les classes uniques
    conf_matrix = np.zeros((len(unique_classes), len(unique_classes)))
    print("len ground_truth",len(ground_truth))
    print("len pred",len(predictions))
    
    #if len(ground_truth) != len(predictions):
            #raise ValueError("Les tailles de ground_truth et predictions ne correspondent pas")
            
    #if len(ground_truth)!= len(predictions):
    min_length = min (len(ground_truth), len(predictions))
    print(f"min length{min_length}")
    ground_truth = ground_truth[:min_length]
    predictions =  predictions[:min_length]
    
            
    # Remplir la matrice de confusion uniquement pour les classes existantes
    for i in tqdm(range(len(predictions))):
        pred_class = predictions[i]
        true_class = ground_truth[i]
        
        # Vérifier si la prédiction a une classe correspondante dans les données réelles
        if true_class in unique_classes:
            pred_index = np.where(unique_classes == pred_class)[0][0]
            true_index = np.where(unique_classes == true_class)[0][0]
            conf_matrix[true_index, pred_index] += 1

    # Créer des titres de classe basés sur les classes uniques
    class_titles = [f"{c}" for c in unique_classes]

    # Créer le DataFrame de la matrice de confusion avec les titres de classe
    conf_matrix_df = pd.DataFrame(conf_matrix, index=class_titles, columns=class_titles)
    conf_matrix_df.index.name = "Classes"
    return conf_matrix_df





class Metrics:
    
    #
    # ====================Initializing my class Metrics================================================#
    
    def __init__(self, true_labels, pred_labels) :
        self.true_labels=true_labels
        self.pred_labels=pred_labels
     
    #
    # ====================Display metrics simply as a dictionnary======================================#    
    def compute_metrics(self):
        """ return metrics by class"""
        IoU=jaccard_score(self.true_labels, self.pred_labels, average=None)
        precision = precision_score(self.true_labels, self.pred_labels, average=None,zero_division='warn')
        recall = recall_score(self.true_labels, self.pred_labels, average=None,zero_division='warn')
        f1 = f1_score(self.true_labels, self.pred_labels, average=None)
        accuracy = accuracy_score(self.true_labels, self.pred_labels)
        
        metrics_dict = {
        'IoU': IoU,
        'Recall': recall,
        'Precision': precision,
        'F1_score': f1,
        'Accuracy': accuracy
    }
        return metrics_dict
    # 
    
    # ====================Display metrics bay class as a frame  without weighting =====================#
    
    def metrics_frame(self,results):
        unique_classes = np.unique(np.concatenate((self.true_labels, self.pred_labels)))
        # Créez des titres de classe basés sur les classes uniques
        """default_dict = {"1":"default","2":"ground","5": "vegetation", 
                  "6":"building","7": "wall", 
                  "196": "structure","211":"cable"
                  }"""
        class_titles = [f"{c}" for c in unique_classes]
        
        nam_metrics = ['IoU', 'Recall', 'Precision', 'F1_score', 'Accuracy']
        Metrics = np.zeros((5, len(results['IoU'])))
        
        for i, metric_name in enumerate(nam_metrics):
            Metrics[i] = results[metric_name]    
        #  
        # Créez le DataMetricsame de la matrice de confusion avec les titres de classe
        Metrics_df = pd.DataFrame(Metrics, index=nam_metrics, columns=class_titles)
        Metrics_df.index.name = 'Métriques'
        Metrics_df.columns.name = 'Classes'
        return Metrics_df
        
    # ====================Display metrics as a frame by weghting each class===========================#   
    def weighted_metrics(self):
        
        """Calculate metrics for each label, and find their average weighted 
        by support (the number of true instances for each label)"""
        
        IoU_w=jaccard_score(self.true_labels, self.pred_labels, average="weighted")
        precision_w = precision_score(self.true_labels, self.pred_labels, average='weighted',zero_division='warn')
        recall_w = recall_score(self.true_labels, self.pred_labels, average='weighted',zero_division='warn')
        f1_w = f1_score(self.true_labels, self.pred_labels, average='weighted')
        accuracy = accuracy_score(self.true_labels, self.pred_labels)
        
        nam_metrics = ['IoU', 'Recall','precision','Accuracy', 'F1_score']
        weigh_metrics=[IoU_w,recall_w,precision_w,f1_w,accuracy]
        weigh_metrics_df=pd.DataFrame({'Valeurs': weigh_metrics}, index=nam_metrics)
        weigh_metrics_df.index.name = 'Métriques'
        
        return weigh_metrics_df
        
        
    # ====================Display metrics as a graphics curves======================================#   
        
    def graphmetrics(self,metrics_data):        
        data=np.arange(1, len(metrics_data['IoU'])+1)
        plt.figure(figsize=(15,7))
        # Tracé des métriques
        for metric, values in metrics_data.items():
            if metric != 'Accuracy':             
                plt.plot(data, values, label=metric)
        
        # Ajout de légendes et de titres
        plt.xlabel('Classes')
        plt.ylabel('Valeur')
        plt.title('Métriques par classe')
        plt.legend()
        plt.grid(True)

            # Affichage du graphique
        plt.tight_layout()
        plt.show()
            
    # ====================Display metrics as subplots graphics on grids===============================#
    
    def n_graphmetrics(self, metrics_data):        
        data = np.arange(1, len(metrics_data['IoU']) + 1)
        metrics_to_plot = [metric for metric in metrics_data if metric != 'Accuracy']
        num_metrics = len(metrics_to_plot)
        
        fig, axes = plt.subplots(num_metrics, 1, figsize=(15, 7*num_metrics))
        
        for i, (metric, values) in enumerate(metrics_data.items()):
            if metric != 'Accuracy':
                ax = axes[i] if num_metrics > 1 else axes  # Utiliser le même axe s'il n'y a qu'une seule métrique
                ax.plot(data, values, label=metric)
                ax.set_xlabel('Classes')
                ax.set_ylabel('Valeur')
                ax.set_title(f'Métrique: {metric}')
                ax.legend()
                ax.grid(True)  # Ajout de la grille
                
        plt.tight_layout()
        plt.show()
        
        # Filter metrics according to the models config
        
class JsonFile:
    def __init__(self, filepath):
        self.filepath = filepath
    
    def read_json(self):
        with open(self.filepath) as file:
            return json.load(file)

    # Extracting  class grouping in with a dictionnary before process in KPConv
    def convdict(self):
        with open(self.filepath) as file:
            f = json.load(file)
        return f['conv_dict']
    
    # Extracting  conversion class in dictionnary before process in KPConv
    def convclass(self):
        with open(self.filepath) as file:
            f = json.load(file)
        return f['conv_class']
    
    # This part is lead to convert ground truth point before computing metrics
    # Indeed is due to the fact that all data are converted duration the inference process
    # and the number of the class after the process is lead by the diuctionary "conv_class" wich is 
    # in config_rte_gson" file
    
    def conv_to_Truclass(self, conv_dict, conv_class):
        # the following dictionnary is a default, some classes can be added inside
        
        default_dict = {"1":"default","2":"ground","5": "vegetation", 
                  "6":"building","7": "wall", 
                  "196": "structure","211":"cable"
                  }
        # conversion from conv_class to truth classs defined in defautlt dict like 
        # "0":1 where "0" is from the conv_class as known proceded by kpconv and 1 the name of truth class
        model = {k:int(l) for k, v in conv_class.items() for l,m in default_dict.items() if v == m}
        #model2 = {"0":1, "1":2, "2":5, "3":6, "4":196, "5":211} 
        #return conv_dict correctly mapped with the output label after inference
        return {k:j for k, v in conv_dict.items() for  i, j in model.items()  if int(i) == v }       
                   