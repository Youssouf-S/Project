# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from metricsyso import *
from tqdm import tqdm

def show_metric(truthpath, predpath, savepath):
    # Créez le répertoire de sauvegarde s'il n'existe pas
    os.makedirs(savepath, exist_ok=True)
    pred_filename = os.path.splitext(os.path.basename(predpath))[0]
    
    conv_dict1 = {"1": 0,"2": 1,"3": 0,"4": 2,"5": 2, "6": 3,"7": 8,"12": 0,"13": 0,"15": 0,"50": 4,"51": 4,
            "52": 4, "53": 4,"54": 4, "55": 4, "64": 4,"72": 7, "101": 5,"102": 5,"103": 5, "104": 5,
        "105": 5,"106": 5,"107": 5, "108": 5,"109": 5,"110": 5,"111": 5,"120": 5,"121": 5,"150": 6, "151": 6,
    "152": 6,"53": 6, "154": 6,"155": 6, "160": 6, "161": 6, "185": 6, "196": 5,
    "197": 5,"200": 6,"201": 6, "202": 6,"211": 6,"212": 6,"213": 6, "221": 6, "222": 6,"223": 6
    }
    model1 = {"0":1, "1":2, "2":5, "3":6, "4":59, "5":196, "6": 211, "7":72}

#

    direct_conv1={k:j for k, v in conv_dict1.items() for  i, j in model1.items()  if int(i) == v }


    # Exemple d'utilisation
    rte1_pred=read_file_class(predpath)
    rte1_truth=read_file_class(truthpath)
    pred_class = num_points(rte1_pred)
    gtruth_class = num_points(rte1_truth)


    gtruth_cleanead = [direct_conv1.get(str(k)) if direct_conv1.get(str(k)) is not None else k for k in gtruth_class]

    rte1_ground_truth =np.array(gtruth_cleanead)
    rte1_predictions = np.array(pred_class)
    
    print("len gtruth_class",len(gtruth_class))
    print("gtruth unique classes",np.unique(gtruth_class))    
    print("") 
    print("len rte1_gtruth_class", len(rte1_ground_truth))
    print("unique gtruth classes merged classe", np.unique(rte1_ground_truth))
    
    print("")
    print(f"len pred_class{len(rte1_predictions)}")
    print(f" unique classe predictions{np.unique(rte1_predictions)}")
    print("")

    unique_classes =  np.unique(np.concatenate((rte1_ground_truth, rte1_predictions)))
    clas_name = [str(name) for name in tqdm(unique_classes)]

    default_dict = {"1":"default","2":"ground","5": "vegetation", 
                "6":"building","7": "wall", 
                "196": "structure","211":"cable"
                }

    conf_matrix = compute_confusion_matrix(rte1_ground_truth, rte1_predictions)
    
    print(conf_matrix.shape[0])
    print(len(clas_name))
    if conf_matrix.shape[0] != len(clas_name) or conf_matrix.shape[1] != len(clas_name):
        raise ValueError("Mismatch between confusion matrix dimensions and class names")
          
    #conf_matrix = confusion_matrix(rte1_ground_truth, rte1_predictions)

    # Sauvegarder la matrice de confusion dans un fichier Excel
    with pd.ExcelWriter(os.path.join(savepath, f"conf_matrix_{pred_filename}.xlsx")) as writer:

        #df = pd.DataFrame(conf_matrix, index=clas_name, columns=clas_name)
        conf_matrix.rename(columns = default_dict, index = default_dict, inplace = True)
        conf_matrix.to_excel(writer, index=True)
        print(f"Fichier sauvegardé avec succès: {os.path.join(savepath, f'conf_matrix_{pred_filename}.xlsx')}")

    # Afficher et sauvegarder la matrice de confusion
        
    #ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=clas_name)
    """dis=ConfusionMatrixDisplay.from_predictions(rte1_ground_truth, rte1_predictions, labels = clas_name )
    mask = np.eye(len(conf_matrix))

    fig, ax = plt.subplots(figsize=(30, 10))
    #ConfusionMatrixDisplay(conf_matrix, display_labels=clas_name).plot(ax=ax)
    dis.plot(ax=ax)
    ax.imshow(mask, cmap='Blues', interpolation='nearest', alpha=0.5)

    plt.savefig(os.path.join(savepath, f"conf_matrix_{pred_filename}.png"))
    plt.show()"""
    
    
    #========================== metrics=============================================#
    
    metrics=Metrics(rte1_ground_truth,rte1_predictions)
    ## calling a methods for computing metrics
    m=metrics.compute_metrics()

    ## for displaying as frame

    fr = metrics.metrics_frame(m)
    wg = metrics.weighted_metrics()


    # Sauvegarder la matrice de confusion dans un fichier Excel
    with pd.ExcelWriter(os.path.join(savepath, f"{pred_filename}_metric_frame.xlsx")) as writer:
        
        fr.rename(columns =default_dict,index = default_dict, inplace = True)
        fr.to_excel(writer, index=True)
        print(f"Fichier sauvegardé avec succès: {os.path.join(savepath, f'{pred_filename}_metric_frame.xlsx')}")


    # Sauvegarder la matrice de confusion dans un fichier Excel
    with pd.ExcelWriter(os.path.join(savepath, f"{pred_filename}_weighted_m.xlsx")) as writer:
        wg.rename(columns =default_dict, index = default_dict, inplace = True)
        wg.to_excel(writer, index=True)
        print(f"Fichier sauvegardé avec succès: {os.path.join(savepath, f'{pred_filename}_weighted_m.xlsx')}")