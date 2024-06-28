# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from metricsyso import read_file_class, num_points, compute_confusion_matrix, Metrics
from tqdm import tqdm

def show_metrics(truthpath, predpath, savepath):
    # Créez le répertoire de sauvegarde s'il n'existe pas
    os.makedirs(savepath, exist_ok=True)
    pred_filename = os.path.splitext(os.path.basename(predpath))[0]
    
    conv_dict2 = {"1": 0, "2": 1, "3": 0, "4": 2, "5": 2, "6": 3, "7": 6,  "12": 0, "13": 0, "15": 0, "50": 0,
        "51": 0, "52": 0, "53": 0,"54": 0, "55": 0,  "64": 0,"72": 0,"101": 0,"102": 4,"103": 4, "104": 4,
        "105": 4,"106": 0,"107": 0,"108": 0,"109": 0, "110": 0,"111": 0, "120": 0, "121": 0,"150": 0,
        "151": 0,"152": 5,"153": 5,"154": 5, "155": 5,"160": 0,"161": 0,"185": 0,"196": 4,"197": 4,
        "200": 5,"201": 5,"202": 5,"211": 5,"212": 5, "213": 5,"221": 5,"222": 5, "223": 5
        }
    model2 = {"0":1, "1":2, "2":5, "3":6, "4":196, "5":211}
    direct_conv2={k:j for k, v in conv_dict2.items() for  i, j in model2.items()  if int(i) == v }


    # Exemple d'utilisation
    rte2_pred=read_file_class(predpath)
    rte2_truth=read_file_class(truthpath)
    pred_class = num_points(rte2_pred)
    gtruth_class = num_points(rte2_truth)


    gtruth_cleanead = [direct_conv2.get(str(k)) if direct_conv2.get(str(k)) is not None else k for k in gtruth_class]

    rte2_ground_truth =np.array(gtruth_cleanead)
    rte2_predictions = np.array(pred_class)
    
    print("len gtruth_class",len(gtruth_class))
    print("gtruth unique classes",np.unique(gtruth_class))    
    print("") 
    print("len rte2_gtruth_class", len(rte2_ground_truth))
    print("unique gtruth classes merged classe", np.unique(rte2_ground_truth))
    
    print("")
    print(f"len pred_class{len(rte2_predictions)}")
    print(f" unique classe predictions{np.unique(rte2_predictions)}")
    print("")

    unique_classes =  np.unique(np.concatenate((rte2_ground_truth, rte2_predictions)))
    clas_name = [str(name) for name in tqdm(unique_classes)]

    default_dict = {"1":"default","2":"ground","5": "vegetation", 
                "6":"building","7": "wall", 
                "196": "structure","211":"cable"
                }



    conf_matrix = compute_confusion_matrix(rte2_ground_truth, rte2_predictions)
                
    #conf_matrix = confusion_matrix(rte2_ground_truth, rte2_predictions)

    # Sauvegarder la matrice de confusion dans un fichier Excel
    with pd.ExcelWriter(os.path.join(savepath, f"conf_matrix_{pred_filename}.xlsx")) as writer:
        #conf_matrix = pd.DataFrame(conf_matrix, index=clas_name, columns=clas_name)
        conf_matrix.rename(columns = default_dict, index = default_dict, inplace = True)
        conf_matrix.to_excel(writer, index=True)
        print(f"Fichier sauvegardé avec succès: {os.path.join(savepath, f'conf_matrix_{pred_filename}.xlsx')}")

    # Afficher et sauvegarder la matrice de confusion
        
    """ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=clas_name)
    mask = np.eye(len(conf_matrix))

    fig, ax = plt.subplots(figsize=(30, 10))
    #ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=clas_name).plot(ax=ax)
    ax.imshow(mask, cmap='Blues', interpolation='nearest', alpha=0.5)

    plt.savefig(os.path.join(savepath, f"conf_matrix_{pred_filename}.png"))
    plt.show()"""
    
    
    #========================== metrics=============================================#
    
    metrics=Metrics(rte2_ground_truth,rte2_predictions)
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