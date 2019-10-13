
import os
import features as f
from tqdm import tqdm
from skimage import io
import matplotlib.pyplot as plt

import pandas as pd

def creationRecordId(path):
    """
    Création des listes qui servirons à créer les lignes de notre fichier CSV
    
    :param type: string
    :rtype:  List[List[],...]
    """
    L = os.listdir(path)
    laListe = []
    
    for i in tqdm(range(len(L))):
        attrib =  f.calcul_attributs(path + "/" + L[i])
        laListe.append( [L[i] ,attrib[0][0],attrib[0][1],attrib[0][2],attrib[1][0],attrib[1][1],attrib[1][2],attrib[2]])
    return (laListe)


def creation_et_fusion_csv(liste):
    """
    crée un csv à partir d'une liste contenant les records_id et leurs features
    :param type: List[List[]...]
    :rtype: None
    """
    df = pd.DataFrame(liste,columns=["record_id","moyR","moyV","moyB","EctR","EctV","EctB","ProportionBlanc"])
    df.to_csv("info_image.csv",index=False)
    df_features = pd.read_csv("info_image.csv")
    df_csv = pd.read_csv("data_clean/train/info_boat.csv")
    df_features["record_id_tmp"] = df_features["record_id"]
    df_csv["record_id_tmp"] = df_csv["image_clean_path"].apply(lambda x: x.split("/")[-1])
    del df_features["record_id"]
    df_merged = df_features.set_index("record_id_tmp").join(df_csv.set_index("record_id_tmp")).reset_index()
    del df_merged["record_id_tmp"]
    df_merged.to_csv("info_merged.csv",index=False)
    
    
if __name__=="__main__":
    leCSVprimaire=creationRecordId("data_clean/train/image")
    creation_et_fusion_csv(leCSVprimaire)
    
    