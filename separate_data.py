import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def separate_data(data, size=1.0):

    #reduction taille des données pour limitations machine
    #on commence par mélanger les données
    data_shuffled = data.sample(frac=1, random_state=42).reset_index(drop=True)
    print("La matrice de données a été mélangée")
    data_mini = data_shuffled.sample(frac=size, random_state=42)
    print(f"La matrice de données a été réduite à {size*100}% de sa taille initiale")

    X_df = data_mini.drop(columns=['ArrDelay', 'IsDelay'])
    y_delay_time = data_mini['ArrDelay'] #combien de temps de delay
    y_is_delay = data_mini['IsDelay'] #est-ce que le vol est en retard ou pas

    y_delay_time = y_delay_time.to_numpy()
    y_is_delay = y_is_delay.to_numpy()

    #on transforme les conlonnes de texte en variables numeriques
    #la colonne airline va etre transformée en plusieurs colonne de 0 ou de 1 (une pour chaque compagnie aerienne)
    #pour l'airline Delta, la colonne aura des 1 si l'avion est opéré par Delta et des 0 sinon
    X_df_encoded = pd.get_dummies(X_df, drop_first=True, dtype=int)
    X_df_encoded = X_df_encoded.to_numpy()
    print(f"la taille de la matrice de data utilisée est : {X_df_encoded.shape}") #print de la taille de la matrice de donnée

    # Split 
    X_train, X_test, y_train_delay_time, y_test_delay_time= train_test_split(
        X_df_encoded, y_delay_time, test_size=0.2, random_state=42 )
    
    X_train, X_test, y_train_is_delay, y_test_is_delay= train_test_split(
        X_df_encoded, y_is_delay, test_size=0.2, random_state=42 )

    return X_train, X_test, y_train_delay_time, y_test_delay_time, y_train_is_delay, y_test_is_delay