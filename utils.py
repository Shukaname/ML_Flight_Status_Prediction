import numpy as np 
import math
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, mean_squared_error, mean_absolute_error


def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def normalize(X) : 
  mu = np.mean(X, axis=0)
  sigma = np.std(X, axis=0)
  X_train_norm = (X - mu) / sigma
  

def import_data(data,size=1):
  """

  Args:
      data (_type_): The dataset itself.
      size (int, optional): The percentage of the data to utilize. Defaults to 1.

  Returns:
    A sample of the dataset in a numpy array format.
  """

  data_mini = data.sample(frac=size, random_state=42)

  X_df = data_mini.drop(columns=['ArrDelay', 'IsDelay'])

  #on transforme les conlonnes de texte en variables numeriques
  #la colonne airline va etre transformée en plusieurs colonne de 0 ou de 1 (une pour chaque compagnie aerienne)
  #pour l'airline Delta, la colonne aura des 1 si l'avion est opéré par Delta et des 0 sinon
  X_df_encoded = pd.get_dummies(X_df, drop_first=True, dtype=int)
  X_df_encoded = X_df_encoded.to_numpy()
  print(f"la taille de la matrice de data utilisée est : {X_df_encoded.size}")

  return X_df_encoded
    

