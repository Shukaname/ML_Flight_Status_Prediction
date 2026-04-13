import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, mean_squared_error, mean_absolute_error
import separate_data as sd



def call_data_preparation(data, size = 1.0):
    X_train_final, X_test_final, y_train_reg, y_test_reg, y_train_class, y_test_class = sd.separate_data(data, size)
    return X_train_final, X_test_final, y_train_reg, y_test_reg, y_train_class, y_test_class


def featureNormalize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    # Pour éviter la division par zéro si une colonne est constante
    sigma[sigma == 0] = 1 
    # Normalisation vectorisée
    X_norm = (X - mu) / sigma

    return X_norm, mu, sigma

def normalisation(X_train, X_test):
    X_train_norm, mu, sigma = featureNormalize(X_train)
    X_test_norm = (X_test - mu) / sigma

    # Ajout de l'intercept
    X_train_final = np.c_[np.ones(X_train_norm.shape[0]), X_train_norm]
    X_test_final = np.c_[np.ones(X_test_norm.shape[0]), X_test_norm]

    return X_train_final, X_test_final

def computeCostMulti(X, y, theta):
    m = len(y)
    diff = X @ theta - y
    J = (1 / (2 * m)) * np.dot(diff, diff)
    return J

def gradientDescentMulti(X, y, theta, alpha, num_iters):
    m = len(y)
    theta = theta.copy()
    J_history = []
    for i in range(num_iters):
        theta -= alpha * (X.T @ (X @ theta - y)) / m
        J_history.append(computeCostMulti(X, y, theta))
    return theta, J_history

def visualisation_conv(J_history):
    plt.figure(figsize=(10, 5))
    plt.plot(J_history, '-b', lw=2)
    plt.title("Convergence de la descente de gradient")
    plt.xlabel('Nombre d\'itérations')
    plt.ylabel('Coût J')
    plt.grid(True)
    plt.show()

def predict_and_evaluate(X_test, y_test_reg, theta_final, seuil=5):
    predictions = X_test @ theta_final
    

    #-------------------------Regression------------------------------------
    print(f"\n--- ÉVALUATION RÉGRESSION (Seuil {seuil} min) ---")

    # On affiche les 5 premières prédictions vs la réalité
    comparaison_reg = pd.DataFrame({
        'Prédiction (min)': predictions[:5],
        'Réalité (min)': y_test_reg[:5]
    })
    print(f"\n--- Comparaison Regression (Seuil de {seuil}min) ---")
    print(comparaison_reg)

    rmse = np.sqrt(mean_squared_error(y_test_reg, predictions))
    mae = mean_absolute_error(y_test_reg, predictions)
    accuracy_reg = np.mean(np.abs(predictions - y_test_reg) <= seuil) * 100

    print(f"Accuracy de la regression pour un seuil de {seuil} minutes: {accuracy_reg:.2f} %")
    print(f"Erreur moyenne (MAE) : {mae:.2f} min | Erreur quadratique (RMSE) : {rmse:.2f} min")

    #-------------------------Classification------------------------------------

    pred_is_delay = (predictions > seuil).astype(int)
    true_is_delay = (y_test_reg > seuil).astype(int)
    accuracy_class = np.mean(pred_is_delay == true_is_delay) * 100
    
    print(f"\n--- ÉVALUATION CLASSIFICATION (Seuil de {seuil} min) ---")

    # On affiche les 5 premiers pour comparer avec la régression au-dessus
    comparaison_class = pd.DataFrame({
        'Prédiction': pred_is_delay[:5],
        'Réalité': true_is_delay[:5],
        'Correct ?': (pred_is_delay[:5] == true_is_delay[:5])
    })

    # Remplacer les 0/1 par du texte pour que ce soit plus parlant
    comparaison_class['Prédiction'] = comparaison_class['Prédiction'].replace({1: "Retardé", 0: "À l'heure"})
    comparaison_class['Réalité'] = comparaison_class['Réalité'].replace({1: "Retardé", 0: "À l'heure"})

    print(f"\n--- Comparaison Classification (Seuil de {seuil}min) ---")
    print(comparaison_class)
    
    cm = confusion_matrix(true_is_delay, pred_is_delay)
    print("\nMatrice de Confusion :")
    print(f"Vrais Négatifs (À l'heure) : {cm[0,0]} | Faux Positifs : {cm[0,1]}")
    print(f"Faux Négatifs : {cm[1,0]} | Vrais Positifs (Retards détectés) : {cm[1,1]}")

    print(f"\nAccuracy de la classification pour un seuil de {seuil} minutes: {accuracy_class:.2f} %")
    
    
    return predictions