# Projet de Régression Linéaire avec Réseau de Neurones
## Vue d'ensemble
Ce projet implémente et compare deux approches de régression linéaire pour prédire une variable cible basée sur deux caractéristiques (features):

 1. Réseau de neurones simple utilisant TensorFlow/Keras
 2. Implémentation "from scratch" de la régression linéaire avec descente de gradient

Le projet travaille avec un jeu de données synthétique généré selon la formule: ``` y = 2*X1 + 3*X2 + 5 + bruit ```

## Contenu du projet

  - devoir_regression_complete.ipynb: Notebook Jupyter contenant tout le code et les analyses

## Prérequis
Pour exécuter ce projet, vous aurez besoin des bibliothèques Python suivantes:

    - NumPy
    - Matplotlib
    - Pandas
    - TensorFlow
    - Scikit-learn

Vous pouvez les installer avec pip:

``` pip install numpy matplotlib pandas tensorflow scikit-learn ```

# Structure du projet
Le notebook est organisé en trois sections principales:
# Section 1: Préparation des données

 - Génération d'un jeu de données synthétique basé sur une relation linéaire avec du bruit
 - Division des données en ensembles d'entraînement (80%) et de test (20%)

## Section 2: Réseau de neurones avec TensorFlow

 - Architecture du réseau: une simple couche de sortie sans couche cachée
 - Compilation et entraînement du modèle
 - Analyse des paramètres du réseau (poids et biais)
 - Visualisation de l'architecture du réseau
 - Prédictions et évaluation des performances sur les ensembles d'entraînement et de test
 - Analyse des résidus

## Section 3: Régression linéaire "from scratch"

Implémentation d'une classe personnalisée de régression linéaire
Utilisation de l'algorithme de descente de gradient pour ajuster les paramètres
Comparaison des performances avec l'approche basée sur les réseaux de neurones

## Points clés de l'implémentation
## Réseau de neurones (TensorFlow/Keras)
``` python
    model_nn = Sequential([
        Dense(units=1, activation='linear', input_shape=(2,))
    ])
    model_nn.compile(optimizer='adam', loss='mse', metrics=['mse'])
    history = model_nn.fit(X_train, y_train, epochs=100, batch_size=8, verbose=1) 
```
## Régression linéaire "from scratch"
``` python
    class LinearRegressionScratch:
    def __init__(self):
        self.weights = None
        self.bias = None
    
    def fit(self, X, y, learning_rate=0.01, epochs=1000):
        # Initialisation des paramètres
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Algorithme de descente de gradient
        for i in range(epochs):
            # Prédictions avec les paramètres actuels
            y_pred = np.dot(X, self.weights) + self.bias
            
            # Calcul des gradients
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # Mise à jour des paramètres
            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db
```

## Évaluation et métriques
Le projet utilise les métriques suivantes pour évaluer les performances des modèles:

 - Mean Squared Error (MSE): Mesure l'erreur quadratique moyenne entre les prédictions et les valeurs réelles
 - Coefficient de détermination (R²): Indique la proportion de variance expliquée par le modèle

## Visualisations
Le projet inclut plusieurs visualisations, notamment:

  - Architecture du réseau de neurones
  - Graphiques des résidus pour les ensembles d'entraînement et de test
  - Comparaison des prédictions des deux modèles
  - Évolution de la perte pendant l'entraînement

## Discussion
Les deux implémentations devraient théoriquement converger vers des solutions similaires, car:

  1. Le problème est intrinsèquement linéaire
  2. Le réseau de neurones n'a pas de couche cachée et utilise une activation linéaire
  3. Les deux approches minimisent l'erreur quadratique moyenne

Les différences de performance observées peuvent être attribuées à:

   - Différences dans les techniques d'optimisation (Adam vs descente de gradient simple)
   - Différences dans les hyperparamètres (learning rate, batch size, nombre d'époques)
   - Initialisation des poids

## Extensions possibles

   - Ajout de couches cachées au réseau de neurones pour modéliser des relations non linéaires
   - Expérimentation avec différentes fonctions d'activation
   - Application à des jeux de données réels
   - Implémentation de techniques de régularisation
   - Validation croisée pour une évaluation plus robuste