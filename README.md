

# Rapport Intelligence artificielle - 5MA 

- Crappe Martin - 14060 
- Maxime Desaintjean - Erasmus

Notre objectif était d'établir une structure CNN dont l'efficacité (temps d'apprentissage/voiture en fonctionnement) soit la plus élevée possible. Il s'agissait donc d'évaluer un algorithme très performant mais coûteux en calculs et de s'en inspirer afin de d'établir une structure spécifique à notre application.

Cette étude s'est faite avec l'outil tensorboard pour visualiser les résultats.

Le temps d'apprentissage à pu être divisé de **50%** (dont l'algorithme fait rouler la voiture!).

## Description des fichiers et répertoires de la repository

### **Fichiers**

1. **model.py** : Fichier contenant notre structure CNN issue de nos recherches et modifications
2. **drive.py** : Algorithme utilisé pour faire fonctionner le modèle généré *(crédits à [naokishibuya](https://github.com/naokishibuya) )* 
3. **model-004.h5** : Modèle généré par notre architecture CNN
4. **utils.py** : Ensemble de fonctions nécessaire pour lancer un apprentissage avec le model.py

### Répertoires

1. **logs** : Ensemble des données de performance enregistrées lors des apprentissages afin de les afficher avec l'outil tensorboard
2. **img** : Images utilisées pour le readme



## Comment démarrer la simulation

Le meilleur modèle généré depuis notre architecture neuronale se nomme `model-004.h5`

Pour le lancer via l'invite de commande, à condition d'avoir les librairies nécessaires :

`python drive.py model-004.h5`

## Algorithmes et bibliothèques utilisées

### Algorithme

La structure d'inspiration est le réseau neuronal convolutif développé par NVDIA.

Notre architecture finale est la suivante :

```python
model.add(Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE))
model.add(Conv2D(16, 5, 5, activation='elu', subsample=(4, 4)))
model.add(Conv2D(32, 3, 3, activation='elu', subsample=(2, 2)))
model.add(Conv2D(64, 3, 3, activation='elu'))
model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='elu'))
model.add(Dense(16, activation='elu'))
model.add(Dense(1))
```

Où l'amélioration d'efficacité est réalisée sur les couches suivantes :

- convolution 2D : couches supprimées, subsamples augmentées et taille des filtres plus petits (les filtres sautent plus de pixels entre chaque application) : cela permet de réduire plus rapidement la taille des fenêtres et de diminuer le temps de traitement.
- Dense : Diminution du nombre des couches et de neurones (modifications assez limitées car peu impactant sur l'efficacité)
- Maxpooling [couche supplémentaire] : nouvelle couche qui permet d'appliquer des filtres sur des images en un coup : permet d'extraire rapidement les données importantes et de réduire rapidement les fenêtres.

### Bibiliothèques

#### Outil d'analyse

Tensorboard nous a permis d'analyser les résultats de nos apprentissages :



<img src="img\1561109877519.png" style="zoom:60%" />

#### Entraînement

Les bibliothèques utiles pour la construction de l'architecture et l'entraînement du modèle :

1. **Panda** : Utile pour analyse de données (utilisée pour la gestion du fichier csv dans notre cas)
2. **Numpy** : Utile pour la manipulation de données scientifiques (manipulation des matrices dans notre cas)
3. **sklearn** : Facile l'apprentissage machine (la division du dataset dans notre cas)
4. **Keras** : API de haut niveau pour les neural networks