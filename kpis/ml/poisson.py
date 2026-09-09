"""
Régression de Poisson pour l'estimation du volume attendu.

Les cibles traitées (nombre de crédits, de cartes...) sont des comptages,
non des variables continues. Le contournement usuel — log(x+1) puis
régression linéaire — traite pourtant la variable comme si elle pouvait
prendre n'importe quelle valeur réelle, y compris négative, ce qu'un
comptage ne peut jamais être.

Le modèle de Poisson postule directement :

    Y_i ~ Poisson(mu_i),    log(mu_i) = beta_0 + beta^T x_i

Deux propriétés en découlent, appropriées à des comptages :

  - la prédiction est nécessairement positive, par construction du lien
    logarithmique, sans troncature a posteriori ;
  - la variance croît avec la moyenne (Var = mu), conforme à ce
    qu'on observe sur des comptages réels — une agence à 300 crédits
    varie naturellement plus, en absolu, qu'une agence à 30.

Une régularisation L2 est ajoutée par prudence : avec deux variables
seulement et une trentaine d'observations, un GLM non régularisé reste
correctement identifié, mais la régularisation stabilise l'estimation
si les deux prédicteurs sont corrélés entre eux.
"""

import numpy as np

from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_poisson_deviance


def ajuster_poisson(X, y_comptages, alpha=0.5):
    """
    Ajuste un GLM Poisson en validation leave-one-out.

    X : variables explicatives standardisées.
    y_comptages : cible en comptages bruts, non transformée — le lien
    logarithmique est interne au modèle, aucun log1p n'est appliqué en amont.

    Renvoie les prédictions en échelle log (mu prédit passé au
    logarithme), afin de rester directement comparables aux prédictions
    des autres modèles de la comparaison, tous exprimés sur cette échelle.
    """
    y = np.asarray(y_comptages, dtype=float)
    if np.any(y < 0):
        return None

    validation = LeaveOneOut()
    modele = PoissonRegressor(alpha=alpha, max_iter=500)

    try:
        predictions_mu = cross_val_predict(modele, X, y, cv=validation)
    except Exception:
        return None

    predictions_mu = np.maximum(predictions_mu, 1e-6)
    predictions_log = np.log(predictions_mu)

    try:
        deviance = float(mean_poisson_deviance(np.maximum(y, 1e-6), predictions_mu))
    except Exception:
        deviance = None

    return {
        'predictions_log': predictions_log,
        'deviance': deviance,
    }


def r2_pseudo(y_log, predictions_log):
    """
    R² usuel calculé sur l'échelle log des prédictions, pour comparer
    ce modèle aux trois autres candidats sur une base commune.

    Ce n'est pas le pseudo-R² propre au GLM (fondé sur la vraisemblance),
    mais un indicateur de comparaison directe : les quatre modèles sont
    ainsi classés selon le même critère.
    """
    y = np.asarray(y_log, dtype=float)
    pred = np.asarray(predictions_log, dtype=float)
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot