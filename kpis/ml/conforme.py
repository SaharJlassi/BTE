"""
Intervalles de prédiction par inférence conforme.

Un modèle renvoie une estimation ponctuelle : « niveau attendu = 147 ».
Cette valeur est nécessairement approchée, et présenter un écart au
point central comme un verdict de performance revient à ignorer
l'erreur du modèle.

L'inférence conforme (Vovk et al., 2005) construit un intervalle
assorti d'une garantie de couverture valable en échantillon fini :
pour un niveau fixé à 90 %, au moins 90 % des observations tombent
dans leur intervalle. Aucune hypothèse n'est faite sur la loi des
erreurs, seulement leur échangeabilité.

Variante retenue : jackknife+ (Barber et al., 2021), qui réutilise
les résidus de la validation leave-one-out déjà calculée. Sur un
échantillon d'une trentaine d'observations, cette variante exploite
toutes les données, là où le conformal split en sacrifierait la
moitié pour la calibration.

Le quantile est pris au niveau ceil((n+1)(1-alpha))/n plutôt que
(1-alpha) : cette correction, d'apparence mineure, est ce qui rend
la garantie exacte et non asymptotique.
"""

import numpy as np


def quantile_conforme(residus_absolus, alpha=0.10):
    """
    Quantile des résidus assurant la couverture visée.

    La correction en (n+1)/n compense le fait que l'observation à
    prédire n'appartient pas à l'échantillon de calibration.
    """
    n = len(residus_absolus)
    if n < 3:
        return float(np.max(residus_absolus)) if n else 0.0

    niveau = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
    return float(np.quantile(residus_absolus, niveau, method='higher'))


def intervalles_conformes(y, predictions, alpha=0.10):
    """
    Renvoie (borne_basse, borne_haute, demi_largeur) en échelle log.

    y et predictions sont exprimés en log ; l'intervalle est donc
    symétrique en log, ce qui correspond à un intervalle multiplicatif
    sur l'échelle d'origine — cohérent avec des comptages dont la
    dispersion croît avec le niveau.
    """
    residus = np.abs(np.asarray(y) - np.asarray(predictions))
    q = quantile_conforme(residus, alpha)
    return predictions - q, predictions + q, q


def couverture_empirique(y, basse, haute):
    """
    Part des observations effectivement contenues dans leur intervalle.
    Sert de contrôle : cette valeur doit avoisiner le niveau visé.
    """
    dedans = (np.asarray(y) >= basse) & (np.asarray(y) <= haute)
    return float(np.mean(dedans))


def expliquer_conforme(resultat):
    """Commentaires en langage courant sur les intervalles."""
    if not resultat or resultat.get('insuffisant'):
        return []

    conforme = resultat.get('conforme')
    if not conforme:
        return []

    points = [
        "Le niveau attendu d'une agence n'est jamais exact : le modèle comporte "
        "une marge d'erreur, qui est ici rendue visible sous forme de fourchette.",
        "Une agence n'est considérée en écart que si son résultat sort de sa "
        "fourchette — un résultat à l'intérieur reste conforme à l'attendu.",
    ]

    n_au_dessus = conforme['n_au_dessus']
    n_en_dessous = conforme['n_en_dessous']
    n_conforme = conforme['n_conforme']

    if n_au_dessus:
        points.append(f"{n_au_dessus} agence(s) dépassent le haut de leur fourchette.")
    if n_en_dessous:
        points.append(f"{n_en_dessous} agence(s) tombent sous le bas de leur fourchette.")
    points.append(
        f"{n_conforme} agence(s) se situent dans leur fourchette : leur résultat "
        "est conforme à ce que le modèle prévoyait."
    )

    points.append(
        f"Largeur typique de la fourchette : de −{conforme['baisse_pct']} % à "
        f"+{conforme['hausse_pct']} % autour du niveau attendu."
    )
    return points