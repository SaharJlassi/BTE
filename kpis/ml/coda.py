"""
Traitement compositionnel des profils d'activité.

Les parts d'activité d'une agence somment à 1 : elles vivent sur un
simplexe, non dans un espace euclidien. Les traiter comme des variables
ordinaires induit trois biais :

  - des corrélations négatives artificielles, puisqu'une part qui monte
    force les autres à baisser ;
  - des distances trompeuses, un passage de 5 % à 10 % pesant autant
    qu'un passage de 50 % à 55 % alors que le premier double
    l'importance du produit et le second l'augmente d'un dixième ;
  - une géométrie contrainte, incompatible avec les hypothèses de l'ACP
    et du K-Means.

Aitchison (1982) propose de raisonner sur les rapports entre parts, qui
échappent à la contrainte de somme. La transformation en log-rapport
centré (CLR) rapporte chaque part à la moyenne géométrique de la
composition, puis en prend le logarithme.

Les zéros posent problème au logarithme. Ils sont remplacés par
multiplication bayésienne (Martín-Fernández et al., 2003) : une valeur
faible leur est attribuée, et les parts non nulles sont réduites
proportionnellement, de sorte que leurs rapports mutuels soient conservés.
"""

import numpy as np


def remplacer_zeros(parts, delta=None):
    """
    Remplace les parts nulles par une valeur faible, sans altérer les
    rapports entre les parts observées.

    delta : valeur attribuée aux zéros. Par défaut 0,65 fois la plus
    petite part strictement positive, seuil usuel de la littérature.
    """
    parts = np.asarray(parts, dtype=float).copy()
    positives = parts[parts > 0]

    if positives.size == 0:
        return np.full_like(parts, 1.0 / parts.shape[1])

    if delta is None:
        delta = 0.65 * float(positives.min())

    for i in range(parts.shape[0]):
        ligne = parts[i]
        zeros = ligne == 0
        n_zeros = int(zeros.sum())
        if n_zeros == 0:
            continue

        masse_ajoutee = n_zeros * delta
        if masse_ajoutee >= 1.0:
            # Cas dégénéré : composition presque entièrement nulle.
            ligne[:] = 1.0 / len(ligne)
            continue

        # Les parts observées sont réduites au prorata, ce qui préserve
        # exactement leurs rapports deux à deux.
        ligne[~zeros] *= (1.0 - masse_ajoutee)
        ligne[zeros] = delta

    # Renormalisation, les arrondis pouvant écarter la somme de l'unité
    return parts / parts.sum(axis=1, keepdims=True)


def transformation_clr(parts):
    """
    Log-rapport centré.

    Chaque part est rapportée à la moyenne géométrique de sa composition,
    puis passée au logarithme. Les distances euclidiennes calculées sur
    les coordonnées obtenues correspondent à la distance d'Aitchison
    entre compositions.

    Les colonnes somment à zéro par construction, ce qui rend la matrice
    singulière : sans conséquence pour l'ACP ou le clustering, mais à
    garder à l'esprit pour toute inversion.
    """
    parts = remplacer_zeros(parts)
    logs = np.log(parts)
    moyennes = logs.mean(axis=1, keepdims=True)
    return logs - moyennes


def preparer_donnees_coda(matrice):
    """
    Prépare l'espace d'analyse en traitant correctement la nature
    compositionnelle des profils.

    Renvoie (parts, donnees_std), de même signature que la préparation
    usuelle afin de rester interchangeable avec elle.

    Deux blocs sont juxtaposés :
      - les coordonnées CLR, qui décrivent la structure d'activité ;
      - la taille en échelle logarithmique, information distincte que la
        transformation compositionnelle évacue par construction.

    Chaque bloc est standardisé séparément, puis la taille est ramenée au
    poids d'une variable unique : sans cette pondération, elle pèserait
    autant que l'ensemble du profil dans les distances.
    """
    totaux = matrice.sum(axis=1, keepdims=True)
    totaux[totaux == 0] = 1.0
    parts = matrice / totaux

    coordonnees = transformation_clr(parts)

    # Standardisation du bloc compositionnel
    ecarts = coordonnees.std(axis=0)
    ecarts[ecarts < 1e-9] = 1.0
    coordonnees_std = (coordonnees - coordonnees.mean(axis=0)) / ecarts

    # Taille, standardisée séparément
    taille = np.log1p(matrice.sum(axis=1)).reshape(-1, 1)
    ecart_taille = float(taille.std()) or 1.0
    taille_std = (taille - taille.mean()) / ecart_taille

    donnees_std = np.hstack([coordonnees_std, taille_std])
    return parts, donnees_std


def comparer_representations(matrice, etiquettes_fn):
    """
    Compare la qualité du regroupement selon la représentation retenue :
    parts brutes standardisées d'un côté, coordonnées CLR de l'autre.

    etiquettes_fn : fonction prenant un tableau standardisé et renvoyant
    (etiquettes, score). Permet d'appliquer la même procédure de
    segmentation aux deux espaces.

    Sert à vérifier que le traitement compositionnel apporte réellement
    un gain sur ces données, plutôt que de l'affirmer par principe.
    """
    from sklearn.preprocessing import StandardScaler

    totaux = matrice.sum(axis=1, keepdims=True)
    totaux[totaux == 0] = 1.0
    parts = matrice / totaux
    taille = np.log1p(matrice.sum(axis=1)).reshape(-1, 1)

    classique = StandardScaler().fit_transform(np.hstack([parts, taille]))
    _, compositionnel = preparer_donnees_coda(matrice)

    resultats = {}
    for nom, donnees in [('classique', classique), ('coda', compositionnel)]:
        try:
            _, score = etiquettes_fn(donnees)
            resultats[nom] = round(float(score), 3)
        except Exception:
            resultats[nom] = None

    return resultats