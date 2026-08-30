"""
Détection d'anomalies avec contrôle du taux de fausses découvertes.

Limite de l'approche usuelle : Isolation Forest exige un taux de
contamination fixé a priori. En demandant 12 %, on obtient 12 % —
le nombre d'anomalies est alors une décision de l'analyste, non un
résultat des données.

Démarche retenue :

  1. Score d'atypie de chaque agence par Isolation Forest, entraîné
     sans hypothèse de contamination.
  2. Distribution de référence obtenue par permutation : les valeurs
     de chaque produit sont redistribuées aléatoirement entre agences,
     ce qui détruit les profils tout en préservant les distributions
     marginales. Les scores issus de ces jeux permutés décrivent ce
     qu'on observerait si aucune agence n'avait de profil particulier.
  3. p-valeur empirique de chaque agence, avec correction de Laplace
     pour éviter les p-valeurs nulles.
  4. Procédure de Benjamini-Hochberg : sur une trentaine de tests
     simultanés, un seuil de 5 % appliqué individuellement produirait
     en moyenne 1,5 fausse détection. BH ajuste les seuils pour que la
     proportion attendue de fausses découvertes reste sous le niveau visé.

Le nombre d'agences signalées devient ainsi un résultat, éventuellement
nul si le réseau est homogène.
"""

import numpy as np

from sklearn.ensemble import IsolationForest


# Nombre d'arbres par forêt. Volontairement modeste : la procédure de
# permutation requiert plusieurs dizaines d'ajustements, et la précision
# gagnée au-delà ne justifie pas le coût de calcul.
N_ARBRES = 150


def _scores_atypie(donnees, graine=0):
    """
    Score d'atypie de chaque ligne. Valeurs élevées = profil plus isolé.

    Isolation Forest renvoie des scores négatifs (plus bas = plus atypique) ;
    on inverse le signe pour obtenir une statistique de test croissante.
    """
    modele = IsolationForest(
        n_estimators=N_ARBRES,
        max_samples='auto',
        contamination='auto',
        random_state=graine,
        n_jobs=1,
    )
    modele.fit(donnees)
    return -modele.score_samples(donnees)


def _permuter(donnees, generateur):
    """
    Permute indépendamment chaque colonne.

    Chaque produit conserve sa distribution sur le réseau, mais les
    associations entre produits au sein d'une agence sont détruites.
    Le jeu obtenu représente donc l'hypothèse nulle : des agences dont
    le profil ne résulte que du hasard.

    L'indexation par colonne renvoyant une copie en NumPy et non une vue,
    la permutation est calculée puis réaffectée explicitement.
    """
    permutee = donnees.copy()
    n = permutee.shape[0]
    for j in range(permutee.shape[1]):
        permutee[:, j] = permutee[generateur.permutation(n), j]
    return permutee


def benjamini_hochberg(p_valeurs, alpha=0.10):
    """
    Procédure de Benjamini-Hochberg (1995).

    Les p-valeurs sont triées par ordre croissant ; on retient les k
    premières telles que p(k) <= alpha * k / m. Contrôle le taux de
    fausses découvertes au niveau alpha, moins conservateur que
    Bonferroni tout en restant rigoureux.

    Renvoie (rejets, p_ajustees).
    """
    p = np.asarray(p_valeurs, dtype=float)
    m = len(p)
    ordre = np.argsort(p)
    p_triees = p[ordre]

    seuils = alpha * np.arange(1, m + 1) / m
    sous_seuil = p_triees <= seuils

    rejets = np.zeros(m, dtype=bool)
    if sous_seuil.any():
        k = int(np.max(np.where(sous_seuil)[0]))
        rejets[ordre[:k + 1]] = True

    # p-valeurs ajustées : minimum cumulé par la droite, borné à 1
    p_ajustees_triees = p_triees * m / np.arange(1, m + 1)
    p_ajustees_triees = np.minimum.accumulate(p_ajustees_triees[::-1])[::-1]
    p_ajustees_triees = np.minimum(p_ajustees_triees, 1.0)

    p_ajustees = np.empty(m)
    p_ajustees[ordre] = p_ajustees_triees

    return rejets, p_ajustees


def detecter_anomalies_test(donnees, noms_agences, noms_features,
                            n_permutations=40, alpha=0.10):
    """
    Détection d'anomalies par test statistique avec contrôle du FDR.

    n_permutations : nombre de jeux permutés servant de référence.
    Quarante permutations sur une trentaine d'agences fournissent plus
    d'un millier de scores nuls, suffisants pour estimer des p-valeurs
    au seuil de 10 %.

    Renvoie un dictionnaire contenant la liste des agences signalées,
    le détail de toutes les agences, et les paramètres du test.
    """
    n = donnees.shape[0]
    if n < 5:
        return None

    scores = _scores_atypie(donnees)

    # --- Distribution de référence sous hypothèse nulle ---
    generateur = np.random.default_rng(0)
    scores_nuls = []

    for tirage in range(n_permutations):
        permutee = _permuter(donnees, generateur)
        # La graine varie d'un tirage à l'autre : conserver la même
        # ferait dépendre tous les scores nuls d'une unique forêt.
        scores_nuls.append(_scores_atypie(permutee, graine=tirage))

    scores_nuls = np.concatenate(scores_nuls)

    # --- p-valeur empirique de chaque agence ---
    # Correction de Laplace : (1 + nb dépassements) / (1 + taille référence),
    # ce qui évite une p-valeur strictement nulle, non défendable sur un
    # nombre fini de permutations.
    p_valeurs = np.array([
        (1.0 + float(np.sum(scores_nuls >= s))) / (1.0 + len(scores_nuls))
        for s in scores
    ])

    rejets, p_ajustees = benjamini_hochberg(p_valeurs, alpha=alpha)

    noms_etendus = list(noms_features) + ["Taille (log)"]

    lignes = []
    for i, nom in enumerate(noms_agences):
        ecarts = donnees[i]
        j = int(np.argmax(np.abs(ecarts)))
        lignes.append({
            'agence': nom,
            'score': round(float(scores[i]), 4),
            'p_valeur': round(float(p_valeurs[i]), 4),
            'p_ajustee': round(float(p_ajustees[i]), 4),
            'signalee': bool(rejets[i]),
            'variable': noms_etendus[j],
            'ecart': round(float(ecarts[j]), 2),
            'sens': 'supérieur' if ecarts[j] > 0 else 'inférieur',
        })

    lignes.sort(key=lambda l: l['p_ajustee'])
    signalees = [l for l in lignes if l['signalee']]

    return {
        'lignes': lignes,
        'signalees': signalees,
        'n_signalees': len(signalees),
        'n_agences': n,
        'alpha': alpha,
        'n_permutations': n_permutations,
        'n_scores_nuls': len(scores_nuls),
        'p_min': round(float(p_valeurs.min()), 4),
        'seuil_naif': int(np.sum(p_valeurs <= alpha)),
    }


def _qualifier(ecart):
    a = abs(ecart)
    if a >= 3:
        return "très nettement"
    if a >= 2:
        return "nettement"
    if a >= 1:
        return "sensiblement"
    return "légèrement"


def expliquer_anomalies(resultat):
    """Commentaires en langage courant."""
    if not resultat:
        return []

    points = [
        "Chaque agence est comparée à ce qu'on observerait si les profils "
        "d'activité étaient dus au pur hasard.",
    ]

    n_sig = resultat['n_signalees']
    alpha_pct = int(resultat['alpha'] * 100)

    if n_sig == 0:
        points.append(
            "Aucune agence ne se démarque de façon statistiquement établie : "
            "les différences observées entre agences restent dans ce que le "
            "hasard peut produire."
        )
    elif n_sig == 1:
        points.append(
            f"Une seule agence a un profil réellement atypique : "
            f"{resultat['signalees'][0]['agence']}."
        )
    else:
        noms = ", ".join(l['agence'] for l in resultat['signalees'][:4])
        points.append(f"{n_sig} agences ont un profil réellement atypique : {noms}.")

    naif = resultat['seuil_naif']
    if naif > n_sig:
        points.append(
            f"Sans correction, {naif} agences auraient été signalées. "
            f"En testant {resultat['n_agences']} agences à la fois, certaines "
            "ressortent par simple effet du hasard : la correction appliquée "
            "écarte ces fausses alertes."
        )

    points.append(
        f"Le nombre d'agences signalées n'est pas fixé à l'avance : il découle "
        f"des données, avec un risque d'erreur maîtrisé à {alpha_pct} %."
    )
    return points


def lignes_lisibles(resultat):
    """Phrases descriptives pour les agences signalées."""
    if not resultat:
        return []

    return [
        {
            'agence': l['agence'],
            'phrase': (
                f"Fait {_qualifier(l['ecart'])} plus de {l['variable'].lower()} "
                "que la moyenne."
                if l['ecart'] > 0 else
                f"Fait {_qualifier(l['ecart'])} moins de {l['variable'].lower()} "
                "que la moyenne."
            ),
            'p_ajustee': l['p_ajustee'],
        }
        for l in resultat['signalees']
    ]