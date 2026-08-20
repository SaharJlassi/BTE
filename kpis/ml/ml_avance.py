"""
Analyses ML avancées du réseau d'agences.

Trois modules complémentaires à analytics.py :

  1. Performance   : régression du volume attendu, puis lecture des résidus
                     pour identifier les agences en sur/sous-performance
  2. Robustesse    : comparaison d'algorithmes de clustering et mesure de
                     stabilité par bootstrap (indice de Rand ajusté)
  3. Classement    : score composite multicritère par méthode TOPSIS

Contexte : 32 observations. Toutes les méthodes retenues sont adaptées
aux petits échantillons (régularisation, validation leave-one-out).
"""

import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import (
    r2_score, mean_absolute_error, adjusted_rand_score,
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
)
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture

from .analytics import construire_matrice, preparer_donnees


# Cibles proposées pour l'analyse de performance
CIBLES = [
    ('credits', 'Crédits'),
    ('cartes', 'Cartes'),
    ('placements', 'Placements'),
    ('comptes', 'Comptes clients'),
]


# ------------------------------------------------------------------
# Repérage des domaines dans la matrice
# ------------------------------------------------------------------

def _domaines(noms_features):
    """
    Répartit les colonnes de la matrice par domaine métier,
    d'après le libellé généré dans analytics.construire_matrice.
    """
    groupes = {'credits': [], 'cartes': [], 'placements': [], 'tpe': [], 'comptes': []}

    for j, nom in enumerate(noms_features):
        if nom.startswith('Crédit') or nom.startswith('Engagement'):
            groupes['credits'].append(j)
        elif nom.startswith('Carte'):
            groupes['cartes'].append(j)
        elif nom.startswith('Placement'):
            groupes['placements'].append(j)
        elif nom.startswith('Ventes TPE'):
            groupes['tpe'].append(j)
        else:
            groupes['comptes'].append(j)

    return groupes


# ------------------------------------------------------------------
# 1. Performance : régression et analyse des résidus
# ------------------------------------------------------------------

def analyser_performance(annee, cible='credits'):
    """
    Compare le volume réalisé d'un domaine d'activité au volume attendu.

    Trois modèles sont mis en concurrence, du plus simple au plus complexe :

      1. Ratio de référence : le rapport médian observé dans le réseau
         entre le domaine cible et le volume d'activité total. Un seul
         paramètre, donc robuste même sur un très petit échantillon.
      2. Régression régularisée sur deux variables seulement
         (volume d'activité, orientation professionnelle du portefeuille).
      3. Forêt aléatoire sur ces mêmes deux variables.

    Le choix de deux variables au maximum est imposé par la taille de
    l'échantillon : avec 32 agences, un modèle à quinze variables
    apprend le bruit et se comporte moins bien que la moyenne.

    Le modèle retenu est celui dont le R² en validation leave-one-out
    est le meilleur ; si aucun modèle appris ne dépasse le ratio de
    référence, c'est ce dernier qui est utilisé.
    """
    noms_agences, noms_features, matrice = construire_matrice(annee)
    if len(noms_agences) < 8:
        return None

    groupes = _domaines(noms_features)
    colonnes_cible = groupes.get(cible, [])
    if not colonnes_cible:
        return None

    colonnes_autres = [
        j for domaine, indices in groupes.items()
        if domaine != cible for j in indices
    ]
    if len(colonnes_autres) < 2:
        return None

    volume_cible = matrice[:, colonnes_cible].sum(axis=1)
    volume_autres = matrice[:, colonnes_autres].sum(axis=1)

    if volume_autres.sum() == 0 or volume_cible.sum() == 0:
        return None

    # --- Variables explicatives : deux seulement ---
    # 1. Le volume d'activité de l'agence hors domaine cible
    # 2. L'orientation professionnelle de son portefeuille
    indices_pro = [
        j for j, nom in enumerate(noms_features)
        if 'PM ' in nom or 'morale' in nom.lower() or 'gestion' in nom.lower()
    ]
    volume_pro = matrice[:, indices_pro].sum(axis=1) if indices_pro else np.zeros(len(noms_agences))
    total_ligne = np.maximum(matrice.sum(axis=1), 1.0)
    part_pro = volume_pro / total_ligne

    y = np.log1p(volume_cible)
    X_brut = np.column_stack([np.log1p(volume_autres), part_pro])
    X = StandardScaler().fit_transform(X_brut)

    if np.std(y) < 1e-6:
        return None

    validation = LeaveOneOut()

    # --- Modèle 1 : ratio de référence, validé en leave-one-out ---
    ratios = volume_cible / np.maximum(volume_autres, 1.0)
    predictions_ratio = np.zeros(len(y))
    for i in range(len(y)):
        autres = np.delete(ratios, i)
        ratio_median = float(np.median(autres))
        predictions_ratio[i] = np.log1p(ratio_median * volume_autres[i])

    # --- Modèle 2 : régression régularisée ---
    ridge = RidgeCV(alphas=np.logspace(-2, 3, 30))
    predictions_ridge = cross_val_predict(ridge, X, y, cv=validation)

    # --- Modèle 3 : forêt aléatoire ---
    foret = RandomForestRegressor(
        n_estimators=300, min_samples_leaf=3, max_depth=4, random_state=0
    )
    predictions_foret = cross_val_predict(foret, X, y, cv=validation)

    candidats = [
        {
            'nom': 'Ratio de référence du réseau',
            'r2': round(float(r2_score(y, predictions_ratio)), 3),
            'mae': round(float(mean_absolute_error(y, predictions_ratio)), 3),
            'predictions': predictions_ratio,
        },
        {
            'nom': 'Régression régularisée (2 variables)',
            'r2': round(float(r2_score(y, predictions_ridge)), 3),
            'mae': round(float(mean_absolute_error(y, predictions_ridge)), 3),
            'predictions': predictions_ridge,
        },
        {
            'nom': 'Forêt aléatoire (2 variables)',
            'r2': round(float(r2_score(y, predictions_foret)), 3),
            'mae': round(float(mean_absolute_error(y, predictions_foret)), 3),
            'predictions': predictions_foret,
        },
    ]

    meilleur = max(candidats, key=lambda m: m['r2'])
    predictions = meilleur['predictions']

    residus = y - predictions
    ecart_type = float(np.std(residus)) or 1.0

    lignes = []
    for i, nom in enumerate(noms_agences):
        realise = float(np.expm1(y[i]))
        attendu = float(np.expm1(predictions[i]))
        if attendu <= 0:
            ecart_pct = 0.0
        else:
            ecart_pct = (realise / attendu - 1.0) * 100
        lignes.append({
            'agence': nom,
            'realise': int(round(realise)),
            'attendu': int(round(max(attendu, 0))),
            'ecart_pct': round(float(np.clip(ecart_pct, -100, 500)), 1),
            'residu_std': round(float(residus[i] / ecart_type), 2),
        })

    lignes.sort(key=lambda l: -l['ecart_pct'])

    facteurs = [
        {'nom': "Volume d'activité de l'agence hors " + dict(CIBLES).get(cible, cible).lower()},
        {'nom': "Orientation professionnelle du portefeuille"},
    ]
    sur = [l for l in lignes if l['residu_std'] >= 1.0]
    sous = [l for l in lignes if l['residu_std'] <= -1.0]

    return {
        'cible': cible,
        'cible_libelle': dict(CIBLES).get(cible, cible),
        'modele_retenu': meilleur['nom'],
        'r2': meilleur['r2'],
        'comparaison': [
            {'nom': m['nom'], 'r2': m['r2'], 'mae': m['mae']}
            for m in candidats
        ],
        'lignes': lignes,
        'facteurs': facteurs,
        'sur_performance': sur,
        'sous_performance': sous,
        'n_agences': len(noms_agences),
    }
    """
    Estime le volume attendu d'un domaine d'activité à partir des autres
    caractéristiques de l'agence, puis compare au réalisé.

    L'écart (résidu) distingue les agences qui font mieux ou moins bien
    que ce que leur profil laisserait prévoir. Travail en échelle
    logarithmique : les résidus s'interprètent alors en pourcentage.
    """
    noms_agences, noms_features, matrice = construire_matrice(annee)
    if len(noms_agences) < 8:
        return None

    groupes = _domaines(noms_features)
    colonnes_cible = groupes.get(cible, [])
    if not colonnes_cible:
        return None

    colonnes_explicatives = [
        j for domaine, indices in groupes.items()
        if domaine != cible for j in indices
    ]
    if len(colonnes_explicatives) < 3:
        return None

    # Échelle log : atténue l'effet des écarts de taille entre agences
    y = np.log1p(matrice[:, colonnes_cible].sum(axis=1))
    X = np.log1p(matrice[:, colonnes_explicatives])
    X = StandardScaler().fit_transform(X)

    if np.std(y) < 1e-6:
        return None

    validation = LeaveOneOut()

    # Modèle linéaire régularisé : alpha choisi par validation interne
    ridge = RidgeCV(alphas=np.logspace(-2, 3, 30))
    y_ridge = cross_val_predict(ridge, X, y, cv=validation)

    # Modèle non linéaire, en comparaison
    foret = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=0)
    y_foret = cross_val_predict(foret, X, y, cv=validation)

    modeles = {
        'ridge': {
            'nom': 'Régression régularisée (Ridge)',
            'r2': round(float(r2_score(y, y_ridge)), 3),
            'mae': round(float(mean_absolute_error(y, y_ridge)), 3),
            'predictions': y_ridge,
        },
        'foret': {
            'nom': 'Forêt aléatoire',
            'r2': round(float(r2_score(y, y_foret)), 3),
            'mae': round(float(mean_absolute_error(y, y_foret)), 3),
            'predictions': y_foret,
        },
    }

    meilleur = max(modeles.values(), key=lambda m: m['r2'])
    predictions = meilleur['predictions']

    # Résidus : en log, un écart de 0,22 correspond à environ +25 %
    residus = y - predictions
    ecart_type = float(np.std(residus)) or 1.0

    lignes = []
    for i, nom in enumerate(noms_agences):
        realise = float(np.expm1(y[i]))
        attendu = float(np.expm1(predictions[i]))
        ecart_relatif = (np.expm1(residus[i]) - 1) * 100 if residus[i] else 0.0
        lignes.append({
            'agence': nom,
            'realise': int(round(realise)),
            'attendu': int(round(max(attendu, 0))),
            'ecart_pct': round(float(ecart_relatif), 1),
            'residu_std': round(float(residus[i] / ecart_type), 2),
        })

    lignes.sort(key=lambda l: -l['ecart_pct'])

    # Poids des variables explicatives (modèle linéaire)
    ridge.fit(X, y)
    poids = ridge.coef_
    ordre = np.argsort(-np.abs(poids))[:6]
    facteurs = [
        {
            'nom': noms_features[colonnes_explicatives[j]],
            'poids': round(float(poids[j]), 3),
            'sens': 'positif' if poids[j] > 0 else 'négatif',
        }
        for j in ordre
    ]

    sur = [l for l in lignes if l['residu_std'] >= 1.0]
    sous = [l for l in lignes if l['residu_std'] <= -1.0]

    return {
        'cible': cible,
        'cible_libelle': dict(CIBLES).get(cible, cible),
        'modele_retenu': meilleur['nom'],
        'r2': meilleur['r2'],
        'comparaison': [
            {'nom': m['nom'], 'r2': m['r2'], 'mae': m['mae']}
            for m in modeles.values()
        ],
        'lignes': lignes,
        'facteurs': facteurs,
        'sur_performance': sur,
        'sous_performance': sous,
        'n_agences': len(noms_agences),
    }


# ------------------------------------------------------------------
# 2. Robustesse du clustering
# ------------------------------------------------------------------

def _partition(nom_algo, donnees, k):
    """Applique un algorithme de clustering et renvoie les étiquettes."""
    if nom_algo == 'kmeans':
        return KMeans(n_clusters=k, n_init=25, random_state=0).fit_predict(donnees)
    if nom_algo == 'gmm':
        return GaussianMixture(
            n_components=k, covariance_type='diag',
            n_init=10, random_state=0
        ).fit_predict(donnees)
    if nom_algo == 'hierarchique':
        return AgglomerativeClustering(n_clusters=k, linkage='ward').fit_predict(donnees)
    if nom_algo == 'dbscan':
        return DBSCAN(eps=2.5, min_samples=3).fit_predict(donnees)
    raise ValueError("Algorithme inconnu.")


def _qualite(donnees, etiquettes):
    """Trois indices internes de qualité d'une partition."""
    valides = etiquettes != -1
    if valides.sum() < 3 or len(set(etiquettes[valides])) < 2:
        return None
    d, e = donnees[valides], etiquettes[valides]
    return {
        'silhouette': round(float(silhouette_score(d, e)), 3),
        'calinski': round(float(calinski_harabasz_score(d, e)), 1),
        'davies': round(float(davies_bouldin_score(d, e)), 3),
        'n_groupes': int(len(set(e))),
    }


def evaluer_robustesse(annee, k=None, n_tirages=60):
    """
    Compare plusieurs familles d'algorithmes sur les mêmes données,
    puis mesure la stabilité du découpage par rééchantillonnage bootstrap.

    La stabilité est quantifiée par l'indice de Rand ajusté entre la
    partition de référence et celles obtenues sur des échantillons tirés
    avec remise. Un indice proche de 1 signale un découpage reproductible.
    """
    noms_agences, noms_features, matrice = construire_matrice(annee)
    if len(noms_agences) < 8:
        return None

    _, donnees = preparer_donnees(matrice)
    n = donnees.shape[0]

    # Nombre de groupes de référence : meilleure silhouette sur K-Means
    if k is None:
        meilleur_k, meilleur_score = 2, -1.0
        for essai in range(2, min(7, n - 1)):
            etiquettes = KMeans(n_clusters=essai, n_init=25, random_state=0).fit_predict(donnees)
            score = silhouette_score(donnees, etiquettes)
            if score > meilleur_score:
                meilleur_k, meilleur_score = essai, score
        k = meilleur_k

    # --- Comparaison des algorithmes ---
    algorithmes = [
        ('kmeans', 'K-Means'),
        ('gmm', 'Mélange gaussien'),
        ('hierarchique', 'Classification hiérarchique'),
        ('dbscan', 'DBSCAN (densité)'),
    ]

    comparaison = []
    for code, libelle in algorithmes:
        try:
            etiquettes = _partition(code, donnees, k)
        except Exception:
            continue
        indices = _qualite(donnees, etiquettes)
        if indices is None:
            comparaison.append({
                'nom': libelle, 'n_groupes': 0, 'silhouette': None,
                'calinski': None, 'davies': None, 'echec': True,
            })
            continue
        comparaison.append({
            'nom': libelle,
            'n_groupes': indices['n_groupes'],
            'silhouette': indices['silhouette'],
            'calinski': indices['calinski'],
            'davies': indices['davies'],
            'echec': False,
        })

    # --- Stabilité par bootstrap ---
    reference = KMeans(n_clusters=k, n_init=25, random_state=0).fit(donnees)
    etiquettes_ref = reference.predict(donnees)

    scores_ari = []
    partitions = []
    generateur = np.random.default_rng(0)

    for _ in range(n_tirages):
        indices = generateur.choice(n, size=n, replace=True)
        if len(np.unique(indices)) < k + 1:
            continue
        try:
            modele = KMeans(n_clusters=k, n_init=10, random_state=None).fit(donnees[indices])
            etiquettes_test = modele.predict(donnees)
            scores_ari.append(adjusted_rand_score(etiquettes_ref, etiquettes_test))
            partitions.append(etiquettes_test)
        except Exception:
            continue

    if scores_ari:
        ari_moyen = float(np.mean(scores_ari))
        ari_ecart = float(np.std(scores_ari))
    else:
        ari_moyen, ari_ecart = 0.0, 0.0

    if ari_moyen >= 0.75:
        verdict = 'stable'
    elif ari_moyen >= 0.55:
        verdict = 'moyennement stable'
    elif ari_moyen >= 0.35:
        verdict = 'peu stable'
    else:
        verdict = 'instable'

    # --- Stabilité individuelle des agences ---
    cooccurrence = np.zeros((n, n))
    for etiquettes in partitions:
        for i in range(n):
            cooccurrence[i] += (etiquettes == etiquettes[i]).astype(float)

    if partitions:
        cooccurrence /= len(partitions)

    fidelite = []
    for i, nom in enumerate(noms_agences):
        meme_groupe = etiquettes_ref == etiquettes_ref[i]
        meme_groupe[i] = False
        taux = float(cooccurrence[i][meme_groupe].mean()) if meme_groupe.any() else 1.0
        fidelite.append({
            'agence': nom,
            'groupe': int(etiquettes_ref[i]) + 1,
            'taux': round(taux * 100, 1),
        })
    fidelite.sort(key=lambda f: f['taux'])

    return {
        'k': k,
        'comparaison': comparaison,
        'ari_moyen': round(ari_moyen, 3),
        'ari_ecart': round(ari_ecart, 3),
        'verdict': verdict,
        'n_tirages': len(partitions),
        'fidelite': fidelite,
        'moins_stables': fidelite[:5],
    }


# ------------------------------------------------------------------
# 3. Classement multicritère (TOPSIS)
# ------------------------------------------------------------------

CRITERES_SCORE = [
    ('comptes', 'Comptes clients'),
    ('credits', 'Crédits'),
    ('cartes', 'Cartes'),
    ('placements', 'Placements'),
    ('tpe', 'TPE'),
]


def calculer_scores(annee, poids=None):
    """
    Classement des agences par la méthode TOPSIS.

    Principe : chaque agence est comparée à une agence idéale (le meilleur
    niveau atteint sur chaque critère) et à une agence anti-idéale. Le score
    final, compris entre 0 et 1, mesure la proximité relative à l'idéal.

    poids : dictionnaire {critère: valeur}. Les valeurs sont normalisées
    pour sommer à 1. Par défaut, tous les critères pèsent également.
    """
    noms_agences, noms_features, matrice = construire_matrice(annee)
    if len(noms_agences) < 3:
        return None

    groupes = _domaines(noms_features)

    criteres, libelles = [], []
    colonnes = []
    for code, libelle in CRITERES_SCORE:
        indices = groupes.get(code, [])
        if indices:
            colonnes.append(matrice[:, indices].sum(axis=1))
            criteres.append(code)
            libelles.append(libelle)

    if len(criteres) < 2:
        return None

    donnees = np.column_stack(colonnes).astype(float)

    # Pondérations
    if poids:
        vecteur = np.array([float(poids.get(c, 1.0)) for c in criteres])
        vecteur = np.clip(vecteur, 0, None)
        if vecteur.sum() == 0:
            vecteur = np.ones(len(criteres))
    else:
        vecteur = np.ones(len(criteres))
    vecteur = vecteur / vecteur.sum()

    # Normalisation vectorielle puis pondération
    normes = np.sqrt((donnees ** 2).sum(axis=0))
    normes[normes == 0] = 1.0
    normalisee = donnees / normes
    ponderee = normalisee * vecteur

    # Solutions idéale et anti-idéale (tous les critères sont à maximiser)
    ideal = ponderee.max(axis=0)
    anti_ideal = ponderee.min(axis=0)

    distance_ideal = np.sqrt(((ponderee - ideal) ** 2).sum(axis=1))
    distance_anti = np.sqrt(((ponderee - anti_ideal) ** 2).sum(axis=1))

    denominateur = distance_ideal + distance_anti
    denominateur[denominateur == 0] = 1.0
    scores = distance_anti / denominateur

    ordre = np.argsort(-scores)

    classement = []
    for rang, i in enumerate(ordre, start=1):
        detail = []
        for j, libelle in enumerate(libelles):
            maximum = donnees[:, j].max() or 1.0
            detail.append({
                'critere': libelle,
                'valeur': int(donnees[i, j]),
                'part_max': round(float(donnees[i, j] / maximum) * 100, 1),
            })
        classement.append({
            'rang': rang,
            'agence': noms_agences[i],
            'score': round(float(scores[i]) * 100, 1),
            'detail': detail,
        })

    return {
        'criteres': libelles,
        'poids': [round(float(p) * 100, 1) for p in vecteur],
        'classement': classement,
        'n_agences': len(noms_agences),
    }


# ------------------------------------------------------------------
# Explications en langage courant
# ------------------------------------------------------------------

def expliquer_performance(resultat):
    if not resultat:
        return []

    points = []
    r2 = resultat['r2']
    cible = resultat['cible_libelle'].lower()

    if r2 < 0:
        points.append(
            f"Le niveau de {cible} d'une agence ne se déduit pas de son activité "
            "sur les autres produits : les écarts ci-dessous sont donc à considérer "
            "comme des repères, pas comme un jugement de performance."
        )
        points.append(
            "Ce constat a du sens : la politique de crédit d'une agence dépend "
            "surtout du tissu économique local, que ces données ne décrivent pas."
        )
    elif r2 < 0.3:
        points.append(
            f"Le niveau de {cible} attendu n'est estimé que grossièrement : "
            "l'activité sur les autres produits n'explique qu'une petite partie du résultat."
        )
    elif r2 < 0.6:
        points.append(
            f"Le niveau de {cible} attendu est estimé de façon correcte "
            "à partir de l'activité de l'agence sur les autres produits."
        )
    else:
        points.append(
            f"Le niveau de {cible} attendu est bien estimé à partir de "
            "l'activité de l'agence sur les autres produits."
        )

    n_sur = len(resultat['sur_performance'])
    n_sous = len(resultat['sous_performance'])

    if n_sur:
        noms = ", ".join(l['agence'] for l in resultat['sur_performance'][:3])
        points.append(f"{n_sur} agences dépassent nettement leur niveau attendu, dont {noms}.")
    if n_sous:
        noms = ", ".join(l['agence'] for l in resultat['sous_performance'][:3])
        points.append(f"{n_sous} agences restent nettement en dessous, dont {noms}.")
    if not n_sur and not n_sous:
        points.append("Aucune agence ne s'écarte fortement de son niveau attendu.")

    points.append(
        "Lecture de l'écart : +30 % signifie que l'agence réalise un tiers de plus "
        "que le niveau estimé pour elle."
    )
    return points


def expliquer_robustesse(resultat):
    if not resultat:
        return []

    points = []
    verdict = resultat['verdict']
    ari = resultat['ari_moyen']

    if verdict == 'stable':
        points.append(
            "Le découpage en familles est fiable : il se reproduit presque à l'identique "
            "quand on refait le calcul sur d'autres échantillons."
        )
    elif verdict == 'moyennement stable':
        points.append(
            "Le découpage est globalement fiable, avec quelques agences qui changent "
            "de famille selon le calcul."
        )
    elif verdict == 'peu stable':
        points.append(
            "Le découpage varie sensiblement d'un calcul à l'autre : à utiliser comme "
            "une indication, pas comme une classification définitive."
        )
    else:
        points.append(
            "Le découpage n'est pas reproductible : les agences changent souvent de famille. "
            "Il vaut mieux raisonner en tendances qu'en groupes."
        )

    points.append(f"Mesure de reproductibilité : {ari} sur 1 (test répété {resultat['n_tirages']} fois).")

    meilleurs = [c for c in resultat['comparaison'] if not c['echec'] and c['silhouette'] is not None]
    if meilleurs:
        gagnant = max(meilleurs, key=lambda c: c['silhouette'])
        points.append(f"Parmi les méthodes testées, {gagnant['nom']} donne le découpage le plus net.")

    if resultat['moins_stables']:
        noms = ", ".join(f['agence'] for f in resultat['moins_stables'][:3])
        points.append(f"Agences les plus difficiles à classer : {noms}.")

    return points


def expliquer_scores(resultat):
    if not resultat:
        return []

    classement = resultat['classement']
    points = [
        "Le score compare chaque agence à une agence idéale, "
        "qui atteindrait le meilleur niveau sur tous les critères.",
        "Un score de 100 signifierait qu'une agence est la meilleure partout ; "
        "0 qu'elle est la dernière partout.",
    ]

    if len(classement) >= 3:
        tete = ", ".join(c['agence'] for c in classement[:3])
        points.append(f"En tête du classement : {tete}.")
        queue = ", ".join(c['agence'] for c in classement[-3:])
        points.append(f"En fin de classement : {queue}.")

    points.append("Les pondérations sont modifiables : ajustez-les selon les priorités du moment.")
    return points