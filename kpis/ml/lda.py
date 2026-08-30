"""
Profils d'activité par allocation de Dirichlet latente.

Complément à la factorisation NMF : celle-ci décompose une matrice de
parts sans définir de vraisemblance, ce qui interdit de comparer
objectivement deux valeurs du nombre de profils. Le choix reste alors
à la charge de l'analyste.

La LDA (Blei, Ng, Jordan, 2003) est un modèle génératif : chaque agence
est supposée tirer ses opérations dans un mélange de thématiques
produits, chaque thématique étant elle-même une distribution sur les
produits. Le modèle définit donc une probabilité d'observer les données,
et sa qualité s'évalue par la perplexité mesurée sur des agences non
utilisées pour l'ajustement.

Deux avantages sur la NMF dans ce contexte :

  - le nombre de profils est déterminé par validation croisée et non fixé
    arbitrairement ;
  - le modèle travaille sur les comptages bruts, de sorte qu'une agence
    à fort volume pèse davantage dans l'estimation, ce qui correspond à
    la quantité d'information qu'elle apporte réellement.
"""

import numpy as np

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import KFold


# Nombre de plis de la validation croisée. Trois suffisent sur une
# trentaine d'observations : au-delà, les échantillons d'apprentissage
# se recouvrent trop pour que le coût supplémentaire améliore l'estimation.
N_PLIS = 3

# Itérations pendant la phase de sélection, puis pour l'ajustement final.
# La sélection compare des modèles entre eux et tolère une convergence
# approchée ; le modèle retenu est ensuite ajusté plus finement.
ITER_SELECTION = 40
ITER_FINAL = 200

# En deçà de cet écart relatif entre les perplexités, le balayage est
# jugé plat : le modèle ne tranche pas entre plusieurs valeurs de k.
SEUIL_BALAYAGE_PLAT = 0.05


def _perplexite_validee(matrice, n_profils, n_plis=N_PLIS, graine=0):
    """
    Perplexité moyenne obtenue sur des agences tenues à l'écart
    de l'ajustement.

    La perplexité mesure la surprise du modèle face à des données
    nouvelles : plus elle est faible, mieux le modèle rend compte
    de la structure. L'évaluer sur des agences non vues évite de
    récompenser le sur-ajustement, auquel un nombre élevé de profils
    conduirait mécaniquement.
    """
    n = matrice.shape[0]
    n_plis = min(n_plis, n)
    if n_plis < 2:
        return None

    decoupage = KFold(n_splits=n_plis, shuffle=True, random_state=graine)
    valeurs = []

    for indices_ajustement, indices_test in decoupage.split(matrice):
        apprentissage = matrice[indices_ajustement]
        test = matrice[indices_test]

        if apprentissage.sum() == 0 or test.sum() == 0:
            continue

        modele = LatentDirichletAllocation(
            n_components=n_profils,
            learning_method='batch',
            max_iter=ITER_SELECTION,
            random_state=graine,
            n_jobs=1,
        )
        try:
            modele.fit(apprentissage)
            valeurs.append(float(modele.perplexity(test)))
        except Exception:
            continue

    return float(np.mean(valeurs)) if valeurs else None


def extraire_profils_lda(matrice, noms_agences, noms_features,
                         k_min=2, k_max=5, graine=0):
    """
    Ajuste une LDA dont le nombre de profils est choisi par perplexité,
    puis renvoie la composition de chaque profil et le mélange propre
    à chaque agence.
    """
    comptages = np.round(np.maximum(matrice, 0)).astype(int)

    n_agences, n_produits = comptages.shape
    if n_agences < 6 or n_produits < 3 or comptages.sum() == 0:
        return None

    k_max = min(k_max, n_produits - 1, n_agences - 2)
    if k_max < k_min:
        return None

    # --- Choix du nombre de profils ---
    balayage = []
    meilleur_k, meilleure_perplexite = None, None

    for k in range(k_min, k_max + 1):
        perplexite = _perplexite_validee(comptages, k, graine=graine)
        if perplexite is None:
            continue
        balayage.append({'k': k, 'perplexite': round(perplexite, 1)})
        if meilleure_perplexite is None or perplexite < meilleure_perplexite:
            meilleur_k, meilleure_perplexite = k, perplexite

    if meilleur_k is None:
        return None

    # --- Netteté du choix ---
    # Un balayage plat signale que plusieurs valeurs de k décrivent les
    # données presque aussi bien : le résultat doit alors être présenté
    # comme indicatif plutôt que comme une structure établie.
    valeurs = [b['perplexite'] for b in balayage]
    if len(valeurs) >= 2 and min(valeurs) > 0:
        etendue = (max(valeurs) - min(valeurs)) / min(valeurs)
    else:
        etendue = 0.0
    choix_net = etendue >= SEUIL_BALAYAGE_PLAT

    # --- Ajustement final sur l'ensemble des agences ---
    modele = LatentDirichletAllocation(
        n_components=meilleur_k,
        learning_method='batch',
        max_iter=ITER_FINAL,
        random_state=graine,
        n_jobs=1,
    )
    melanges = modele.fit_transform(comptages)          # agences x profils
    composition = modele.components_                    # profils x produits

    # Normalisation : chaque profil devient une distribution sur les produits
    composition = composition / composition.sum(axis=1, keepdims=True)

    # Chaque agence devient une répartition entre profils
    melanges = melanges / np.maximum(melanges.sum(axis=1, keepdims=True), 1e-12)

    # --- Ordonnancement des profils par importance dans le réseau ---
    # Un profil très minoritaire présenté en premier prêterait à confusion.
    poids_reseau = melanges.mean(axis=0)
    ordre_profils = np.argsort(-poids_reseau)
    composition = composition[ordre_profils]
    melanges = melanges[:, ordre_profils]
    poids_reseau = poids_reseau[ordre_profils]

    profils = []
    for p in range(meilleur_k):
        indices = np.argsort(-composition[p])[:5]
        profils.append({
            'numero': p + 1,
            'part_reseau': round(float(poids_reseau[p]) * 100, 1),
            'produits': [
                {
                    'nom': noms_features[j],
                    'poids': round(float(composition[p][j]) * 100, 1),
                }
                for j in indices
            ],
        })

    # --- Concentration : une agence est-elle mono-profil ou mélangée ? ---
    # Entropie normalisée : 0 pour une agence relevant d'un seul profil,
    # 1 pour une agence répartie également entre tous.
    appartenances = []
    for i, nom in enumerate(noms_agences):
        parts = melanges[i]
        parts_positives = parts[parts > 1e-12]
        entropie = -float((parts_positives * np.log(parts_positives)).sum())
        entropie_max = np.log(meilleur_k) if meilleur_k > 1 else 1.0
        melange = entropie / entropie_max

        appartenances.append({
            'agence': nom,
            'parts': [round(float(v) * 100, 1) for v in parts],
            'dominant': int(np.argmax(parts)) + 1,
            'part_dominante': round(float(parts.max()) * 100, 1),
            'melange': round(melange * 100, 1),
        })

    appartenances.sort(key=lambda a: (a['dominant'], -a['part_dominante']))

    n_mono = sum(1 for a in appartenances if a['part_dominante'] >= 70)
    n_mixte = len(appartenances) - n_mono

    return {
        'k': meilleur_k,
        'perplexite': round(meilleure_perplexite, 1),
        'balayage': balayage,
        'choix_net': choix_net,
        'etendue': round(etendue * 100, 1),
        'n_plis': N_PLIS,
        'profils': profils,
        'appartenances': appartenances,
        'n_mono': n_mono,
        'n_mixte': n_mixte,
        'n_agences': n_agences,
    }


def expliquer_lda(resultat):
    """Commentaires en langage courant."""
    if not resultat:
        return []

    k = resultat['k']
    points = [
        f"Le réseau se décrit par {k} façons types de travailler, "
        "identifiées automatiquement à partir des données.",
        "Le nombre de profils n'est pas choisi à la main : plusieurs valeurs "
        "sont testées, et la retenue est celle qui décrit le mieux des agences "
        "que le modèle n'avait pas vues.",
    ]

    dominant = resultat['profils'][0]
    points.append(
        f"Le profil le plus répandu représente {dominant['part_reseau']} % "
        f"de l'activité du réseau, centré sur {dominant['produits'][0]['nom'].lower()}."
    )

    n_mono = resultat['n_mono']
    n_mixte = resultat['n_mixte']

    if n_mono > n_mixte:
        points.append(
            f"{n_mono} agences relèvent nettement d'un seul profil, "
            f"contre {n_mixte} qui en combinent plusieurs."
        )
    else:
        points.append(
            f"{n_mixte} agences combinent plusieurs profils, contre {n_mono} "
            "qui relèvent nettement d'un seul : le réseau est plutôt mixte."
        )

    if not resultat.get('choix_net', True):
        points.append(
            "Ce découpage reste indicatif : deux, trois ou quatre profils "
            "décrivent les données presque aussi bien. Le plus simple a été retenu."
        )

    return points