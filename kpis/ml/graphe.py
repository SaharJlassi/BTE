"""
Détection de communautés d'agences par optimisation de modularité.

Complément au K-Means : celui-ci suppose des groupes de forme sphérique
et de tailles voisines, hypothèses souvent fausses sur un réseau
d'agences. L'approche par graphe n'impose aucune forme.

Démarche :

  1. Construction d'un graphe de similarité : chaque agence est reliée à
     ses k plus proches voisines, avec un poids décroissant avec la
     distance (noyau gaussien de largeur adaptative).
  2. Détection de communautés par l'algorithme de Louvain (Blondel et al.,
     2008) : optimisation gloutonne de la modularité, suivie d'un
     repliement du graphe, répété jusqu'à convergence.
  3. Évaluation par la modularité obtenue, comparée à celle de graphes
     aléatoires de même degré, afin de vérifier que la structure trouvée
     n'est pas un simple artefact.

La modularité mesure l'excès de liens internes aux communautés par
rapport à ce qu'un graphe aléatoire produirait. Une valeur supérieure
à 0,3 indique une structure de communautés marquée.
"""

import numpy as np


# ------------------------------------------------------------------
# Construction du graphe de similarité
# ------------------------------------------------------------------

def construire_graphe(donnees, k_voisins=None):
    """
    Graphe des k plus proches voisins, pondéré par un noyau gaussien.

    La largeur du noyau est adaptative : elle vaut, pour chaque agence,
    la distance à sa k-ième voisine. Cela évite qu'une zone dense et une
    zone clairsemée soient traitées avec la même échelle.

    Renvoie une matrice d'adjacence symétrique à diagonale nulle.
    """
    n = donnees.shape[0]
    if k_voisins is None:
        # Règle usuelle pour les graphes de similarité : k ~ log(n),
        # borné pour rester exploitable sur un petit effectif.
        k_voisins = int(np.clip(round(np.log(n) * 2), 3, n - 1))

    # Distances euclidiennes deux à deux
    diff = donnees[:, None, :] - donnees[None, :, :]
    distances = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(distances, np.inf)

    # Largeur locale du noyau : distance à la k-ième voisine
    tries = np.sort(distances, axis=1)
    sigma = tries[:, min(k_voisins - 1, n - 2)]
    sigma[sigma <= 0] = 1e-6

    adjacence = np.zeros((n, n))
    for i in range(n):
        voisins = np.argsort(distances[i])[:k_voisins]
        for j in voisins:
            poids = np.exp(-(distances[i, j] ** 2) / (sigma[i] * sigma[j]))
            # Symétrisation : un lien existe si l'une des deux agences
            # considère l'autre comme voisine.
            adjacence[i, j] = max(adjacence[i, j], poids)
            adjacence[j, i] = max(adjacence[j, i], poids)

    np.fill_diagonal(adjacence, 0.0)
    return adjacence, k_voisins


# ------------------------------------------------------------------
# Modularité
# ------------------------------------------------------------------

def modularite(adjacence, communautes):
    """
    Modularité de Newman-Girvan pour un graphe pondéré.

        Q = (1 / 2m) * somme_ij [ A_ij - k_i k_j / 2m ] * delta(c_i, c_j)

    où m est le poids total des arêtes et k_i le degré pondéré du sommet i.
    Q compare la densité des liens internes aux communautés à celle
    attendue dans un graphe aléatoire de mêmes degrés.
    """
    degres = adjacence.sum(axis=1)
    m2 = degres.sum()
    if m2 <= 0:
        return 0.0

    q = 0.0
    for c in set(communautes):
        masque = np.array([x == c for x in communautes])
        poids_interne = adjacence[np.ix_(masque, masque)].sum()
        degre_total = degres[masque].sum()
        q += poids_interne / m2 - (degre_total / m2) ** 2
    return float(q)


# ------------------------------------------------------------------
# Algorithme de Louvain
# ------------------------------------------------------------------

def _phase_locale(adjacence, degres, m2, communautes):
    """
    Première phase : chaque sommet rejoint la communauté voisine qui
    procure le plus fort gain de modularité, jusqu'à stabilisation.
    """
    n = adjacence.shape[0]
    ameliore = True
    total_ameliorations = 0

    while ameliore:
        ameliore = False
        for i in range(n):
            communaute_actuelle = communautes[i]

            # Poids des liens de i vers chaque communauté
            poids_vers = {}
            for j in range(n):
                if j != i and adjacence[i, j] > 0:
                    c = communautes[j]
                    poids_vers[c] = poids_vers.get(c, 0.0) + adjacence[i, j]

            # Retrait provisoire de i de sa communauté
            communautes[i] = -1
            degre_communaute = {}
            for j in range(n):
                if communautes[j] >= 0:
                    c = communautes[j]
                    degre_communaute[c] = degre_communaute.get(c, 0.0) + degres[j]

            meilleure_communaute = communaute_actuelle
            meilleur_gain = 0.0

            for c, poids in poids_vers.items():
                gain = poids / m2 - (degre_communaute.get(c, 0.0) * degres[i]) / (m2 ** 2 / 2)
                if gain > meilleur_gain + 1e-12:
                    meilleur_gain = gain
                    meilleure_communaute = c

            communautes[i] = meilleure_communaute
            if meilleure_communaute != communaute_actuelle:
                ameliore = True
                total_ameliorations += 1

    return communautes, total_ameliorations


def louvain(adjacence, n_iterations=20):
    """
    Optimisation de la modularité par l'algorithme de Louvain.

    Alterne deux phases : déplacement local des sommets, puis repliement
    du graphe où chaque communauté devient un sommet. Répété jusqu'à ce
    qu'aucun déplacement n'améliore la modularité.
    """
    n = adjacence.shape[0]
    if n < 3:
        return [0] * n, 0.0

    # Correspondance entre sommets d'origine et communautés courantes
    appartenance = list(range(n))
    graphe = adjacence.copy()
    etiquettes = list(range(n))

    for _ in range(n_iterations):
        degres = graphe.sum(axis=1)
        m2 = degres.sum()
        if m2 <= 0:
            break

        communautes = list(range(graphe.shape[0]))
        communautes, changements = _phase_locale(graphe, degres, m2, communautes)

        # Renumérotation compacte
        uniques = sorted(set(communautes))
        renum = {c: i for i, c in enumerate(uniques)}
        communautes = [renum[c] for c in communautes]

        # Report sur les sommets d'origine
        appartenance = [communautes[etiquettes[i]] for i in range(n)]

        if changements == 0 or len(uniques) == graphe.shape[0]:
            break

        # Repliement : chaque communauté devient un sommet
        taille = len(uniques)
        nouveau = np.zeros((taille, taille))
        for i in range(graphe.shape[0]):
            for j in range(graphe.shape[0]):
                if graphe[i, j] > 0:
                    nouveau[communautes[i], communautes[j]] += graphe[i, j]

        graphe = nouveau
        etiquettes = appartenance

    q = modularite(adjacence, appartenance)
    return appartenance, q


# ------------------------------------------------------------------
# Validation par graphes aléatoires
# ------------------------------------------------------------------

def _modularite_aleatoire(adjacence, n_tirages=50, graine=0):
    """
    Modularité obtenue sur des graphes dont les liens sont redistribués
    au hasard, à degrés préservés. Sert de référence : toute partition,
    même sur un graphe sans structure, produit une modularité positive.
    """
    n = adjacence.shape[0]
    gen = np.random.default_rng(graine)
    poids = adjacence[np.triu_indices(n, k=1)]
    poids = poids[poids > 0]

    if len(poids) == 0:
        return 0.0, 0.0

    valeurs = []
    for _ in range(n_tirages):
        alea = np.zeros((n, n))
        indices = np.triu_indices(n, k=1)
        choix = gen.choice(len(indices[0]), size=len(poids), replace=False)
        melange = gen.permutation(poids)
        for pos, p in zip(choix, melange):
            i, j = indices[0][pos], indices[1][pos]
            alea[i, j] = alea[j, i] = p
        _, q = louvain(alea, n_iterations=8)
        valeurs.append(q)

    return float(np.mean(valeurs)), float(np.std(valeurs))


# ------------------------------------------------------------------
# Point d'entrée
# ------------------------------------------------------------------

def analyser_communautes(donnees, noms_agences, noms_features, matrice=None):
    """
    Construit le graphe de similarité, détecte les communautés et
    prépare les données d'affichage.
    """
    n = donnees.shape[0]
    if n < 6:
        return None

    adjacence, k_voisins = construire_graphe(donnees)
    communautes, q = louvain(adjacence)

    q_alea, ecart_alea = _modularite_aleatoire(adjacence)
    ecart_type_significatif = (q - q_alea) / ecart_alea if ecart_alea > 1e-9 else 0.0

    # --- Caractérisation de chaque communauté ---
    noms_etendus = list(noms_features) + ["Taille (log)"]
    groupes = []

    for c in sorted(set(communautes)):
        masque = np.array([x == c for x in communautes])
        profil = donnees[masque].mean(axis=0)
        ordre = np.argsort(-np.abs(profil))[:4]
        groupes.append({
            'numero': int(c) + 1,
            'effectif': int(masque.sum()),
            'agences': [noms_agences[i] for i in range(n) if masque[i]],
            'traits': [
                {
                    'nom': noms_etendus[j],
                    'ecart': round(float(profil[j]), 2),
                    'sens': 'fort' if profil[j] > 0 else 'faible',
                }
                for j in ordre
            ],
        })

    # --- Position des sommets pour le tracé ---
    # Disposition par forces : les agences reliées s'attirent, les autres
    # se repoussent, ce qui rapproche visuellement les communautés.
    positions = _disposer(adjacence, communautes)

    noeuds = []
    for i, nom in enumerate(noms_agences):
        degre = float(adjacence[i].sum())
        noeuds.append({
            'agence': nom,
            'communaute': int(communautes[i]) + 1,
            'x': round(float(positions[i, 0]), 3),
            'y': round(float(positions[i, 1]), 3),
            'degre': round(degre, 2),
            'volume': int(matrice[i].sum()) if matrice is not None else 0,
        })

    # --- Arêtes, limitées aux liens les plus forts pour rester lisible ---
    aretes = []
    seuil = float(np.percentile(adjacence[adjacence > 0], 40)) if (adjacence > 0).any() else 0
    for i in range(n):
        for j in range(i + 1, n):
            if adjacence[i, j] > seuil:
                aretes.append({
                    'source': i,
                    'cible': j,
                    'poids': round(float(adjacence[i, j]), 3),
                    'interne': communautes[i] == communautes[j],
                })

    if q >= 0.4:
        verdict = 'marquée'
    elif q >= 0.25:
        verdict = 'modérée'
    else:
        verdict = 'faible'

    return {
        'modularite': round(q, 3),
        'modularite_aleatoire': round(q_alea, 3),
        'ecart_significatif': round(ecart_type_significatif, 1),
        'verdict': verdict,
        'n_communautes': len(groupes),
        'n_agences': n,
        'k_voisins': k_voisins,
        'groupes': groupes,
        'noeuds': noeuds,
        'aretes': aretes,
        'n_aretes': len(aretes),
    }


def _disposer(adjacence, communautes, n_iterations=300, graine=1):
    """
    Disposition du graphe par modèle de forces (Fruchterman-Reingold
    simplifié) : les sommets reliés s'attirent, tous se repoussent.
    """
    n = adjacence.shape[0]
    gen = np.random.default_rng(graine)
    positions = gen.normal(0, 1, (n, 2))

    k = np.sqrt(1.0 / n)
    temperature = 0.5

    for pas in range(n_iterations):
        deplacement = np.zeros((n, 2))

        diff = positions[:, None, :] - positions[None, :, :]
        distances = np.sqrt((diff ** 2).sum(axis=2)) + 1e-9

        # Répulsion générale
        force = (k ** 2) / distances
        np.fill_diagonal(force, 0)
        deplacement += (diff / distances[:, :, None] * force[:, :, None]).sum(axis=1)

        # Attraction le long des arêtes
        attraction = adjacence * distances / k
        deplacement -= (diff / distances[:, :, None] * attraction[:, :, None]).sum(axis=1)

        norme = np.sqrt((deplacement ** 2).sum(axis=1))[:, None] + 1e-9
        positions += deplacement / norme * np.minimum(norme, temperature)
        temperature *= 0.97

    # Normalisation dans un carré unité centré
    positions -= positions.mean(axis=0)
    echelle = np.abs(positions).max() or 1.0
    return positions / echelle


def expliquer_communautes(resultat):
    """Commentaires en langage courant."""
    if not resultat:
        return []

    points = [
        "Les agences sont reliées entre elles quand leur activité se ressemble ; "
        "on cherche ensuite les groupes les plus densément connectés.",
    ]

    n_com = resultat['n_communautes']
    verdict = resultat['verdict']

    if verdict == 'marquée':
        points.append(
            f"Le réseau se sépare nettement en {n_com} communautés : les agences "
            "d'un même groupe se ressemblent beaucoup plus entre elles qu'avec le reste."
        )
    elif verdict == 'modérée':
        points.append(
            f"Le réseau se sépare en {n_com} communautés aux contours visibles, "
            "avec quelques agences à la frontière entre deux groupes."
        )
    else:
        points.append(
            f"Le découpage en {n_com} communautés reste peu marqué : les agences "
            "forment un ensemble assez homogène."
        )

    if resultat['ecart_significatif'] >= 2:
        points.append(
            "Cette structure ne doit rien au hasard : elle est nettement plus "
            "marquée que sur des réseaux dont les liens seraient distribués au hasard."
        )
    else:
        points.append(
            "Attention : un découpage comparable s'obtiendrait sur un réseau "
            "sans structure réelle. Ce résultat est donc à prendre avec prudence."
        )

    plus_grand = max(resultat['groupes'], key=lambda g: g['effectif'])
    points.append(
        f"Le groupe le plus important rassemble {plus_grand['effectif']} agences "
        f"sur {resultat['n_agences']}."
    )

    return points