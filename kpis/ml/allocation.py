"""
Allocation d'un effort commercial sous contrainte de budget.

Question traitée : disposant d'un nombre limité d'interventions
(visites, formations, appuis), sur quelles agences les concentrer
pour maximiser le gain attendu sur le réseau ?

Cibler les agences les plus en retard n'est pas optimal : une agence
très en retard mais de petite taille rapporte peu même redressée. Le
critère retenu est le gain absolu récupérable, soit l'écart au niveau
attendu multiplié par le volume concerné.

Le gain est pondéré par deux facteurs :

  - la fiabilité de l'écart, issue de l'ajustement bayésien : une agence
    dont l'écart n'est pas établi statistiquement voit son gain escompté
    réduit d'autant ;
  - un taux de redressement, part de l'écart qu'une intervention permet
    raisonnablement de combler.

La sélection est un problème de sac à dos, résolu exactement par
programmation dynamique. Le nombre d'agences étant modeste, l'optimum
est atteint sans recourir à une heuristique.
"""

import numpy as np


# Part de l'écart qu'une intervention permet de combler, par hypothèse.
TAUX_REDRESSEMENT = 0.40

# Coût relatif d'une intervention selon la taille de l'agence.
# Une grande agence demande davantage de moyens qu'une petite.
COUTS = [
    (0.33, 1, 'légère'),    # tiers inférieur des volumes
    (0.67, 2, 'moyenne'),   # tiers médian
    (1.00, 3, 'lourde'),    # tiers supérieur
]


def _cout_intervention(rang_relatif):
    """Coût en unités de budget, selon la position de l'agence en volume."""
    for seuil, cout, libelle in COUTS:
        if rang_relatif <= seuil:
            return cout, libelle
    return COUTS[-1][1], COUTS[-1][2]


def _sac_a_dos(gains, couts, budget):
    """
    Sélection optimale sous contrainte de budget, par programmation
    dynamique. Renvoie la liste des indices retenus.

    Complexité O(n × budget), négligeable ici, et garantit l'optimum
    exact contrairement à une sélection gloutonne par ratio.
    """
    n = len(gains)
    if n == 0 or budget <= 0:
        return []

    # table[i][b] : gain maximal atteignable avec les i premières
    # agences et un budget b.
    table = np.zeros((n + 1, budget + 1))

    for i in range(1, n + 1):
        cout = int(couts[i - 1])
        gain = float(gains[i - 1])
        for b in range(budget + 1):
            sans = table[i - 1][b]
            avec = table[i - 1][b - cout] + gain if cout <= b else -np.inf
            table[i][b] = max(sans, avec)

    # Remontée du chemin optimal
    retenus = []
    b = budget
    for i in range(n, 0, -1):
        if table[i][b] != table[i - 1][b]:
            retenus.append(i - 1)
            b -= int(couts[i - 1])

    return sorted(retenus)


def optimiser(performance, budget=6, taux_redressement=TAUX_REDRESSEMENT):
    """
    Détermine les agences à cibler en priorité.

    performance : résultat de analyser_performance enrichi par
    ajuster_performance (les écarts corrigés sont nécessaires).
    budget : nombre d'unités d'intervention disponibles.
    """
    if not performance or performance.get('insuffisant'):
        return None

    lignes = performance.get('lignes') or []
    if len(lignes) < 3:
        return None

    # --- Classement par volume, pour établir le coût d'intervention ---
    volumes = np.array([l['realise'] for l in lignes], dtype=float)
    rangs = np.argsort(np.argsort(volumes)) / max(len(volumes) - 1, 1)

    candidats = []

    for i, ligne in enumerate(lignes):
        # Seules les agences en retard sont candidates
        ecart = ligne.get('ecart_ajuste', ligne.get('ecart_pct', 0))
        if ecart >= 0:
            continue

        attendu = max(ligne['attendu'], 0)
        realise = max(ligne['realise'], 0)
        manque = max(attendu - realise, 0)
        if manque <= 0:
            continue

        # Fiabilité : probabilité que l'agence soit réellement en dessous.
        # Une agence dont l'écart n'est pas établi voit son gain minoré.
        probabilite = ligne.get('probabilite', 50.0)
        fiabilite = max(0.0, (100.0 - probabilite) / 100.0)

        gain = manque * taux_redressement * fiabilite
        cout, intensite = _cout_intervention(float(rangs[i]))

        candidats.append({
            'agence': ligne['agence'],
            'realise': realise,
            'attendu': attendu,
            'manque': int(round(manque)),
            'ecart': ecart,
            'fiabilite': round(fiabilite * 100, 1),
            'gain': round(float(gain), 1),
            'cout': cout,
            'intensite': intensite,
            'rendement': round(float(gain / cout), 1),
        })

    if not candidats:
        return {
            'budget': budget,
            'aucun_candidat': True,
            'candidats': [],
            'retenus': [],
            'gain_total': 0,
            'cout_engage': 0,
            'taux_redressement': taux_redressement,
        }

    gains = [c['gain'] for c in candidats]
    couts = [c['cout'] for c in candidats]

    indices = _sac_a_dos(gains, couts, budget)
    for i in indices:
        candidats[i]['retenu'] = True
    for c in candidats:
        c.setdefault('retenu', False)

    retenus = [candidats[i] for i in indices]
    retenus.sort(key=lambda c: -c['gain'])

    candidats.sort(key=lambda c: -c['rendement'])

    gain_total = sum(c['gain'] for c in retenus)
    cout_engage = sum(c['cout'] for c in retenus)

    # --- Comparaison à une stratégie naïve ---
    # Cibler les plus fortes baisses en pourcentage, sans tenir compte
    # ni du volume ni du coût.
    naif = sorted(candidats, key=lambda c: c['ecart'])
    selection_naive, budget_restant = [], budget
    for c in naif:
        if c['cout'] <= budget_restant:
            selection_naive.append(c)
            budget_restant -= c['cout']
    gain_naif = sum(c['gain'] for c in selection_naive)

    return {
        'budget': budget,
        'aucun_candidat': False,
        'candidats': candidats,
        'retenus': retenus,
        'gain_total': int(round(gain_total)),
        'cout_engage': cout_engage,
        'n_retenus': len(retenus),
        'n_candidats': len(candidats),
        'gain_naif': int(round(gain_naif)),
        'n_naif': len(selection_naive),
        'progression': (
            round((gain_total / gain_naif - 1) * 100, 1)
            if gain_naif > 0 else 0.0
        ),
        'taux_redressement': taux_redressement,
        'cible_libelle': performance.get('cible_libelle', ''),
    }


def expliquer_allocation(resultat):
    """Commentaires en langage courant."""
    if not resultat:
        return []

    if resultat.get('aucun_candidat'):
        return [
            "Aucune agence n'est en retard sur son niveau attendu : "
            "il n'y a pas de rattrapage à organiser sur ce produit.",
        ]

    points = [
        "Cibler les agences les plus en retard n'est pas la meilleure stratégie : "
        "une petite agence très en retard rapporte moins qu'une grande agence "
        "légèrement en retard.",
        "Le classement retenu tient compte du nombre de dossiers réellement "
        "récupérables, et non du seul pourcentage d'écart.",
    ]

    cible = resultat['cible_libelle'].lower()
    points.append(
        f"Avec {resultat['budget']} unités d'intervention, "
        f"{resultat['n_retenus']} agences sont à cibler, pour un gain estimé de "
        f"{resultat['gain_total']} dossiers de {cible}."
    )

    if resultat['progression'] > 5:
        points.append(
            f"Cette sélection rapporte {resultat['progression']} % de plus que si "
            "l'on ciblait simplement les agences aux plus fortes baisses."
        )

    points.append(
        f"Hypothèse retenue : une intervention permet de combler "
        f"{int(resultat['taux_redressement'] * 100)} % de l'écart constaté."
    )
    return points