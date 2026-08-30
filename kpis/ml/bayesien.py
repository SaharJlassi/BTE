"""
Ajustement bayésien hiérarchique des écarts de performance.

Problème traité : sur une petite agence, un écart de +268 % peut n'être
que du bruit d'échantillonnage, alors que +30 % sur une grosse agence
est un signal solide. Comparer ces deux nombres directement induit en
erreur.

Modèle :
    r_i  ~ N(theta_i, s_i²)     écart observé de l'agence i
    theta_i ~ N(0, tau²)        écart véritable, tiré vers la moyenne

où s_i² est la variance d'échantillonnage de l'agence (approximée par
1/volume, dérivée de la variance d'un comptage sur échelle log) et tau²
la dispersion réelle entre agences, estimée sur les données par la
méthode des moments (estimateur de DerSimonian-Laird).

La distribution a posteriori de theta_i a une forme fermée :

    theta_i | r_i  ~  N( B_i · r_i ,  B_i · s_i² )
    avec  B_i = tau² / (tau² + s_i²)

B_i est le coefficient de rappel : proche de 1 pour une agence à fort
volume (l'écart est conservé), proche de 0 pour une petite agence
(l'écart est ramené vers zéro).

Ce résultat est celui de Stein (1955) : sur un ensemble d'estimations
simultanées, cet estimateur contracté domine l'estimateur brut au sens
de l'erreur quadratique.

Le verdict affiché ('au-dessus', 'en-dessous', 'conforme') provient de
l'intervalle de prédiction conforme calculé en amont, plus direct à
interpréter. L'ajustement bayésien fournit ici l'ampleur corrigée de
l'écart et sa probabilité, non la décision.
"""

import math

import numpy as np


def _phi(x):
    """Fonction de répartition de la loi normale centrée réduite."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _estimer_tau2(residus, variances):
    """
    Estime la dispersion réelle entre agences par la méthode des moments.

    Sépare la variabilité observée en deux parts : celle due au hasard
    d'échantillonnage, et celle due à de vraies différences entre agences.
    Seule la seconde est retenue.
    """
    poids = 1.0 / np.maximum(variances, 1e-9)
    somme_poids = poids.sum()

    if somme_poids <= 0:
        return 0.0

    moyenne_ponderee = float((poids * residus).sum() / somme_poids)

    # Statistique Q : variabilité totale observée
    Q = float((poids * (residus - moyenne_ponderee) ** 2).sum())
    k = len(residus)

    denominateur = somme_poids - float((poids ** 2).sum()) / somme_poids
    if denominateur <= 0:
        return 0.0

    tau2 = (Q - (k - 1)) / denominateur
    return max(0.0, float(tau2))


def ajuster_performance(resultat, niveau=0.90):
    """
    Enrichit le résultat de analyser_performance avec un écart ajusté,
    un intervalle de crédibilité et une probabilité de sur-performance.

    Chaque ligne reçoit :
      ecart_ajuste   : écart en %, après rappel vers la moyenne
      borne_basse    : borne inférieure de l'intervalle, en %
      borne_haute    : borne supérieure de l'intervalle, en %
      probabilite    : probabilité que l'agence soit réellement au-dessus
                       de son niveau attendu, en %
      fiabilite      : part de l'écart brut conservée après ajustement
      verdict        : reprise du verdict conforme, ou verdict bayésien
                       à défaut
    """
    if not resultat or resultat.get('insuffisant') or not resultat.get('lignes'):
        return resultat

    lignes = resultat['lignes']

    # Écart sur échelle logarithmique : additif, symétrique, adapté au modèle
    residus = []
    variances = []

    for ligne in lignes:
        realise = max(ligne['realise'], 0)
        attendu = max(ligne['attendu'], 1)
        residu = math.log((realise + 1) / (attendu + 1))
        residus.append(residu)

        # Variance d'un comptage sur échelle log (méthode delta) : ~1/n
        variances.append(1.0 / max(realise, 1))

    residus = np.array(residus, dtype=float)
    variances = np.array(variances, dtype=float)

    tau2 = _estimer_tau2(residus, variances)

    # Si aucune dispersion réelle n'est détectée, tous les écarts
    # sont attribuables au hasard : ils sont ramenés à zéro.
    if tau2 <= 0:
        rappel = np.zeros_like(residus)
    else:
        rappel = tau2 / (tau2 + variances)

    theta = rappel * residus                     # écart ajusté (échelle log)
    variance_post = rappel * variances           # variance a posteriori
    ecart_type_post = np.sqrt(np.maximum(variance_post, 1e-12))

    z = 1.645 if abs(niveau - 0.90) < 1e-6 else 1.96

    n_au_dessus = 0
    n_en_dessous = 0

    for i, ligne in enumerate(lignes):
        ajuste_pct = (math.exp(theta[i]) - 1.0) * 100
        basse_pct = (math.exp(theta[i] - z * ecart_type_post[i]) - 1.0) * 100
        haute_pct = (math.exp(theta[i] + z * ecart_type_post[i]) - 1.0) * 100

        probabilite = _phi(theta[i] / ecart_type_post[i]) if ecart_type_post[i] > 0 else 0.5

        # Verdict bayésien, retenu seulement si le verdict conforme
        # n'a pas été calculé en amont.
        if basse_pct > 0:
            verdict_bayes = 'au-dessus'
        elif haute_pct < 0:
            verdict_bayes = 'en-dessous'
        else:
            verdict_bayes = 'conforme'

        verdict = ligne.get('verdict_conforme', verdict_bayes)
        if verdict == 'au-dessus':
            n_au_dessus += 1
        elif verdict == 'en-dessous':
            n_en_dessous += 1

        ligne['ecart_ajuste'] = round(float(np.clip(ajuste_pct, -99, 999)), 1)
        ligne['borne_basse'] = round(float(np.clip(basse_pct, -99, 999)), 1)
        ligne['borne_haute'] = round(float(np.clip(haute_pct, -99, 999)), 1)
        ligne['probabilite'] = round(float(probabilite) * 100, 1)
        ligne['fiabilite'] = round(float(rappel[i]) * 100, 1)
        ligne['verdict_bayesien'] = verdict_bayes
        ligne['verdict'] = verdict

    lignes.sort(key=lambda l: -l['ecart_ajuste'])

    resultat['bayesien'] = {
        'tau': round(float(math.sqrt(tau2)), 3),
        'rappel_moyen': round(float(np.mean(rappel)) * 100, 1),
        'niveau': int(niveau * 100),
        'n_au_dessus': n_au_dessus,
        'n_en_dessous': n_en_dessous,
        'n_conforme': len(lignes) - n_au_dessus - n_en_dessous,
    }

    resultat['au_dessus'] = [l for l in lignes if l['verdict'] == 'au-dessus']
    resultat['en_dessous'] = [l for l in lignes if l['verdict'] == 'en-dessous']

    return resultat


def expliquer_bayesien(resultat):
    """Commentaires en langage courant sur l'ajustement."""
    if not resultat or resultat.get('insuffisant') or 'bayesien' not in resultat:
        return []

    infos = resultat['bayesien']
    points = [
        "Un écart mesuré sur une petite agence est moins fiable que le même "
        "écart sur une grande : il peut n'être dû qu'au hasard.",
        "Les écarts affichés ici sont corrigés en conséquence : plus le volume "
        "d'une agence est faible, plus son écart est ramené vers la moyenne.",
    ]

    if infos['rappel_moyen'] < 40:
        points.append(
            "La correction appliquée est forte : les volumes sont faibles et "
            "les écarts bruts peu fiables."
        )
    elif infos['rappel_moyen'] > 80:
        points.append(
            "La correction appliquée est légère : les volumes sont suffisants "
            "pour que les écarts bruts soient déjà fiables."
        )
    else:
        points.append(
            f"En moyenne, {infos['rappel_moyen']} % de l'écart brut est conservé "
            "après correction."
        )

    return points