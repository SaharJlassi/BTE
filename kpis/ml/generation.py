"""
Génération d'historique calibrée sur les données sectorielles réelles.

Les fichiers produits restent SYNTHÉTIQUES : ils reconstituent un passé
plausible, ils ne constituent pas des observations. Ils servent à tester
l'application et à illustrer les analyses pluriannuelles.

Méthode :
  1. Les taux de croissance macro proviennent des rapports annuels de la
     Banque Centrale de Tunisie (crédits, monétique).
  2. Chaque agence reçoit un taux propre, obtenu en pondérant ces taux
     sectoriels par la composition réelle de son portefeuille.
  3. Une classification non supervisée (K-Means) regroupe les agences par
     profil ; chaque groupe porte une dynamique différenciée.
"""

import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from .analytics import construire_matrice, preparer_donnees


# ------------------------------------------------------------------
# Taux de croissance observés, par segment et par année
# Source : rapports annuels BCT (crédits à l'économie, monétique).
# Les valeurs marquées "hypothèse" ne sont pas publiées au niveau
# de granularité requis et ont été estimées par continuité.
# ------------------------------------------------------------------

TAUX_SECTORIELS = {
    # Crédits aux particuliers : voiture, mobilier, consommation
    'credit_particuliers': {2023: 0.031, 2024: 0.023, 2025: 0.017},

    # Crédits aux professionnels : morale non terme, engagement par signature
    'credit_professionnels': {2023: 0.027, 2024: 0.029, 2025: 0.036},

    # Crédits de gestion (court terme professionnel, escompte)
    'credit_gestion': {2023: 0.044, 2024: 0.035, 2025: 0.058},

    # Monétique : parc de cartes bancaires
    'cartes': {2023: 0.050, 2024: 0.053, 2025: 0.066},

    # Monétique : parc de terminaux de paiement
    'tpe': {2023: 0.080, 2024: 0.080, 2025: 0.100},

    # Comptes clients : bancarisation (hypothèse, non publié par agence)
    'comptes': {2023: 0.035, 2024: 0.030, 2025: 0.028},

    # Placements : collecte de l'épargne (hypothèse)
    'placements': {2023: 0.060, 2024: 0.055, 2025: 0.050},
}

SOURCES = [
    "BCT — Rapport annuel sur la supervision bancaire : crédits à l'économie 2023-2025",
    "BCT — Bulletin « Les paiements en chiffres en Tunisie » : monétique 2024-2025",
]


def segment_du_produit(type_fichier, code_type=None):
    """Associe une ligne de fichier au segment sectoriel qui la gouverne."""
    if type_fichier == 'comptes':
        return 'comptes'
    if type_fichier == 'cartes':
        return 'cartes'
    if type_fichier == 'placements':
        return 'placements'
    if type_fichier == 'credits_gestion':
        return 'credit_gestion'
    if type_fichier == 'cmlt':
        return 'credit_professionnels'
    if type_fichier == 'credits_pp':
        return 'credit_particuliers'
    if type_fichier == 'tpe':
        return 'tpe'
    return 'comptes'


# ------------------------------------------------------------------
# Profil des agences, appris sur les données réelles
# ------------------------------------------------------------------

def profiler_agences(annee_reference):
    """
    Classe les agences par profil d'activité à partir des données réelles,
    puis calcule pour chacune :
      - son groupe d'appartenance,
      - l'orientation de son portefeuille (part professionnelle),
      - un coefficient de dynamisme propre à son groupe.

    Renvoie un dictionnaire, ou None si les données sont insuffisantes.
    """
    noms_agences, noms_features, matrice = construire_matrice(annee_reference)
    if len(noms_agences) < 6:
        return None

    _, donnees = preparer_donnees(matrice)

    # Nombre de groupes retenu par maximisation de la silhouette
    meilleur_k, meilleur_score, meilleures_etiquettes = 2, -1.0, None
    for k in range(2, min(7, len(noms_agences) - 1)):
        etiquettes = KMeans(n_clusters=k, n_init=25, random_state=0).fit_predict(donnees)
        score = silhouette_score(donnees, etiquettes)
        if score > meilleur_score:
            meilleur_k, meilleur_score, meilleures_etiquettes = k, score, etiquettes

    # Repérage des colonnes de crédit professionnel vs particulier
    indices_pro, indices_part = [], []
    for j, nom in enumerate(noms_features):
        minuscule = nom.lower()
        if 'gestion' in minuscule or 'morale' in minuscule or 'engagement' in minuscule:
            indices_pro.append(j)
        elif 'voiture' in minuscule or 'mobilier' in minuscule or 'consommation' in minuscule:
            indices_part.append(j)

    volumes_pro = matrice[:, indices_pro].sum(axis=1) if indices_pro else np.zeros(len(noms_agences))
    volumes_part = matrice[:, indices_part].sum(axis=1) if indices_part else np.zeros(len(noms_agences))
    total_credit = volumes_pro + volumes_part
    total_credit[total_credit == 0] = 1.0
    part_pro = volumes_pro / total_credit

    # Dynamisme par groupe : calé sur la dispersion réelle des volumes
    volumes = matrice.sum(axis=1)
    volumes_log = np.log1p(volumes)
    dispersion = float(np.std(volumes_log)) or 1.0

    dynamisme_groupe = {}
    for groupe in sorted(set(meilleures_etiquettes)):
        masque = meilleures_etiquettes == groupe
        ecart = float(np.mean(volumes_log[masque]) - np.mean(volumes_log)) / dispersion
        dynamisme_groupe[int(groupe)] = float(np.clip(ecart * 0.012, -0.025, 0.025))

    profils = {}
    for i, nom in enumerate(noms_agences):
        code = nom.split(' - ')[0].strip() if ' - ' in nom else nom[:3]
        groupe = int(meilleures_etiquettes[i])
        profils[code] = {
            'groupe': groupe,
            'part_pro': float(part_pro[i]),
            'dynamisme': dynamisme_groupe[groupe],
            'volume': float(volumes[i]),
        }

    return {
        'profils': profils,
        'k': meilleur_k,
        'silhouette': round(float(meilleur_score), 3),
        'n_agences': len(noms_agences),
    }


def taux_agence(code_agence, segment, annee, profils):
    """
    Taux de croissance appliqué à une agence pour un segment et une année.

    Le taux sectoriel BCT est modulé :
      - pour les crédits, par la composition réelle du portefeuille,
      - pour tous les segments, par le dynamisme du groupe d'appartenance.
    """
    table = TAUX_SECTORIELS.get(segment, TAUX_SECTORIELS['comptes'])
    taux = table.get(annee, list(table.values())[-1])

    profil = profils.get(code_agence) if profils else None
    if profil is None:
        return taux

    if segment in ('credit_particuliers', 'credit_professionnels'):
        taux_part = TAUX_SECTORIELS['credit_particuliers'].get(annee, 0.02)
        taux_pro = TAUX_SECTORIELS['credit_professionnels'].get(annee, 0.03)
        p = profil['part_pro']
        taux = p * taux_pro + (1 - p) * taux_part

    return taux + profil['dynamisme']