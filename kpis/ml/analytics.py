"""
Module d'analyse du réseau d'agences.

Trois méthodes non supervisées appliquées à une coupe transversale
(une année, une agence par ligne) :

  1. Segmentation    : ACP + K-Means, k choisi par score de silhouette
  2. Anomalies       : Isolation Forest sur l'espace standardisé
  3. Profils latents : NMF sur la matrice agence x produit

Parti pris méthodologique : les agences sont décrites par la RÉPARTITION
de leur activité (données compositionnelles) et non par des volumes bruts,
afin que la segmentation capture le modèle commercial et non la taille.
La taille est réintroduite comme variable distincte, en échelle log.
"""

import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, NMF
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest
from .lda import extraire_profils_lda, expliquer_lda
from django.db.models import Sum
from .anomalies import detecter_anomalies_test, expliquer_anomalies, lignes_lisibles
from kpis.models import (
    Agence, CompteClient, Credit, VenteCarte, VenteTPE, Placement,
)

from .coda import preparer_donnees_coda, comparer_representations
# Agences écartées des analyses statistiques.
#
# NEO-BTE (031) est l'agence digitale : elle centralise les ouvertures de
# comptes et la délivrance de cartes pour l'ensemble du réseau, sans activité
# de crédit propre (1 dossier de crédit pour 82 783 opérations par ailleurs).
# Son profil n'est comparable à aucune agence physique et fausse toute
# estimation portant sur le réseau.
AGENCES_EXCLUES = ['031']


# Regroupement des 23 types de cartes en 7 familles produit,
# pour limiter la dimensionnalité (une trentaine d'observations seulement).
FAMILLES_CARTES = {
    'VCENA': 'Visa classique', 'VCENI': 'Visa classique',
    'VBSNI': 'Visa classique', 'VBSNIP': 'Visa classique',
    'VGPIA': 'Visa Gold', 'VGPII': 'Visa Gold',
    'VGPNA': 'Visa Gold', 'VGPNI': 'Visa Gold',
    'VINFIA': 'Visa Infinite', 'VINFII': 'Visa Infinite',
    'VINFNA': 'Visa Infinite', 'VINFNI': 'Visa Infinite',
    'MPIA': 'Mastercard', 'MPII': 'Mastercard',
    'MPNA': 'Mastercard', 'MPNI': 'Mastercard', 'MVI': 'Mastercard',
    'CTIA': 'Technologique', 'CTII': 'Technologique',
    'EPARGNE': 'Épargne', 'GREEN': 'Épargne',
    'BNPL': 'Autres', 'CAT': 'Autres',
}

ORDRE_FAMILLES_CARTES = [
    'Visa classique', 'Visa Gold', 'Visa Infinite',
    'Mastercard', 'Technologique', 'Épargne', 'Autres',
]

LIBELLES_CREDIT = {
    'voiture': 'Crédit voiture',
    'mobilier': 'Crédit mobilier',
    'consommation': 'Crédit consommation',
    'gestion': 'Crédit gestion',
    'morale_non_terme': 'Crédit morale non terme',
    'engagement_signature': 'Engagement par signature',
}


# ------------------------------------------------------------------
# Construction de la matrice de données
# ------------------------------------------------------------------

def construire_matrice(annee):
    """
    Renvoie (noms_agences, noms_features, matrice_comptages).

    matrice_comptages : tableau (n_agences x n_features) de comptages bruts.
    Les agences listées dans AGENCES_EXCLUES sont écartées, ainsi que
    celles ne présentant aucune donnée pour l'année demandée.
    """
    agences = list(Agence.objects.exclude(code_agence__in=AGENCES_EXCLUES))
    index_agence = {a.id: i for i, a in enumerate(agences)}

    # --- Définition des colonnes ---
    categories_compte = sorted(
        CompteClient.objects.filter(annee=annee)
        .values_list('categorie__libelle', flat=True).distinct()
    )
    categories_credit = [c for c, _ in Credit.CATEGORIE_CHOICES]
    types_placement = sorted(
        Placement.objects.filter(annee=annee)
        .values_list('type_placement__libelle', flat=True).distinct()
    )

    colonnes = (
        [('compte', c) for c in categories_compte]
        + [('credit', c) for c in categories_credit]
        + [('carte', f) for f in ORDRE_FAMILLES_CARTES]
        + [('placement', p) for p in types_placement]
        + [('tpe', 'TPE')]
    )
    index_colonne = {cle: j for j, cle in enumerate(colonnes)}

    matrice = np.zeros((len(agences), len(colonnes)), dtype=float)

    # --- Remplissage ---
    # Les lignes rattachées à une agence exclue sont ignorées silencieusement.
    for ligne in (CompteClient.objects.filter(annee=annee)
                  .values('agence_id', 'categorie__libelle')
                  .annotate(total=Sum('nombre'))):
        i = index_agence.get(ligne['agence_id'])
        cle = ('compte', ligne['categorie__libelle'])
        if i is not None and cle in index_colonne:
            matrice[i, index_colonne[cle]] += ligne['total'] or 0

    for ligne in (Credit.objects.filter(annee=annee)
                  .values('agence_id', 'categorie')
                  .annotate(total=Sum('nombre'))):
        i = index_agence.get(ligne['agence_id'])
        cle = ('credit', ligne['categorie'])
        if i is not None and cle in index_colonne:
            matrice[i, index_colonne[cle]] += ligne['total'] or 0

    for ligne in (VenteCarte.objects.filter(annee=annee)
                  .values('agence_id', 'type_carte__code')
                  .annotate(total=Sum('nombre'))):
        i = index_agence.get(ligne['agence_id'])
        if i is None:
            continue
        famille = FAMILLES_CARTES.get(ligne['type_carte__code'], 'Autres')
        matrice[i, index_colonne[('carte', famille)]] += ligne['total'] or 0

    for ligne in (Placement.objects.filter(annee=annee)
                  .values('agence_id', 'type_placement__libelle')
                  .annotate(total=Sum('nombre'))):
        i = index_agence.get(ligne['agence_id'])
        cle = ('placement', ligne['type_placement__libelle'])
        if i is not None and cle in index_colonne:
            matrice[i, index_colonne[cle]] += ligne['total'] or 0

    for ligne in (VenteTPE.objects.filter(annee=annee)
                  .values('agence_id').annotate(total=Sum('nombre'))):
        i = index_agence.get(ligne['agence_id'])
        if i is not None:
            matrice[i, index_colonne[('tpe', 'TPE')]] += ligne['total'] or 0

    # --- Retrait des agences sans aucune donnée ---
    actives = matrice.sum(axis=1) > 0
    matrice = matrice[actives]
    noms_agences = [str(a) for a, garde in zip(agences, actives) if garde]

    noms_features = []
    for domaine, valeur in colonnes:
        if domaine == 'credit':
            noms_features.append(LIBELLES_CREDIT.get(valeur, valeur))
        elif domaine == 'carte':
            noms_features.append(f"Carte {valeur}")
        elif domaine == 'compte':
            noms_features.append(valeur)
        elif domaine == 'placement':
            noms_features.append(f"Placement {valeur}")
        else:
            noms_features.append("Ventes TPE")

    # --- Retrait des colonnes constamment nulles ---
    colonnes_utiles = matrice.sum(axis=0) > 0
    matrice = matrice[:, colonnes_utiles]
    noms_features = [n for n, garde in zip(noms_features, colonnes_utiles) if garde]

    return noms_agences, noms_features, matrice


def preparer_donnees(matrice, compositionnel=True):
    """
    Transforme les comptages bruts en un espace exploitable.

    Par défaut, les parts d'activité sont traitées comme des données
    compositionnelles (transformation log-rapport centré), ce qui
    corrige les distances et supprime les corrélations induites par la
    contrainte de somme.

    Le paramètre permet de revenir à la standardisation directe des
    parts, afin de comparer les deux représentations.
    """
    if compositionnel:
        return preparer_donnees_coda(matrice)

    totaux = matrice.sum(axis=1, keepdims=True)
    totaux[totaux == 0] = 1.0
    parts = matrice / totaux

    taille = np.log1p(matrice.sum(axis=1)).reshape(-1, 1)

    donnees = np.hstack([parts, taille])
    donnees_std = StandardScaler().fit_transform(donnees)

    return parts, donnees_std
# ------------------------------------------------------------------
# 1. Segmentation : ACP + K-Means
# ------------------------------------------------------------------

def segmenter(donnees_std, noms_features, k_min=2, k_max=6):
    """
    Projette en 2 dimensions par ACP, puis segmente par K-Means.
    Le nombre de groupes est choisi par maximisation du score de silhouette.
    """
    n = donnees_std.shape[0]
    k_max = min(k_max, n - 1)

    acp = PCA(n_components=2, random_state=0)
    coordonnees = acp.fit_transform(donnees_std)
    variance_expliquee = acp.explained_variance_ratio_

    resultats_k = []
    meilleur = None

    for k in range(k_min, k_max + 1):
        modele = KMeans(n_clusters=k, n_init=25, random_state=0)
        etiquettes = modele.fit_predict(donnees_std)
        if len(set(etiquettes)) < 2:
            continue
        score = silhouette_score(donnees_std, etiquettes)
        resultats_k.append({'k': k, 'silhouette': round(float(score), 3)})
        if meilleur is None or score > meilleur['score']:
            meilleur = {'k': k, 'score': score, 'etiquettes': etiquettes}

    if meilleur is None:
        return None

    # Axes de l'ACP : variables les plus contributives
    noms_etendus = noms_features + ["Taille (log)"]
    axes = []
    for i in range(2):
        poids = acp.components_[i]
        ordre = np.argsort(-np.abs(poids))[:4]
        axes.append({
            'numero': i + 1,
            'variance': round(float(variance_expliquee[i]) * 100, 1),
            'variables': [
                {'nom': noms_etendus[j], 'poids': round(float(poids[j]), 2)}
                for j in ordre
            ],
        })

    return {
        'k': meilleur['k'],
        'silhouette': round(float(meilleur['score']), 3),
        'etiquettes': meilleur['etiquettes'],
        'coordonnees': coordonnees,
        'axes': axes,
        'variance_totale': round(float(variance_expliquee.sum()) * 100, 1),
        'balayage': resultats_k,
    }


def caracteriser_groupes(donnees_std, etiquettes, noms_features):
    """
    Pour chaque groupe, identifie les variables qui le distinguent
    le plus de la moyenne du réseau (écart en unités d'écart-type).
    """
    noms_etendus = noms_features + ["Taille (log)"]
    groupes = []

    for c in sorted(set(etiquettes)):
        masque = etiquettes == c
        profil_moyen = donnees_std[masque].mean(axis=0)
        ordre = np.argsort(-np.abs(profil_moyen))[:4]
        groupes.append({
            'numero': int(c) + 1,
            'effectif': int(masque.sum()),
            'traits': [
                {
                    'nom': noms_etendus[j],
                    'ecart': round(float(profil_moyen[j]), 2),
                    'sens': 'fort' if profil_moyen[j] > 0 else 'faible',
                }
                for j in ordre
            ],
        })
    return groupes


# ------------------------------------------------------------------
# 2. Détection d'anomalies : Isolation Forest
# ------------------------------------------------------------------

def detecter_anomalies(donnees_std, noms_agences, noms_features, contamination=0.12):
    """
    Isolation Forest : isole les agences dont le profil d'activité
    s'écarte structurellement du reste du réseau.
    """
    modele = IsolationForest(
        contamination=contamination, n_estimators=300, random_state=0
    )
    predictions = modele.fit_predict(donnees_std)
    scores = modele.score_samples(donnees_std)   # plus bas = plus atypique

    noms_etendus = noms_features + ["Taille (log)"]
    anomalies = []

    for i, (prediction, score) in enumerate(zip(predictions, scores)):
        if prediction == -1:
            ecarts = donnees_std[i]
            j = int(np.argmax(np.abs(ecarts)))
            anomalies.append({
                'agence': noms_agences[i],
                'score': round(float(score), 3),
                'variable': noms_etendus[j],
                'ecart': round(float(ecarts[j]), 2),
                'sens': 'supérieur' if ecarts[j] > 0 else 'inférieur',
            })

    anomalies.sort(key=lambda a: a['score'])
    return anomalies, predictions


# ------------------------------------------------------------------
# 3. Profils latents : NMF
# ------------------------------------------------------------------

def extraire_profils_latents(parts, noms_agences, noms_features, n_profils=3):
    """
    Factorisation en matrices non négatives.

    La matrice des parts d'activité est décomposée en un petit nombre
    d'archétypes ; chaque agence s'exprime comme un mélange de ces
    archétypes, ce qui est plus nuancé qu'une affectation exclusive
    à un groupe.
    """
    n_profils = min(n_profils, parts.shape[0], parts.shape[1])

    modele = NMF(
        n_components=n_profils, init='nndsvda',
        max_iter=800, random_state=0
    )
    poids_agences = modele.fit_transform(parts)   # agences x profils
    composition = modele.components_              # profils x produits

    erreur = float(modele.reconstruction_err_)

    profils = []
    for p in range(n_profils):
        ordre = np.argsort(-composition[p])[:5]
        total = composition[p].sum() or 1.0
        profils.append({
            'numero': p + 1,
            'produits': [
                {
                    'nom': noms_features[j],
                    'poids': round(float(composition[p][j] / total) * 100, 1),
                }
                for j in ordre
            ],
        })

    # Composition de chaque agence en pourcentage
    sommes = poids_agences.sum(axis=1, keepdims=True)
    sommes[sommes == 0] = 1.0
    parts_agences = poids_agences / sommes

    appartenances = []
    for i, nom in enumerate(noms_agences):
        appartenances.append({
            'agence': nom,
            'parts': [round(float(v) * 100, 1) for v in parts_agences[i]],
            'dominant': int(np.argmax(parts_agences[i])) + 1,
        })
    appartenances.sort(key=lambda a: (a['dominant'], -max(a['parts'])))

    return {
        'profils': profils,
        'appartenances': appartenances,
        'erreur': round(erreur, 4),
    }


# ------------------------------------------------------------------
# Génération des explications en langage courant
# ------------------------------------------------------------------

def _formuler_orientation(traits):
    """Traduit les écarts statistiques d'un groupe en phrase lisible."""
    forts = [t['nom'] for t in traits if t['ecart'] > 0.5][:2]
    faibles = [t['nom'] for t in traits if t['ecart'] < -0.5][:1]

    phrase = ""
    if forts:
        phrase = "Ces agences se distinguent par " + " et ".join(
            f"leur activité sur {f.lower()}" for f in forts
        )
    if faibles:
        liaison = ", mais restent en retrait sur " if phrase else "Ces agences sont surtout en retrait sur "
        phrase += liaison + faibles[0].lower()
    return (phrase or "Ces agences présentent un profil proche de la moyenne du réseau") + "."


def _qualifier_ecart(ecart):
    """Transforme un écart-type en appréciation verbale."""
    a = abs(ecart)
    if a >= 3:
        return "très nettement"
    if a >= 2:
        return "nettement"
    if a >= 1:
        return "sensiblement"
    return "légèrement"


def rediger_explications(resultat):
    """
    Produit des commentaires en français courant à partir des résultats
    chiffrés. Le texte s'adapte aux valeurs réellement obtenues.
    """
    if not resultat:
        return {}

    n = resultat['n_agences']
    k = resultat['k']
    sil = resultat['silhouette']
    var = resultat['variance_totale']

    # --- Points clés ---
    points_cles = []

    if sil >= 0.5:
        points_cles.append(f"Les {n} agences se répartissent clairement en {k} familles bien distinctes.")
        niveau = "solide"
    elif sil >= 0.3:
        points_cles.append(f"Les {n} agences se répartissent en {k} familles, aux contours un peu flous.")
        niveau = "moyen"
    else:
        points_cles.append(f"Les {n} agences se ressemblent beaucoup : pas de familles vraiment tranchées.")
        points_cles.append(f"Le découpage en {k} groupes donne des tendances, pas des catégories strictes.")
        niveau = "indicatif"

    petits = [g for g in resultat['groupes'] if g['effectif'] == 1]
    if petits:
        points_cles.append(
            f"{len(petits)} groupe ne contient qu'une agence : c'est un cas isolé, pas une famille."
        )
    anomalies = resultat['anomalies']
    if anomalies:
        noms = ", ".join(a['agence'] for a in anomalies)
        points_cles.append(f"{len(anomalies)} agences ont un profil réellement atypique : {noms}.")
        points_cles.append(
            "Sortir de l'ordinaire n'est pas un problème : c'est une activité "
            "différente du reste du réseau."
        )
    else:
        points_cles.append(
            "Aucune agence ne se démarque de façon établie : les écarts observés "
            "restent dans ce que le hasard peut produire."
        )

    # --- Lecture de la carte ---
    axe1 = resultat['axes'][0]['variables']
    axe2 = resultat['axes'][1]['variables']
    pos1 = [v['nom'] for v in axe1 if v['poids'] > 0][:1]
    neg1 = [v['nom'] for v in axe1 if v['poids'] < 0][:1]
    pos2 = [v['nom'] for v in axe2 if v['poids'] > 0][:1]

    points_carte = [
        "Chaque point est une agence ; deux agences proches travaillent de façon similaire."
    ]
    if pos1 and neg1:
        points_carte.append(f"De gauche à droite : de {neg1[0].lower()} vers {pos1[0].lower()}.")
    if pos2:
        points_carte.append(f"Vers le haut : les agences les plus actives sur {pos2[0].lower()}.")
    points_carte.append(
        f"La carte résume {var}% de l'information : une bonne vue d'ensemble, sans tous les détails."
    )
    points_carte.append("Un cercle rouge signale une agence qui sort de l'ordinaire.")

    # --- Commentaire par groupe ---
    commentaires_groupes = {}
    for g in resultat['groupes']:
        if g['effectif'] == 1:
            texte = "Agence seule dans son cas : son activité ne ressemble à aucune autre."
        else:
            texte = _formuler_orientation(g['traits'])
        commentaires_groupes[g['numero']] = texte

    # --- Détail des agences atypiques ---
   
    lignes_anomalies = lignes_lisibles(resultat.get('test_anomalies'))
    # --- Profils types ---
    profils_nommes = []
    for p in resultat['latents']['profils']:
        principal = p['produits'][0]
        secondaire = p['produits'][1] if len(p['produits']) > 1 else None
        description = f"Surtout {principal['nom'].lower()} ({principal['poids']}%)"
        if secondaire and secondaire['poids'] >= 10:
            description += f", puis {secondaire['nom'].lower()} ({secondaire['poids']}%)"
        profils_nommes.append({
            'numero': p['numero'],
            'titre': f"Profil {p['numero']}",
            'description': description + ".",
            'produits': p['produits'],
        })

    points_melanges = [
        "La plupart des agences combinent plusieurs façons de travailler.",
        "La barre colorée montre la part de chaque profil dans l'activité de l'agence.",
    ]

    return {
        'niveau': niveau,
        'points_cles': points_cles,
        'points_carte': points_carte,
        'groupes': commentaires_groupes,
        'lignes_anomalies': lignes_anomalies,
        'profils': profils_nommes,
        'points_melanges': points_melanges,
    }


# ------------------------------------------------------------------
# Point d'entrée
# ------------------------------------------------------------------

def analyser_reseau(annee, n_profils=3):
    """
    Exécute les trois analyses et renvoie un dictionnaire prêt
    pour l'affichage, ou None si les données sont insuffisantes.
    """
    noms_agences, noms_features, matrice = construire_matrice(annee)

    if len(noms_agences) < 5 or len(noms_features) < 3:
        return None

    parts, donnees_std = preparer_donnees(matrice)

    
    segmentation = segmenter(donnees_std, noms_features)
    if segmentation is None:
        return None

    # Contrôle de l'apport du traitement compositionnel : la même
    # procédure de segmentation est appliquée aux deux représentations.
    def _segmenter_pour_comparaison(donnees):
        meilleur_score, meilleures_etiquettes = -1.0, None
        for k in range(2, min(7, donnees.shape[0] - 1)):
            etiquettes = KMeans(n_clusters=k, n_init=25, random_state=0).fit_predict(donnees)
            score = silhouette_score(donnees, etiquettes)
            if score > meilleur_score:
                meilleur_score, meilleures_etiquettes = score, etiquettes
        return meilleures_etiquettes, meilleur_score

    comparaison_espaces = comparer_representations(matrice, _segmenter_pour_comparaison)
    groupes = caracteriser_groupes(donnees_std, segmentation['etiquettes'], noms_features)
    test_anomalies = detecter_anomalies_test(donnees_std, noms_agences, noms_features)
    anomalies = test_anomalies['signalees'] if test_anomalies else []
    signalees_noms = {a['agence'] for a in anomalies}   
    # NMF conservée pour comparaison : deux décompositions indépendantes
    # qui convergent renforcent la lecture des profils.
    latents = extraire_profils_latents(parts, noms_agences, noms_features, n_profils)
    lda = extraire_profils_lda(matrice, noms_agences, noms_features)
    # Points du plan factoriel, pour le nuage de points
    points = []
    for i, nom in enumerate(noms_agences):
        points.append({
            'agence': nom,
            'x': round(float(segmentation['coordonnees'][i, 0]), 3),
            'y': round(float(segmentation['coordonnees'][i, 1]), 3),
            'groupe': int(segmentation['etiquettes'][i]) + 1,
            'anomalie': nom in signalees_noms,
            'volume': int(matrice[i].sum()),
        })

    resultat = {
        'n_agences': len(noms_agences),
        'n_variables': len(noms_features) + 1,
        'k': segmentation['k'],
        'silhouette': segmentation['silhouette'],
        'variance_totale': segmentation['variance_totale'],
        'axes': segmentation['axes'],
        'balayage': segmentation['balayage'],
        'groupes': groupes,
        'points': points,
        'anomalies': anomalies,
        'latents': latents,
        'comparaison_espaces': comparaison_espaces,
    }
    resultat['test_anomalies'] = test_anomalies
    resultat['exp_anomalies'] = expliquer_anomalies(test_anomalies)
    resultat['lda'] = lda
    resultat['exp_lda'] = expliquer_lda(lda)
    resultat['explications'] = rediger_explications(resultat)
    return resultat