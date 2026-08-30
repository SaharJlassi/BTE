from datetime import date

from django.contrib.auth.models import User
from django.contrib.auth.hashers import make_password
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.db.models import Sum, Q, DecimalField
from django.db.models.functions import Coalesce
from django.http import Http404, HttpResponse
from django.core.cache import cache

from .models import (
    Profil, Agence,
    CompteClient, Credit, VenteCarte, VenteTPE, Placement, Credoc,
)
from .forms import (
    CompteClientForm, CreditForm, VenteCarteForm,
    VenteTPEForm, PlacementForm, CredocForm,
)
from .excel_import import TYPES_FICHIER, importer_fichier
from .excel_export import TYPES_EXPORT, generer_export

from .ml.analytics import analyser_reseau, construire_matrice, preparer_donnees
from .ml.ml_avance import CIBLES, analyser_performance, expliquer_performance
from .ml.bayesien import ajuster_performance, expliquer_bayesien
from .ml.conforme import expliquer_conforme
from .ml.allocation import optimiser, expliquer_allocation
from .ml.graphe import analyser_communautes, expliquer_communautes


# --- Registre des types de KPI saisissables manuellement ---
MODELES_KPI = {
    'compte':    (CompteClient, CompteClientForm, 'Comptes clients'),
    'credit':    (Credit,       CreditForm,       'Crédits'),
    'carte':     (VenteCarte,   VenteCarteForm,   'Ventes Cartes'),
    'tpe':       (VenteTPE,     VenteTPEForm,     'Ventes TPE'),
    'placement': (Placement,    PlacementForm,    'Placements'),
    'credoc':    (Credoc,       CredocForm,       'CREDOC'),
}

# Modèles portant un champ 'annee', interrogés pour connaître les
# exercices renseignés.
MODELES_ANNUELS = (CompteClient, VenteCarte, Credit, Placement, VenteTPE)

# Durée de conservation des résultats d'analyse, en secondes.
DUREE_CACHE = 3600


def _get_type_kpi(type_kpi):
    if type_kpi not in MODELES_KPI:
        raise Http404("Type de KPI inconnu.")
    return MODELES_KPI[type_kpi]


def _annees_renseignees():
    """
    Exercices pour lesquels des données existent réellement en base.
    """
    annees = set()
    for modele in MODELES_ANNUELS:
        annees.update(modele.objects.values_list('annee', flat=True).distinct())
    return sorted(a for a in annees if a)


def _derniere_annee_disponible():
    """
    Dernier exercice renseigné.

    Sert de valeur par défaut aux pages d'analyse : ouvrir sur l'année
    en cours conduirait à une page vide tant qu'aucune donnée n'a été
    importée pour cet exercice.
    """
    annees = _annees_renseignees()
    return max(annees) if annees else date.today().year


def _annees_disponibles():
    """
    Exercices proposés dans les menus des pages d'analyse.

    Seules les années renseignées sont listées : proposer un exercice
    vide mènerait l'utilisateur vers une page sans résultat.
    """
    return _annees_renseignees() or [date.today().year]


def _annees_saisie():
    """
    Exercices proposés à l'import et à l'export.

    L'année en cours y figure toujours, puisqu'elle doit rester
    sélectionnable pour recevoir de nouvelles données.
    """
    return sorted(set(_annees_renseignees()) | {date.today().year})


def _vider_cache_analyses():
    """
    Invalide les résultats d'analyse conservés en cache.

    Appelée après toute modification des données : les résultats
    conservés porteraient sinon sur un état périmé de la base.
    """
    cache.clear()


@login_required
def dashboard(request):
    try:
        profil = request.user.profil
    except Profil.DoesNotExist:
        messages.error(request, "Aucun profil associé à ce compte. Contactez l'administrateur.")
        return redirect('login')

    # --- Année (par défaut, le dernier exercice renseigné) ---
    annees_dispo = _annees_disponibles()
    annee_str = request.GET.get('annee')
    try:
        annee = int(annee_str) if annee_str else _derniere_annee_disponible()
    except ValueError:
        annee = _derniere_annee_disponible()

    # --- Agence(s) concernée(s) ---
    agences_dispo = None
    agence_id = request.GET.get('agence', '')

    if profil.est_super_admin():
        agences_dispo = Agence.objects.all()
        if agence_id:
            try:
                agence_obj = Agence.objects.get(id=agence_id)
                agence_filter = Q(agence_id=agence_id)
                agence_label = str(agence_obj)
            except Agence.DoesNotExist:
                agence_filter = Q()
                agence_label = "Toutes les agences"
                agence_id = ''
        else:
            agence_filter = Q()
            agence_label = "Toutes les agences"
    else:
        if not profil.agence:
            messages.error(request, "Aucune agence associée à votre profil.")
            return redirect('login')
        agence_filter = Q(agence=profil.agence)
        agence_label = str(profil.agence)

    periode = Q(annee=annee)

    # --- 1. Comptes clients (par catégorie) ---
    comptes_qs = CompteClient.objects.filter(agence_filter & periode)
    comptes_par_categorie = list(
        comptes_qs
        .values('categorie__libelle')
        .annotate(total=Coalesce(Sum('nombre'), 0))
        .order_by('-total')
    )
    nb_comptes_total = comptes_qs.aggregate(total=Coalesce(Sum('nombre'), 0))['total']

    # --- 2. Crédits (par catégorie) ---
    credits_qs = Credit.objects.filter(agence_filter & periode)
    credits_bruts = (
        credits_qs
        .values('categorie')
        .annotate(
            total=Coalesce(Sum('nombre'), 0),
            montant_total=Coalesce(Sum('montant'), 0, output_field=DecimalField())
        )
        .order_by('-total')
    )
    categorie_labels = dict(Credit.CATEGORIE_CHOICES)
    credits_par_categorie = [
        {**c, 'categorie_label': categorie_labels.get(c['categorie'], c['categorie'])}
        for c in credits_bruts
    ]

    # --- 3. Ventes cartes (par type) ---
    cartes_qs = VenteCarte.objects.filter(agence_filter & periode)
    cartes_par_type = list(
        cartes_qs
        .values('type_carte__libelle')
        .annotate(total=Coalesce(Sum('nombre'), 0))
        .order_by('-total')
    )
    nb_cartes_total = cartes_qs.aggregate(total=Coalesce(Sum('nombre'), 0))['total']

    # --- 4. Ventes TPE ---
    tpe_qs = VenteTPE.objects.filter(agence_filter & periode)
    agg_tpe = tpe_qs.aggregate(
        total_nombre=Coalesce(Sum('nombre'), 0),
        total_montant=Coalesce(Sum('montant'), 0, output_field=DecimalField()),
    )

    # --- 5. Placements (par type) ---
    placements_qs = Placement.objects.filter(agence_filter & periode)
    placements_par_type = list(
        placements_qs
        .values('type_placement__libelle')
        .annotate(total=Coalesce(Sum('nombre'), 0))
        .order_by('-total')
    )

    # --- 6. Credoc (import / export) ---
    credoc_qs = Credoc.objects.filter(agence_filter & Q(date__year=annee))
    credoc_import = credoc_qs.filter(sens='import').aggregate(
        total=Coalesce(Sum('montant'), 0, output_field=DecimalField())
    )['total']
    credoc_export = credoc_qs.filter(sens='export').aggregate(
        total=Coalesce(Sum('montant'), 0, output_field=DecimalField())
    )['total']

    # --- Données pour les graphiques (Chart.js) ---
    donnees_graphiques = {
        'comptes': {
            'labels': [c['categorie__libelle'] for c in comptes_par_categorie],
            'valeurs': [c['total'] for c in comptes_par_categorie],
        },
        'credits': {
            'labels': [c['categorie_label'] for c in credits_par_categorie],
            'valeurs': [c['total'] for c in credits_par_categorie],
        },
        'cartes': {
            'labels': [c['type_carte__libelle'] for c in cartes_par_type],
            'valeurs': [c['total'] for c in cartes_par_type],
        },
    }

    context = {
        'profil': profil,
        'agence': agence_label,
        'agences_dispo': agences_dispo,
        'agence_selectionnee': agence_id,
        'annee': annee,
        'annees_dispo': annees_dispo,

        'nb_comptes_total': nb_comptes_total,
        'comptes_par_categorie': comptes_par_categorie,

        'credits_par_categorie': credits_par_categorie,

        'nb_cartes_total': nb_cartes_total,
        'cartes_par_type': cartes_par_type,

        'total_tpe': agg_tpe['total_nombre'],
        'montant_tpe': agg_tpe['total_montant'],

        'placements_par_type': placements_par_type,

        'credoc_import': credoc_import,
        'credoc_export': credoc_export,

        'donnees_graphiques': donnees_graphiques,
    }
    return render(request, 'dashboard.html', context)


# ============================================================
# IMPORT EXCEL
# ============================================================

@login_required
def importer_excel(request):
    profil = request.user.profil

    if not profil.peut_importer_excel():
        messages.error(request, "Accès refusé.")
        return redirect('dashboard')

    if profil.est_chef_agence() and not profil.agence:
        messages.error(request, "Aucune agence associée à votre profil.")
        return redirect('dashboard')

    annees_dispo = _annees_saisie()
    rapport = None
    type_choisi = ''
    annee_choisie = date.today().year

    if request.method == 'POST':
        type_choisi = request.POST.get('type_fichier', '')
        fichier = request.FILES.get('fichier')

        try:
            annee_choisie = int(request.POST.get('annee'))
        except (TypeError, ValueError):
            annee_choisie = None

        if not fichier:
            messages.error(request, "Veuillez sélectionner un fichier Excel.")
        elif type_choisi not in dict(TYPES_FICHIER):
            messages.error(request, "Veuillez choisir un type de données valide.")
        elif annee_choisie is None:
            messages.error(request, "Veuillez choisir une année valide.")
        else:
            agence_limitee = profil.agence if profil.est_chef_agence() else None
            try:
                rapport = importer_fichier(
                    fichier, fichier.name, type_choisi, annee_choisie, agence_limitee
                )
                if rapport and rapport.get('importes'):
                    _vider_cache_analyses()
            except ValueError as e:
                messages.error(request, str(e))
            except Exception as e:
                messages.error(request, f"Impossible de lire le fichier : {e}")

    context = {
        'profil': profil,
        'types_fichier': TYPES_FICHIER,
        'annees_dispo': annees_dispo,
        'type_choisi': type_choisi,
        'annee_choisie': annee_choisie or date.today().year,
        'rapport': rapport,
    }
    return render(request, 'importer_excel.html', context)


# ============================================================
# EXPORT EXCEL — réservé au super_admin
# ============================================================

@login_required
def exporter_excel(request):
    profil = request.user.profil

    if not profil.peut_exporter_excel():
        messages.error(request, "Accès refusé.")
        return redirect('dashboard')

    annees_dispo = _annees_saisie()

    if request.method == 'POST':
        type_export = request.POST.get('type_export', '')
        agence_id = request.POST.get('agence', '')

        try:
            annee = int(request.POST.get('annee'))
        except (TypeError, ValueError):
            messages.error(request, "Année invalide.")
            return redirect('exporter_excel')

        if type_export not in dict(TYPES_EXPORT):
            messages.error(request, "Type de données invalide.")
            return redirect('exporter_excel')

        agence = None
        if agence_id:
            agence = get_object_or_404(Agence, id=agence_id)

        contenu, nom_fichier = generer_export(type_export, annee, agence)

        reponse = HttpResponse(
            contenu,
            content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        reponse['Content-Disposition'] = f'attachment; filename="{nom_fichier}"'
        return reponse

    context = {
        'profil': profil,
        'types_export': TYPES_EXPORT,
        'annees_dispo': annees_dispo,
        'annee_courante': _derniere_annee_disponible(),
        'agences': Agence.objects.all(),
    }
    return render(request, 'exporter_excel.html', context)


# ============================================================
# GESTION DES COMPTES UTILISATEURS
# ============================================================

@login_required
def gerer_comptes(request):
    profil = request.user.profil

    if profil.est_super_admin():
        comptes = Profil.objects.filter(role='chef_agence').select_related('user', 'agence')
    elif profil.est_chef_agence():
        comptes = Profil.objects.filter(role='employe', agence=profil.agence).select_related('user')
    else:
        messages.error(request, "Accès refusé.")
        return redirect('dashboard')

    return render(request, 'gerer_comptes.html', {'comptes': comptes, 'profil': profil})


@login_required
def creer_compte(request):
    profil = request.user.profil

    if not (profil.est_super_admin() or profil.est_chef_agence()):
        messages.error(request, "Accès refusé.")
        return redirect('dashboard')

    agences = Agence.objects.all() if profil.est_super_admin() else None

    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        if User.objects.filter(username=username).exists():
            messages.error(request, "Ce nom d'utilisateur existe déjà.")
            return redirect('creer_compte')

        if profil.est_super_admin():
            agence_id = request.POST.get('agence')
            agence_obj = get_object_or_404(Agence, id=agence_id)
            user = User.objects.create(
                username=username,
                password=make_password(password),
                is_staff=True,
            )
            Profil.objects.create(user=user, agence=agence_obj, role='chef_agence')
            messages.success(request, f"Chef d'agence {username} créé pour {agence_obj}.")

        else:
            user = User.objects.create(
                username=username,
                password=make_password(password),
                is_staff=False,
            )
            permissions = request.POST.getlist('permissions')
            Profil.objects.create(
                user=user,
                agence=profil.agence,
                role='employe',
                peut_gerer_comptes='compte' in permissions,
                peut_gerer_credits='credit' in permissions,
                peut_gerer_tpe='tpe' in permissions,
                peut_gerer_cartes='carte' in permissions,
                peut_gerer_credoc='credoc' in permissions,
                peut_gerer_placements='placement' in permissions,
            )
            messages.success(request, f"Employé {username} créé.")

        return redirect('gerer_comptes')

    return render(request, 'creer_compte.html', {'profil': profil, 'agences': agences})


@login_required
def suspendre_compte(request, profil_id):
    profil = request.user.profil
    if not profil.est_super_admin():
        messages.error(request, "Seul le Super Admin peut suspendre un compte.")
        return redirect('dashboard')

    cible = get_object_or_404(Profil, id=profil_id, role='chef_agence')
    cible.user.is_active = not cible.user.is_active
    cible.user.save()

    etat = "réactivé" if cible.user.is_active else "suspendu"
    messages.success(request, f"Compte {cible.user.username} {etat}.")
    return redirect('gerer_comptes')


@login_required
def supprimer_compte(request, profil_id):
    profil = request.user.profil

    if profil.est_super_admin():
        cible = get_object_or_404(Profil, id=profil_id, role='chef_agence')
    elif profil.est_chef_agence():
        cible = get_object_or_404(Profil, id=profil_id, role='employe', agence=profil.agence)
    else:
        messages.error(request, "Accès refusé.")
        return redirect('dashboard')

    cible.user.delete()
    messages.success(request, "Compte supprimé.")
    return redirect('gerer_comptes')


# ============================================================
# SAISIE MANUELLE DES KPIs
# ============================================================

@login_required
def liste_kpi(request, type_kpi):
    profil = request.user.profil
    model, form_class, label = _get_type_kpi(type_kpi)

    if not (profil.est_chef_agence() or profil.est_employe()):
        messages.error(request, "Accès refusé.")
        return redirect('dashboard')
    if not profil.agence:
        messages.error(request, "Aucune agence associée à votre profil.")
        return redirect('dashboard')

    objets = model.objects.filter(agence=profil.agence).order_by('-id')[:200]

    context = {
        'type_kpi': type_kpi,
        'label': label,
        'objets': objets,
        'peut_saisir': profil.peut_saisir_kpi(type_kpi),
        'peut_supprimer': profil.peut_supprimer_kpi(),
    }
    return render(request, 'liste_kpi.html', context)


@login_required
def ajouter_kpi(request, type_kpi):
    profil = request.user.profil
    model, form_class, label = _get_type_kpi(type_kpi)

    if not profil.peut_saisir_kpi(type_kpi):
        messages.error(request, "Vous n'avez pas la permission d'ajouter ce type de donnée.")
        return redirect('liste_kpi', type_kpi=type_kpi)
    if not profil.agence:
        messages.error(request, "Aucune agence associée à votre profil.")
        return redirect('dashboard')

    if request.method == 'POST':
        form = form_class(request.POST)
        if form.is_valid():
            obj = form.save(commit=False)
            obj.agence = profil.agence
            try:
                obj.save()
            except Exception:
                messages.error(
                    request,
                    "Une donnée identique existe déjà pour cette agence, ce type et cette année."
                )
                return render(request, 'form_kpi.html', {
                    'form': form, 'type_kpi': type_kpi, 'titre': f"Ajouter — {label}"
                })
            _vider_cache_analyses()
            messages.success(request, f"{label} : élément ajouté avec succès.")
            return redirect('liste_kpi', type_kpi=type_kpi)
    else:
        form = form_class()

    return render(request, 'form_kpi.html', {
        'form': form, 'type_kpi': type_kpi, 'titre': f"Ajouter — {label}"
    })


@login_required
def modifier_kpi(request, type_kpi, pk):
    profil = request.user.profil
    model, form_class, label = _get_type_kpi(type_kpi)

    if not profil.peut_saisir_kpi(type_kpi):
        messages.error(request, "Vous n'avez pas la permission de modifier ce type de donnée.")
        return redirect('liste_kpi', type_kpi=type_kpi)

    obj = get_object_or_404(model, pk=pk, agence=profil.agence)

    if request.method == 'POST':
        form = form_class(request.POST, instance=obj)
        if form.is_valid():
            form.save()
            _vider_cache_analyses()
            messages.success(request, f"{label} : élément modifié avec succès.")
            return redirect('liste_kpi', type_kpi=type_kpi)
    else:
        form = form_class(instance=obj)

    return render(request, 'form_kpi.html', {
        'form': form, 'type_kpi': type_kpi, 'titre': f"Modifier — {label}"
    })


@login_required
def supprimer_kpi(request, type_kpi, pk):
    profil = request.user.profil
    model, form_class, label = _get_type_kpi(type_kpi)

    if not profil.peut_supprimer_kpi():
        messages.error(request, "Seul le chef d'agence peut supprimer une donnée.")
        return redirect('liste_kpi', type_kpi=type_kpi)

    obj = get_object_or_404(model, pk=pk, agence=profil.agence)
    obj.delete()
    _vider_cache_analyses()
    messages.success(request, f"{label} : élément supprimé.")
    return redirect('liste_kpi', type_kpi=type_kpi)


# ============================================================
# ANALYSE DU RÉSEAU — réservé au super_admin
# ============================================================

@login_required
def analyse_reseau(request):
    profil = request.user.profil

    if not profil.est_super_admin():
        messages.error(request, "Cette analyse porte sur l'ensemble du réseau et est réservée au siège.")
        return redirect('dashboard')

    annees_dispo = _annees_disponibles()
    try:
        annee = int(request.GET.get('annee') or _derniere_annee_disponible())
    except ValueError:
        annee = _derniere_annee_disponible()

    try:
        n_profils = int(request.GET.get('profils', 3))
    except ValueError:
        n_profils = 3
    n_profils = max(2, min(n_profils, 5))

    # Le calcul dure une trentaine de secondes, du fait des permutations
    # du test d'anomalies. Le résultat est conservé jusqu'à la prochaine
    # modification des données.
    cle = f"analyse_reseau_{annee}_{n_profils}"
    resultat = cache.get(cle)
    if resultat is None:
        resultat = analyser_reseau(annee, n_profils)
        cache.set(cle, resultat, DUREE_CACHE)

    context = {
        'profil': profil,
        'annee': annee,
        'annees_dispo': annees_dispo,
        'n_profils': n_profils,
        'resultat': resultat,
    }
    return render(request, 'analyse_reseau.html', context)


# ============================================================
# PERFORMANCE DU RÉSEAU — réservé au super_admin
# ============================================================

@login_required
def performance_reseau(request):
    profil = request.user.profil

    if not profil.est_super_admin():
        messages.error(request, "Cette analyse porte sur l'ensemble du réseau et est réservée au siège.")
        return redirect('dashboard')

    annees_dispo = _annees_disponibles()
    try:
        annee = int(request.GET.get('annee') or _derniere_annee_disponible())
    except ValueError:
        annee = _derniere_annee_disponible()

    cible = request.GET.get('cible', 'comptes')
    if cible not in dict(CIBLES):
        cible = 'comptes'

    try:
        budget = int(request.GET.get('budget', 6))
    except ValueError:
        budget = 6
    budget = max(1, min(budget, 30))

    cle = f"performance_{annee}_{cible}"
    performance = cache.get(cle)
    if performance is None:
        performance = analyser_performance(annee, cible)
        if performance:
            performance = ajuster_performance(performance)
        cache.set(cle, performance, DUREE_CACHE)

    # L'allocation dépend du budget, modifiable à la volée : son calcul
    # est immédiat et n'est donc pas mis en cache.
    allocation = optimiser(performance, budget) if performance else None

    context = {
        'profil': profil,
        'annee': annee,
        'annees_dispo': annees_dispo,
        'cibles': CIBLES,
        'cible': cible,
        'budget': budget,
        'performance': performance,
        'allocation': allocation,
        'exp_performance': expliquer_performance(performance),
        'exp_bayesien': expliquer_bayesien(performance),
        'exp_conforme': expliquer_conforme(performance),
        'exp_allocation': expliquer_allocation(allocation),
    }
    return render(request, 'performance_reseau.html', context)


# ============================================================
# COMMUNAUTÉS D'AGENCES — analyse par graphe, super_admin
# ============================================================

@login_required
def communautes_reseau(request):
    profil = request.user.profil

    if not profil.est_super_admin():
        messages.error(request, "Cette analyse porte sur l'ensemble du réseau et est réservée au siège.")
        return redirect('dashboard')

    annees_dispo = _annees_disponibles()
    try:
        annee = int(request.GET.get('annee') or _derniere_annee_disponible())
    except ValueError:
        annee = _derniere_annee_disponible()

    cle = f"communautes_{annee}"
    resultat = cache.get(cle)
    if resultat is None:
        noms_agences, noms_features, matrice = construire_matrice(annee)
        if len(noms_agences) >= 6:
            _, donnees = preparer_donnees(matrice)
            resultat = analyser_communautes(donnees, noms_agences, noms_features, matrice)
        cache.set(cle, resultat, DUREE_CACHE)

    context = {
        'profil': profil,
        'annee': annee,
        'annees_dispo': annees_dispo,
        'resultat': resultat,
        'explications': expliquer_communautes(resultat),
    }
    return render(request, 'communautes_reseau.html', context)