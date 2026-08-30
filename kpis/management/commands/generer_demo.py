"""
Génère un jeu de données de démonstration sur plusieurs exercices.

Ces fichiers ne décrivent aucune activité réelle. Ils servent à valider
le pipeline complet (import, agrégation, modèles) sur des données dont
la structure est connue et paramétrable.

Modèle génératif :

  - Chaque agence possède deux caractéristiques stables dans le temps :
    une taille et une orientation commerciale (particuliers ou entreprises).
  - Les volumes de chaque produit dérivent de ces deux facteurs.
  - Le volume de crédits est construit à partir du volume des autres
    activités, avec un bruit calibré pour atteindre le coefficient de
    détermination visé : R² = var(signal) / (var(signal) + var(bruit)).
  - D'un exercice à l'autre, les volumes croissent selon un taux propre à
    chaque agence, de sorte qu'une agence performante le reste.

Les fichiers produits portent le préfixe DEMO.
"""

import os

import numpy as np
from openpyxl import Workbook

from django.core.management.base import BaseCommand, CommandError

from kpis.models import (
    Agence, CategorieCompte, TypeCarte, TypeCreditGestion, TypePlacement,
)


FAMILLES_PP = ['CREVOIT', 'CREIMMO', 'CONSOM']


class Command(BaseCommand):
    help = "Génère un jeu de démonstration multi-années à corrélation contrôlée."

    def add_arguments(self, parser):
        parser.add_argument('--sortie', required=True,
                            help="Dossier de destination des fichiers.")
        parser.add_argument('--annees', nargs='+', type=int,
                            default=[2023, 2024, 2025],
                            help="Exercices à générer (défaut : 2023 2024 2025).")
        parser.add_argument('--r2', type=float, default=0.80,
                            help="Coefficient de détermination visé (défaut : 0.80).")
        parser.add_argument('--croissance', type=float, default=0.06,
                            help="Croissance annuelle moyenne (défaut : 0.06).")
        parser.add_argument('--graine', type=int, default=7,
                            help="Graine aléatoire, pour reproductibilité.")

    def handle(self, *args, **options):
        sortie = options['sortie']
        annees = sorted(options['annees'])
        r2_cible = float(np.clip(options['r2'], 0.05, 0.98))
        croissance = options['croissance']
        graine = options['graine']

        os.makedirs(sortie, exist_ok=True)

        agences = list(Agence.objects.all().order_by('code_agence'))
        if len(agences) < 10:
            raise CommandError("Agences absentes de la base. Lancez populate_ref.")

        categories = list(CategorieCompte.objects.all())
        types_carte = list(TypeCarte.objects.all())
        types_gestion = list(TypeCreditGestion.objects.all())
        types_placement = list(TypePlacement.objects.all())

        if not (categories and types_carte and types_gestion and types_placement):
            raise CommandError("Tables de référence vides. Lancez populate_ref.")

        n = len(agences)
        gen = np.random.default_rng(graine)

        # --- Caractéristiques permanentes de chaque agence ---
        # Tirées une seule fois : elles ne changent pas d'un exercice à l'autre,
        # ce qui garantit qu'une agence conserve son positionnement dans le temps.
        taille = gen.normal(0, 1, n)
        orientation = gen.normal(0, 1, n)
        taux_agence = gen.normal(croissance, croissance * 0.4, n)

        # Répartitions entre sous-catégories, également stables
        poids_comptes = gen.dirichlet(np.ones(len(categories)) * 4, n)
        poids_cartes = gen.dirichlet(np.ones(len(types_carte)) * 1.5, n)
        poids_pp = gen.dirichlet(np.ones(len(FAMILLES_PP)) * 5, n)
        poids_gestion = gen.dirichlet(np.ones(len(types_gestion)) * 2, n)
        poids_placement = gen.dirichlet(np.ones(len(types_placement)) * 3, n)

        # Résidu de crédit propre à chaque agence : c'est lui qui fixe le R².
        # Tiré une fois, il est conservé sur tous les exercices, de sorte
        # qu'une agence sur-performante le reste d'une année sur l'autre.
        residu_credit = gen.normal(0, 1, n)

        annee_base = annees[0]
        recap = []

        for annee in annees:
            facteur = np.exp(taux_agence * (annee - annee_base))

            log_comptes = 5.6 + 0.75 * taille - 0.20 * orientation + gen.normal(0, 0.10, n)
            log_cartes = 5.8 + 0.80 * taille - 0.15 * orientation + gen.normal(0, 0.10, n)
            log_placem = 4.0 + 0.55 * taille + 0.45 * orientation + gen.normal(0, 0.12, n)

            comptes = np.exp(log_comptes) * facteur
            cartes = np.exp(log_cartes) * facteur
            placem = np.exp(log_placem) * facteur

            volume_autres = comptes + cartes + placem
            log_autres = np.log1p(volume_autres)

            # --- Crédits : signal calibré sur le R² visé ---
            signal = 0.95 * (log_autres - log_autres.mean()) + 0.30 * orientation
            variance_signal = float(np.var(signal))
            ecart_bruit = float(np.sqrt(variance_signal * (1.0 - r2_cible) / r2_cible))

            log_credits = 4.4 + signal + residu_credit * ecart_bruit
            credits_total = np.maximum(np.round(np.exp(log_credits)), 1)

            r = float(np.corrcoef(log_credits, log_autres)[0, 1])
            recap.append((annee, r))

            # Répartition entre crédits particuliers et crédits de gestion,
            # d'autant plus orientée gestion que l'agence sert des entreprises.
            part_gestion = 1.0 / (1.0 + np.exp(-(orientation * 0.8 - 0.9)))
            credits_gestion = np.maximum(np.round(credits_total * part_gestion), 1)
            credits_pp = np.maximum(credits_total - credits_gestion, 1)

            comptes = np.maximum(np.round(comptes), 1)
            cartes = np.maximum(np.round(cartes), 1)
            placem = np.maximum(np.round(placem), 1)

            fichiers = []

            # 1. Comptes clients
            lignes = []
            for i, ag in enumerate(agences):
                for k, cat in enumerate(categories):
                    v = int(round(comptes[i] * poids_comptes[i, k]))
                    if v > 0:
                        lignes.append([ag.code_agence, ag.nom, cat.code, cat.libelle, v])
            fichiers.append((
                f"DEMO_COMPTES_CLIENTS_{annee}.xlsx",
                ['COD_UG', 'LIB_UG', 'COD_CAT_CPT', 'LIB_CAT_CPT', f'NOMBRE EN {annee}'],
                lignes,
            ))

            # 2. Cartes
            lignes = []
            for i, ag in enumerate(agences):
                for k, tc in enumerate(types_carte):
                    v = int(round(cartes[i] * poids_cartes[i, k]))
                    if v > 0:
                        lignes.append([ag.code_agence, ag.nom, tc.code, tc.libelle, v])
            fichiers.append((
                f"DEMO_CARTES_{annee}.xlsx",
                ['COD_UG', 'LIB_UG', 'COD_NAT_CAR', 'LIB_NAT_CAR', f'NOMBRE EN {annee}'],
                lignes,
            ))

            # 3. Crédits aux particuliers
            lignes = []
            for i, ag in enumerate(agences):
                for k, fam in enumerate(FAMILLES_PP):
                    v = int(round(credits_pp[i] * poids_pp[i, k]))
                    if v > 0:
                        lignes.append([ag.code_agence, ag.nom, fam, v])
            fichiers.append((
                f"DEMO_CREDITS_PP_{annee}.xlsx",
                ['COD_UG', 'LIB_UG', 'FAM_COMPTA', f'NOMBRE EN {annee}'],
                lignes,
            ))

            # 4. Crédits de gestion
            lignes = []
            for i, ag in enumerate(agences):
                for k, tg in enumerate(types_gestion):
                    v = int(round(credits_gestion[i] * poids_gestion[i, k]))
                    if v > 0:
                        lignes.append([ag.code_agence, ag.nom, tg.code, tg.libelle, v])
            fichiers.append((
                f"DEMO_CREDITS_GESTION_{annee}.xlsx",
                ['COD_UG', 'LIB_UG', 'COD_PRO', 'LIB_PRO', f'NOMBRE EN {annee}'],
                lignes,
            ))

            # 5. Placements
            lignes = []
            for i, ag in enumerate(agences):
                for k, tp in enumerate(types_placement):
                    v = int(round(placem[i] * poids_placement[i, k]))
                    if v > 0:
                        lignes.append([ag.code_agence, ag.nom, tp.code, v])
            fichiers.append((
                f"DEMO_PLACEMENTS_{annee}.xlsx",
                ['COD_UG', 'LIB_UG', 'TYP_PLACEMENT', f'NOMBRE EN {annee}'],
                lignes,
            ))

            for nom_fichier, entetes, lignes in fichiers:
                classeur = Workbook()
                feuille = classeur.active
                feuille.title = "Données"
                feuille.append(entetes)
                for l in lignes:
                    feuille.append(l)
                classeur.save(os.path.join(sortie, nom_fichier))

            self.stdout.write(
                f"  {annee} — 5 fichiers écrits, corrélation crédits/activité "
                f"r = {r:.3f} (R² ≈ {r ** 2:.3f})"
            )

        self.stdout.write(self.style.SUCCESS(f"\nFichiers écrits dans {sortie}"))
        self.stdout.write("Corrélations obtenues par exercice :")
        for annee, r in recap:
            self.stdout.write(f"  {annee} : r = {r:.3f}  →  R² ≈ {r ** 2:.3f}")
        self.stdout.write(self.style.WARNING(
            "\nJeu de DÉMONSTRATION. Ces données ne décrivent aucune activité "
            "réelle et ne doivent pas être présentées comme telles."
        ))