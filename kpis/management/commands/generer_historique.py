"""
Reconstitue des fichiers Excel d'historique calibrés sur les taux
sectoriels réels du secteur bancaire tunisien.

Les fichiers produits sont SYNTHÉTIQUES et doivent être présentés
comme tels dans tout rapport.
"""

import os
import re
import hashlib

import numpy as np
from openpyxl import Workbook

from django.core.management.base import BaseCommand, CommandError

from kpis.ml.generation import (
    SOURCES, profiler_agences, segment_du_produit, taux_agence,
)


def _lire_fichier(chemin):
    if chemin.lower().endswith('.xls'):
        import xlrd
        classeur = xlrd.open_workbook(chemin)
        feuille = classeur.sheet_by_index(0)
        entetes = [str(feuille.cell_value(0, c)).strip() for c in range(feuille.ncols)]
        lignes = [
            [feuille.cell_value(r, c) for c in range(feuille.ncols)]
            for r in range(1, feuille.nrows)
        ]
        return entetes, lignes

    if chemin.lower().endswith('.xlsx'):
        import openpyxl
        classeur = openpyxl.load_workbook(chemin, data_only=True)
        feuille = classeur.active
        donnees = list(feuille.values)
        entetes = [str(h).strip() if h is not None else '' for h in donnees[0]]
        lignes = [list(l) for l in donnees[1:]]
        return entetes, lignes

    raise CommandError(f"Format non géré : {chemin}")


def _index(entetes, nom):
    for i, e in enumerate(entetes):
        if e.upper() == nom.upper():
            return i
    return None


def _index_nombre(entetes):
    for i, e in enumerate(entetes):
        if e.upper().startswith('NOMBRE'):
            return i
    return None


def _texte(valeur):
    if valeur is None:
        return ''
    if isinstance(valeur, float) and valeur.is_integer():
        return str(int(valeur))
    return str(valeur).strip()


def _graine(*elements):
    empreinte = hashlib.md5("|".join(str(e) for e in elements).encode()).hexdigest()
    return int(empreinte[:8], 16)


def _type_fichier(nom):
    """Devine le type de données d'après le nom du fichier."""
    n = nom.upper()
    if 'COMPTE' in n:
        return 'comptes'
    if 'CARTE' in n:
        return 'cartes'
    if 'GESTION' in n:
        return 'credits_gestion'
    if 'CMLT' in n:
        return 'cmlt'
    if 'PLACEMENT' in n:
        return 'placements'
    if 'PP' in n or 'PARTICULIER' in n:
        return 'credits_pp'
    if 'TPE' in n:
        return 'tpe'
    return 'comptes'


class Command(BaseCommand):
    help = (
        "Reconstitue des fichiers d'historique calibrés sur les taux "
        "sectoriels réels (BCT) et sur le profil de chaque agence."
    )

    def add_arguments(self, parser):
        parser.add_argument('--source', required=True,
                            help="Dossier des fichiers de l'année de référence.")
        parser.add_argument('--sortie', default=None,
                            help="Dossier de destination (défaut : <source>/historique).")
        parser.add_argument('--reference', type=int, default=2025,
                            help="Année des fichiers source (défaut : 2025).")
        parser.add_argument('--annees', nargs='+', type=int, default=[2023, 2024],
                            help="Années à reconstituer (défaut : 2023 2024).")
        parser.add_argument('--bruit', type=float, default=0.06,
                            help="Amplitude du résidu par ligne (défaut : 0.06).")

    def handle(self, *args, **options):
        source = options['source']
        sortie = options['sortie'] or os.path.join(source, 'historique')
        reference = options['reference']
        annees = sorted(options['annees'])
        bruit = options['bruit']

        if not os.path.isdir(source):
            raise CommandError(f"Dossier introuvable : {source}")
        os.makedirs(sortie, exist_ok=True)

        # --- Étape 1 : profilage des agences sur les données réelles ---
        self.stdout.write("Classification des agences sur les données réelles...")
        resultat = profiler_agences(reference)

        if resultat is None:
            self.stdout.write(self.style.WARNING(
                f"  Aucune donnée en base pour {reference} : les taux sectoriels "
                "seront appliqués uniformément, sans modulation par agence."
            ))
            profils = {}
        else:
            profils = resultat['profils']
            self.stdout.write(
                f"  {resultat['n_agences']} agences réparties en {resultat['k']} profils "
                f"(silhouette {resultat['silhouette']})."
            )

        # --- Étape 2 : génération ---
        fichiers = [
            f for f in os.listdir(source)
            if f.lower().endswith(('.xls', '.xlsx'))
            and str(reference) in f
            and not f.startswith('~$')
        ]
        if not fichiers:
            raise CommandError(f"Aucun fichier de {reference} trouvé dans {source}.")

        self.stdout.write(f"\n{len(fichiers)} fichier(s) source détecté(s).")
        total = 0

        for nom_fichier in sorted(fichiers):
            entetes, lignes = _lire_fichier(os.path.join(source, nom_fichier))

            i_nombre = _index_nombre(entetes)
            i_ug = _index(entetes, 'COD_UG')
            if i_nombre is None or i_ug is None:
                self.stdout.write(self.style.WARNING(
                    f"  {nom_fichier} ignoré (colonnes attendues absentes)."
                ))
                continue

            type_fichier = _type_fichier(nom_fichier)
            segment = segment_du_produit(type_fichier)

            # Colonne de sous-type, si elle existe
            i_type = None
            for i, e in enumerate(entetes):
                if i not in (i_nombre, i_ug) and e.upper().startswith(('COD_', 'FAM_', 'TYP_')):
                    if 'UG' not in e.upper():
                        i_type = i

            for annee in annees:
                if annee >= reference:
                    continue

                lignes_sortie = []

                for ligne in lignes:
                    code_ug = _texte(ligne[i_ug]).zfill(3)
                    code_type = _texte(ligne[i_type]) if i_type is not None else 'X'

                    try:
                        valeur = float(ligne[i_nombre]) if ligne[i_nombre] not in (None, '') else 0.0
                    except (TypeError, ValueError):
                        valeur = 0.0

                    # Remontée année par année, avec le taux propre à chaque exercice
                    valeur_passee = valeur
                    for an in range(reference, annee, -1):
                        taux = taux_agence(code_ug, segment, an, profils)
                        valeur_passee = valeur_passee / (1 + max(taux, -0.2))

                    # Résidu idiosyncratique, reproductible
                    generateur = np.random.default_rng(
                        _graine(code_ug, code_type, segment, annee)
                    )
                    facteur = float(generateur.normal(1.0, bruit))
                    valeur_passee *= max(facteur, 0.7)

                    nouvelle = list(ligne)
                    nouvelle[i_nombre] = max(0, int(round(valeur_passee)))
                    lignes_sortie.append(nouvelle)

                entetes_sortie = list(entetes)
                entetes_sortie[i_nombre] = (
                    re.sub(r'\d{4}', str(annee), entetes[i_nombre])
                    if re.search(r'\d{4}', entetes[i_nombre])
                    else f"NOMBRE EN {annee}"
                )

                nom_sortie = (
                    nom_fichier.replace(str(reference), str(annee))
                    if str(reference) in nom_fichier
                    else f"{annee}_{nom_fichier}"
                )
                nom_sortie = os.path.splitext(nom_sortie)[0] + '.xlsx'

                classeur = Workbook()
                feuille = classeur.active
                feuille.title = "Données"
                feuille.append(entetes_sortie)
                for l in lignes_sortie:
                    feuille.append(l)
                classeur.save(os.path.join(sortie, nom_sortie))

                total += 1
                self.stdout.write(f"  {nom_sortie} — {len(lignes_sortie)} lignes ({segment})")

        self.stdout.write(self.style.SUCCESS(f"\n{total} fichier(s) écrit(s) dans {sortie}"))
        self.stdout.write("\nCalibrage fondé sur :")
        for s in SOURCES:
            self.stdout.write(f"  · {s}")
        self.stdout.write(self.style.WARNING(
            "\nCes fichiers reconstituent un passé plausible à partir de taux "
            "sectoriels publiés. Ils restent synthétiques et doivent être "
            "présentés comme tels."
        ))