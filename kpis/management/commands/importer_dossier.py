"""
Importe en une fois tous les fichiers Excel d'un dossier.

Le type de données et l'année sont déduits du nom du fichier.
Les règles anti-doublon habituelles s'appliquent : une agence déjà
renseignée pour l'année concernée est ignorée.
"""

import os
import re

from django.core.management.base import BaseCommand, CommandError

from kpis.excel_import import importer_fichier


def _type_fichier(nom):
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
    return None


def _annee(nom):
    trouve = re.search(r'(20\d{2})', nom)
    return int(trouve.group(1)) if trouve else None


class Command(BaseCommand):
    help = "Importe tous les fichiers Excel d'un dossier (type et année déduits du nom)."

    def add_arguments(self, parser):
        parser.add_argument('--dossier', required=True)

    def handle(self, *args, **options):
        dossier = options['dossier']
        if not os.path.isdir(dossier):
            raise CommandError(f"Dossier introuvable : {dossier}")

        fichiers = sorted(
            f for f in os.listdir(dossier)
            if f.lower().endswith(('.xls', '.xlsx')) and not f.startswith('~$')
        )
        if not fichiers:
            raise CommandError("Aucun fichier Excel dans ce dossier.")

        total = 0
        for nom in fichiers:
            type_fichier = _type_fichier(nom)
            annee = _annee(nom)

            if not type_fichier or not annee:
                self.stdout.write(self.style.WARNING(
                    f"  {nom} ignoré (type ou année indéterminable)."
                ))
                continue

            chemin = os.path.join(dossier, nom)
            with open(chemin, 'rb') as f:
                rapport = importer_fichier(f, nom, type_fichier, annee)

            total += rapport['importes']
            self.stdout.write(
                f"  {nom} → {rapport['importes']} lignes, "
                f"{len(rapport['agences_ok'])} agences"
                + (f", {len(rapport['agences_ignorees'])} ignorées"
                   if rapport['agences_ignorees'] else "")
            )
            for e in rapport['erreurs'][:3]:
                self.stdout.write(self.style.WARNING(f"     {e}"))

        self.stdout.write(self.style.SUCCESS(f"\n{total} lignes importées au total."))