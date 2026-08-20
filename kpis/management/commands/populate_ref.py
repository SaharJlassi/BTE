from django.core.management.base import BaseCommand
from kpis.models import Agence, CategorieCompte, TypeCreditGestion, TypeCarte, TypePlacement


AGENCES = {
    "000": "SIEGE", "001": "LES BERGES DU LAC", "002": "ENNASR", "003": "MEGRINE",
    "004": "SOUSSE", "005": "ARIANA", "006": "SFAX", "007": "BARDO", "008": "BIZERTE",
    "009": "NABEUL", "010": "MSAKEN", "011": "BEN AROUS", "012": "MARSA",
    "013": "SOUSSE 2", "014": "GABES", "015": "MONASTIR", "016": "BEJA",
    "017": "AL AOUINA", "018": "BOUMHAL", "019": "EL MENZAH 4", "020": "AL HRAIRIA",
    "021": "MONTPLAISIR", "022": "KAIROUAN", "023": "SFAX 2", "024": "MOKNINE",
    "025": "LAC2", "026": "KRAM", "027": "DJERBA", "028": "BTE MANOUBA",
    "029": "JENDOUBA", "030": "PASTEUR", "031": "NEO -BTE",
}

CATEGORIES_COMPTE = {
    "25111000": "PP Comptes chèques",
    "25122000": "PA Comptes courants de la clientele",
    "25123000": "PM Comptes courants de la clientele",
}

TYPES_CREDIT_GESTION = {
    "14002": "Crédit de campagne",
    "14005": "Préfinancement de marché",
    "14006": "Financement obligations",
    "14007": "Découvert mobilisé",
    "14008": "Préfinancement d'exportation en TND",
    "14009": "Crédit de financement de stock",
    "14011": "MCNE en TND",
    "14012": "Avance sur creances administratives",
    "14013": "Avance sur Factures",
    "14016": "Crédit de dessaisissement",
}

TYPES_CARTE = {
    "BNPL": "BUY NOW PAY LATER",
    "CAT": "CARTE ALLOCATION TOURISTIQUE",
    "CTIA": "CARTE TECHNOLOGIQUE INTERNATIONAL AFFAIRE",
    "CTII": "CARTE TECHNOLOGIQUE INTERNATIONAL INDIV.",
    "EPARGNE": "VISA EPARGNE",
    "GREEN": "VISA EPARGNE GREEN",
    "MPIA": "MCD PLATINUM INTERNATIONAL AFFAIRE",
    "MPII": "MCD PLATINUM INTERNATIONAL INDIVIDUEL",
    "MPNA": "MCD PLATINUM NATIONAL AFFAIRE",
    "MPNI": "MCD PLATINUM NATIONAL INDIVIDUEL",
    "MVI": "MCD VIRTUELLE INTERNATIONAL",
    "VBSNI": "VISA BIN SPONSORSHIP NATIONAL INDIV",
    "VBSNIP": "VISA BIN SPONSORSHIP NATIONAL INDIV PHYSIQUE",
    "VCENA": "VISA CLASSIQUE ELECTRON NATIONAL AFFAIRE",
    "VCENI": "VISA CLASSIQUE ELECTRON NATIONAL INDIV.",
    "VGPIA": "VISA GOLD PREMIER INTERNATIONAL AFFAIRE",
    "VGPII": "VISA GOLD PREMIER INTERNATIONAL INDIV.",
    "VGPNA": "VISA GOLD PREMIER NATIONAL AFFAIRE",
    "VGPNI": "VISA GOLD PREMIER NATIONAL INDIV.",
    "VINFIA": "VISA INFINITE INTERNATIONAL AFFAIRE",
    "VINFII": "VISA INFINITE INTERNATIONAL INDIV",
    "VINFNA": "VISA INFINITE NATIONAL AFFAIRE",
    "VINFNI": "VISA INFINITE NATIONAL INDIV",
}

TYPES_PLACEMENT = {
    "PRECOMPTE": "Précompte",
    "PSOTCOMPTE": "Post-compte",
}


class Command(BaseCommand):
    help = "Peuple les 32 agences et les tables de référence (cartes, comptes, crédits gestion, placements) avec les vraies valeurs BTE."

    def handle(self, *args, **options):
        created_count = 0

        for code, nom in AGENCES.items():
            obj, created = Agence.objects.get_or_create(
                code_agence=code, defaults={"nom": nom}
            )
            created_count += int(created)
        self.stdout.write(self.style.SUCCESS(f"Agences : {len(AGENCES)} vérifiées."))

        for code, libelle in CATEGORIES_COMPTE.items():
            CategorieCompte.objects.get_or_create(code=code, defaults={"libelle": libelle})
        self.stdout.write(self.style.SUCCESS(f"Catégories de compte : {len(CATEGORIES_COMPTE)} vérifiées."))

        for code, libelle in TYPES_CREDIT_GESTION.items():
            TypeCreditGestion.objects.get_or_create(code=code, defaults={"libelle": libelle})
        self.stdout.write(self.style.SUCCESS(f"Sous-types crédit gestion : {len(TYPES_CREDIT_GESTION)} vérifiées."))

        for code, libelle in TYPES_CARTE.items():
            TypeCarte.objects.get_or_create(code=code, defaults={"libelle": libelle})
        self.stdout.write(self.style.SUCCESS(f"Types de carte : {len(TYPES_CARTE)} vérifiées."))

        for code, libelle in TYPES_PLACEMENT.items():
            TypePlacement.objects.get_or_create(code=code, defaults={"libelle": libelle})
        self.stdout.write(self.style.SUCCESS(f"Types de placement : {len(TYPES_PLACEMENT)} vérifiées."))

        self.stdout.write(self.style.SUCCESS("Peuplement terminé avec succès."))