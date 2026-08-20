"""
Moteur d'import des fichiers Excel BTE.

Chaque type de fichier a sa propre structure de colonnes :
  comptes          : COD_UG | LIB_UG | COD_CAT_CPT  | LIB_CAT_CPT | NOMBRE EN <annee>
  cartes           : COD_UG | LIB_UG | COD_NAT_CAR  | LIB_NAT_CAR | NOMBRE EN <annee>
  credits_pp       : COD_UG | LIB_UG | FAM_COMPTA   | NOMBRE EN <annee>
  credits_gestion  : COD_UG | LIB_UG | COD_PRO      | LIB_PRO     | NOMBRE EN <annee>
  cmlt             : COD_UG | LIB_UG | NOMBRE EN <annee>
  placements       : COD_UG | LIB_UG | TYP_PLACEMENT| NOMBRE EN <annee>
"""

from django.db import transaction

from .models import (
    Agence, CategorieCompte, TypeCreditGestion, TypeCarte, TypePlacement,
    CompteClient, Credit, VenteCarte, Placement,
)


# Types de fichiers proposés à l'utilisateur (valeur, libellé affiché)
TYPES_FICHIER = [
    ('comptes', 'Comptes clients'),
    ('cartes', 'Cartes'),
    ('credits_pp', 'Crédits aux particuliers (voiture / mobilier / consommation)'),
    ('credits_gestion', 'Crédits de gestion'),
    ('cmlt', 'CMLT (crédit morale non terme)'),
    ('placements', 'Placements'),
]

# Correspondance FAM_COMPTA -> catégorie du modèle Credit
FAM_COMPTA_VERS_CATEGORIE = {
    'CREVOIT': 'voiture',
    'CREIMMO': 'mobilier',
    'CONSOM': 'consommation',
}


# ------------------------------------------------------------------
# Lecture du fichier
# ------------------------------------------------------------------

def lire_lignes(fichier, nom_fichier):
    """
    Lit un fichier Excel (.xls ou .xlsx) et renvoie (entetes, lignes).
    entetes : liste de chaînes ; lignes : liste de listes de valeurs brutes.
    """
    nom = (nom_fichier or '').lower()

    if nom.endswith('.xls'):
        import xlrd
        classeur = xlrd.open_workbook(file_contents=fichier.read())
        feuille = classeur.sheet_by_index(0)
        if feuille.nrows < 2:
            return [], []
        entetes = [str(feuille.cell_value(0, c)).strip() for c in range(feuille.ncols)]
        lignes = [
            [feuille.cell_value(r, c) for c in range(feuille.ncols)]
            for r in range(1, feuille.nrows)
        ]
        return entetes, lignes

    if nom.endswith('.xlsx'):
        import openpyxl
        classeur = openpyxl.load_workbook(fichier, data_only=True)
        feuille = classeur.active
        donnees = list(feuille.values)
        if len(donnees) < 2:
            return [], []
        entetes = [str(h).strip() if h is not None else '' for h in donnees[0]]
        lignes = [list(l) for l in donnees[1:]]
        return entetes, lignes

    raise ValueError("Format non supporté. Utilisez un fichier .xls ou .xlsx.")


# ------------------------------------------------------------------
# Normalisation des valeurs
# ------------------------------------------------------------------

def _texte(valeur):
    """Convertit une cellule en texte propre (12.0 -> '12', None -> '')."""
    if valeur is None:
        return ''
    if isinstance(valeur, float) and valeur.is_integer():
        return str(int(valeur))
    return str(valeur).strip()


def _code_agence(valeur):
    """Le code agence fait 3 chiffres : 0 -> '000', '12' -> '012'."""
    code = _texte(valeur)
    return code.zfill(3) if code.isdigit() else code


def _nombre(valeur):
    """Convertit une cellule en entier positif ; 0 si vide ou illisible."""
    if valeur is None or valeur == '':
        return 0
    try:
        return max(0, int(float(valeur)))
    except (TypeError, ValueError):
        return 0


def _index_colonne(entetes, nom_exact):
    for i, e in enumerate(entetes):
        if e.upper() == nom_exact.upper():
            return i
    return None


def _index_colonne_nombre(entetes):
    """La colonne du compteur s'appelle 'NOMBRE EN 2025', 'NOMBRE EN 2024'... """
    for i, e in enumerate(entetes):
        if e.upper().startswith('NOMBRE'):
            return i
    return None


# ------------------------------------------------------------------
# Configuration par type de fichier
# ------------------------------------------------------------------

def _config(type_fichier):
    """
    Renvoie un dictionnaire décrivant comment traiter ce type de fichier :
      colonne_type : nom de la colonne contenant le code du sous-type (ou None)
      modele       : modèle Django cible
      filtre_existant : fonction(agence, annee) -> queryset des données déjà présentes
    """
    if type_fichier == 'comptes':
        return {
            'colonne_type': 'COD_CAT_CPT',
            'modele': CompteClient,
            'filtre_existant': lambda ag, an: CompteClient.objects.filter(agence=ag, annee=an),
        }
    if type_fichier == 'cartes':
        return {
            'colonne_type': 'COD_NAT_CAR',
            'modele': VenteCarte,
            'filtre_existant': lambda ag, an: VenteCarte.objects.filter(agence=ag, annee=an),
        }
    if type_fichier == 'placements':
        return {
            'colonne_type': 'TYP_PLACEMENT',
            'modele': Placement,
            'filtre_existant': lambda ag, an: Placement.objects.filter(agence=ag, annee=an),
        }
    if type_fichier == 'credits_pp':
        return {
            'colonne_type': 'FAM_COMPTA',
            'modele': Credit,
            'filtre_existant': lambda ag, an: Credit.objects.filter(
                agence=ag, annee=an, categorie__in=list(FAM_COMPTA_VERS_CATEGORIE.values())
            ),
        }
    if type_fichier == 'credits_gestion':
        return {
            'colonne_type': 'COD_PRO',
            'modele': Credit,
            'filtre_existant': lambda ag, an: Credit.objects.filter(
                agence=ag, annee=an, categorie='gestion'
            ),
        }
    if type_fichier == 'cmlt':
        return {
            'colonne_type': None,  # pas de sous-type : un seul total par agence
            'modele': Credit,
            'filtre_existant': lambda ag, an: Credit.objects.filter(
                agence=ag, annee=an, categorie='morale_non_terme'
            ),
        }
    raise ValueError("Type de fichier inconnu.")


def _tables_reference(type_fichier):
    """Charge en mémoire la table de référence associée : {code: objet}."""
    if type_fichier == 'comptes':
        return {r.code: r for r in CategorieCompte.objects.all()}
    if type_fichier == 'cartes':
        return {r.code: r for r in TypeCarte.objects.all()}
    if type_fichier == 'placements':
        return {r.code: r for r in TypePlacement.objects.all()}
    if type_fichier == 'credits_gestion':
        return {r.code: r for r in TypeCreditGestion.objects.all()}
    return {}


def _construire_objet(type_fichier, agence, annee, code_type, nombre, references):
    """
    Construit l'instance à enregistrer, ou lève ValueError si le code
    du sous-type est absent de la table de référence.
    """
    if type_fichier == 'comptes':
        ref = references.get(code_type)
        if ref is None:
            raise ValueError(f"catégorie de compte inconnue : {code_type}")
        return CompteClient(agence=agence, categorie=ref, annee=annee, nombre=nombre)

    if type_fichier == 'cartes':
        ref = references.get(code_type)
        if ref is None:
            raise ValueError(f"type de carte inconnu : {code_type}")
        return VenteCarte(agence=agence, type_carte=ref, annee=annee, nombre=nombre)

    if type_fichier == 'placements':
        ref = references.get(code_type)
        if ref is None:
            raise ValueError(f"type de placement inconnu : {code_type}")
        return Placement(agence=agence, type_placement=ref, annee=annee, nombre=nombre)

    if type_fichier == 'credits_pp':
        categorie = FAM_COMPTA_VERS_CATEGORIE.get(code_type.upper())
        if categorie is None:
            raise ValueError(f"famille de crédit inconnue : {code_type}")
        return Credit(agence=agence, categorie=categorie, sous_type=None,
                      annee=annee, nombre=nombre)

    if type_fichier == 'credits_gestion':
        ref = references.get(code_type)
        if ref is None:
            raise ValueError(f"sous-type de crédit gestion inconnu : {code_type}")
        return Credit(agence=agence, categorie='gestion', sous_type=ref,
                      annee=annee, nombre=nombre)

    if type_fichier == 'cmlt':
        return Credit(agence=agence, categorie='morale_non_terme', sous_type=None,
                      annee=annee, nombre=nombre)

    raise ValueError("Type de fichier inconnu.")


# ------------------------------------------------------------------
# Import principal
# ------------------------------------------------------------------

def importer_fichier(fichier, nom_fichier, type_fichier, annee, agence_limitee=None):
    """
    Importe un fichier Excel.

    agence_limitee : si fourni (cas du chef d'agence), seules les lignes de
    cette agence sont prises en compte, le reste du fichier est ignoré.

    Renvoie un rapport : nombre de lignes importées, agences traitées,
    agences ignorées pour cause de doublon, et erreurs éventuelles.
    """
    config = _config(type_fichier)
    references = _tables_reference(type_fichier)

    entetes, lignes = lire_lignes(fichier, nom_fichier)
    if not entetes:
        return {'importes': 0, 'agences_ok': [], 'agences_ignorees': [],
                'lignes_hors_agence': 0, 'erreurs': ["Le fichier est vide."]}

    i_ug = _index_colonne(entetes, 'COD_UG')
    i_nombre = _index_colonne_nombre(entetes)
    i_type = _index_colonne(entetes, config['colonne_type']) if config['colonne_type'] else None

    if i_ug is None:
        return {'importes': 0, 'agences_ok': [], 'agences_ignorees': [],
                'lignes_hors_agence': 0,
                'erreurs': ["Colonne COD_UG introuvable : ce fichier ne correspond pas au format attendu."]}
    if i_nombre is None:
        return {'importes': 0, 'agences_ok': [], 'agences_ignorees': [],
                'lignes_hors_agence': 0,
                'erreurs': ["Colonne 'NOMBRE EN ...' introuvable."]}
    if config['colonne_type'] and i_type is None:
        return {'importes': 0, 'agences_ok': [], 'agences_ignorees': [],
                'lignes_hors_agence': 0,
                'erreurs': [f"Colonne {config['colonne_type']} introuvable : "
                            f"le type de fichier sélectionné ne correspond pas au fichier envoyé."]}

    agences_par_code = {a.code_agence: a for a in Agence.objects.all()}

    # 1) Regrouper les lignes par agence, en cumulant les doublons éventuels
    #    du fichier lui-même (même agence + même sous-type sur deux lignes).
    donnees = {}          # code_agence -> {code_type: nombre}
    erreurs = []
    lignes_hors_agence = 0
    codes_inconnus = set()

    for ligne in lignes:
        code_ug = _code_agence(ligne[i_ug])
        if not code_ug:
            continue

        if agence_limitee is not None and code_ug != agence_limitee.code_agence:
            lignes_hors_agence += 1
            continue

        if code_ug not in agences_par_code:
            codes_inconnus.add(code_ug)
            continue

        code_type = _texte(ligne[i_type]) if i_type is not None else '__unique__'
        nombre = _nombre(ligne[i_nombre])

        donnees.setdefault(code_ug, {})
        donnees[code_ug][code_type] = donnees[code_ug].get(code_type, 0) + nombre

    for code in sorted(codes_inconnus):
        erreurs.append(f"Agence de code {code} absente de la base : lignes ignorées.")

    # 2) Traiter agence par agence : celles déjà renseignées sont ignorées,
    #    les autres sont importées normalement.
    importes = 0
    agences_ok = []
    agences_ignorees = []

    for code_ug in sorted(donnees.keys()):
        agence = agences_par_code[code_ug]

        if config['filtre_existant'](agence, annee).exists():
            agences_ignorees.append(str(agence))
            continue

        objets = []
        erreur_agence = False
        for code_type, nombre in donnees[code_ug].items():
            try:
                objets.append(
                    _construire_objet(type_fichier, agence, annee, code_type, nombre, references)
                )
            except ValueError as e:
                erreurs.append(f"{agence} : {e}")
                erreur_agence = True

        if erreur_agence or not objets:
            continue

        with transaction.atomic():
            config['modele'].objects.bulk_create(objets)
        importes += len(objets)
        agences_ok.append(str(agence))

    return {
        'importes': importes,
        'agences_ok': agences_ok,
        'agences_ignorees': agences_ignorees,
        'lignes_hors_agence': lignes_hors_agence,
        'erreurs': erreurs,
    }