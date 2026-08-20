from django.db import models
from django.contrib.auth.models import User


class Agence(models.Model):
    nom = models.CharField(max_length=100)
    code_agence = models.CharField(max_length=10, unique=True)  # ex: "000", "001"...

    class Meta:
        ordering = ['code_agence']

    def __str__(self):
        return f"{self.code_agence} - {self.nom}"


class Profil(models.Model):
    ROLE_CHOICES = [
        ('super_admin', 'Super Admin (Siège)'),
        ('chef_agence', 'Chef d\'Agence'),
        ('employe', 'Employé'),
    ]

    user = models.OneToOneField(User, on_delete=models.CASCADE)
    agence = models.ForeignKey(Agence, on_delete=models.SET_NULL, null=True, blank=True)
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default='employe')

    # Permissions fines, utiles uniquement si role == 'employe'.
    # Chacune autorise Créer + Modifier. La suppression n'est JAMAIS
    # autorisée à un employé, quelle que soit la valeur de ces champs
    # (contrôlé dans les vues, pas ici).
    peut_gerer_comptes = models.BooleanField(default=False)
    peut_gerer_credits = models.BooleanField(default=False)
    peut_gerer_tpe = models.BooleanField(default=False)
    peut_gerer_cartes = models.BooleanField(default=False)
    peut_gerer_credoc = models.BooleanField(default=False)
    peut_gerer_placements = models.BooleanField(default=False)

    def __str__(self):
        return f"{self.user.username} - {self.get_role_display()}"

    def est_super_admin(self):
        return self.role == 'super_admin'

    def est_chef_agence(self):
        return self.role == 'chef_agence'

    def est_employe(self):
        return self.role == 'employe'

    def peut_gerer_employes(self):
        return self.role in ('super_admin', 'chef_agence')

    def peut_supprimer_kpi(self):
        return self.role in ('super_admin', 'chef_agence')

    def peut_saisir_kpi(self, type_kpi):
        """
        type_kpi doit être l'un de :
        'compte', 'credit', 'tpe', 'carte', 'credoc', 'placement'.
        """
        if self.role in ('super_admin', 'chef_agence'):
            return True
        mapping = {
            'compte': self.peut_gerer_comptes,
            'credit': self.peut_gerer_credits,
            'tpe': self.peut_gerer_tpe,
            'carte': self.peut_gerer_cartes,
            'credoc': self.peut_gerer_credoc,
            'placement': self.peut_gerer_placements,
        }
        return mapping.get(type_kpi, False)

    def peut_importer_excel(self):
        """Import Excel : chef_agence (limité à sa propre agence) et super_admin (import global)."""
        return self.role in ('super_admin', 'chef_agence')

    def peut_exporter_excel(self):
        return self.role == 'super_admin'


# ============================================================
# TABLES DE RÉFÉRENCE
# Gérables via /admin/ par le super_admin uniquement.
# Permettent d'ajouter un nouveau type (ex: nouvelle carte) sans
# toucher au code.
# ============================================================

class CategorieCompte(models.Model):
    code = models.CharField(max_length=20, unique=True)     # ex: "25111000"
    libelle = models.CharField(max_length=100)               # ex: "PP Comptes chèques"

    class Meta:
        ordering = ['libelle']
        verbose_name = "Catégorie de compte"
        verbose_name_plural = "Catégories de compte"

    def __str__(self):
        return self.libelle


class TypeCreditGestion(models.Model):
    code = models.CharField(max_length=20, unique=True)      # ex: "14012"
    libelle = models.CharField(max_length=150)                # ex: "Avance sur créances"

    class Meta:
        ordering = ['libelle']
        verbose_name = "Sous-type de crédit gestion"
        verbose_name_plural = "Sous-types de crédit gestion"

    def __str__(self):
        return self.libelle


class TypeCarte(models.Model):
    code = models.CharField(max_length=20, unique=True)       # ex: "VGPNA"
    libelle = models.CharField(max_length=150)                 # ex: "Visa Gold Premier National Affaire"

    class Meta:
        ordering = ['libelle']
        verbose_name = "Type de carte"
        verbose_name_plural = "Types de carte"

    def __str__(self):
        return self.libelle


class TypePlacement(models.Model):
    code = models.CharField(max_length=20, unique=True)        # ex: "PRECOMPTE"
    libelle = models.CharField(max_length=100)

    class Meta:
        ordering = ['libelle']
        verbose_name = "Type de placement"
        verbose_name_plural = "Types de placement"

    def __str__(self):
        return self.libelle


# ============================================================
# MODÈLES DE DONNÉES (un enregistrement = un total pour
# une agence + un type + une année)
# ============================================================

class CompteClient(models.Model):
    agence = models.ForeignKey(Agence, on_delete=models.CASCADE)
    categorie = models.ForeignKey(CategorieCompte, on_delete=models.PROTECT)
    annee = models.PositiveIntegerField()
    nombre = models.PositiveIntegerField(default=0)
    montant = models.DecimalField(max_digits=15, decimal_places=3, null=True, blank=True)

    class Meta:
        unique_together = ('agence', 'categorie', 'annee')
        verbose_name = "Compte client"
        verbose_name_plural = "Comptes clients"

    def __str__(self):
        return f"{self.agence} - {self.categorie} - {self.annee} : {self.nombre}"


class Credit(models.Model):
    CATEGORIE_CHOICES = [
        ('voiture', 'Crédit Voiture'),
        ('mobilier', 'Crédit Mobilier'),
        ('consommation', 'Crédit Consommation'),
        ('gestion', 'Crédit Gestion'),
        ('morale_non_terme', 'Crédit Morale Non Terme'),
        ('engagement_signature', 'Engagement par Signature'),
    ]

    agence = models.ForeignKey(Agence, on_delete=models.CASCADE)
    categorie = models.CharField(max_length=30, choices=CATEGORIE_CHOICES)
    # Rempli uniquement quand categorie == 'gestion' ; vide sinon.
    sous_type = models.ForeignKey(
        TypeCreditGestion, on_delete=models.PROTECT, null=True, blank=True
    )
    annee = models.PositiveIntegerField()
    nombre = models.PositiveIntegerField(default=0)
    montant = models.DecimalField(max_digits=15, decimal_places=3, null=True, blank=True)

    class Meta:
        unique_together = ('agence', 'categorie', 'sous_type', 'annee')
        verbose_name = "Crédit"
        verbose_name_plural = "Crédits"

    def __str__(self):
        detail = f" ({self.sous_type})" if self.sous_type else ""
        return f"{self.agence} - {self.get_categorie_display()}{detail} - {self.annee} : {self.nombre}"


class VenteCarte(models.Model):
    agence = models.ForeignKey(Agence, on_delete=models.CASCADE)
    type_carte = models.ForeignKey(TypeCarte, on_delete=models.PROTECT)
    annee = models.PositiveIntegerField()
    nombre = models.PositiveIntegerField(default=0)
    montant = models.DecimalField(max_digits=15, decimal_places=3, null=True, blank=True)

    class Meta:
        unique_together = ('agence', 'type_carte', 'annee')
        verbose_name = "Vente carte"
        verbose_name_plural = "Ventes cartes"

    def __str__(self):
        return f"{self.agence} - {self.type_carte} - {self.annee} : {self.nombre}"


class VenteTPE(models.Model):
    agence = models.ForeignKey(Agence, on_delete=models.CASCADE)
    annee = models.PositiveIntegerField()
    nombre = models.PositiveIntegerField(default=0)
    montant = models.DecimalField(max_digits=15, decimal_places=3, null=True, blank=True)

    class Meta:
        unique_together = ('agence', 'annee')
        verbose_name = "Vente TPE"
        verbose_name_plural = "Ventes TPE"

    def __str__(self):
        return f"{self.agence} - TPE - {self.annee} : {self.nombre}"


class Placement(models.Model):
    agence = models.ForeignKey(Agence, on_delete=models.CASCADE)
    type_placement = models.ForeignKey(TypePlacement, on_delete=models.PROTECT)
    annee = models.PositiveIntegerField()
    nombre = models.PositiveIntegerField(default=0)
    montant = models.DecimalField(max_digits=15, decimal_places=3, null=True, blank=True)

    class Meta:
        unique_together = ('agence', 'type_placement', 'annee')
        verbose_name = "Placement"
        verbose_name_plural = "Placements"

    def __str__(self):
        return f"{self.agence} - {self.type_placement} - {self.annee} : {self.nombre}"


class Credoc(models.Model):
    """
    Modèle conservé tel quel en attendant le fichier réel.
    Reste basé sur une date précise (pas encore uniformisé sur 'annee').
    """
    SENS_CHOICES = [
        ('import', 'Import'),
        ('export', 'Export'),
    ]
    agence = models.ForeignKey(Agence, on_delete=models.CASCADE)
    sens = models.CharField(max_length=10, choices=SENS_CHOICES)
    montant = models.DecimalField(max_digits=15, decimal_places=3)
    date = models.DateField()

    def __str__(self):
        return f"{self.agence} - {self.sens} - {self.montant}"