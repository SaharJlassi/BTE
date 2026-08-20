from django.contrib import admin
from django.contrib.auth.models import User, Group
from django.contrib.auth.admin import UserAdmin
from .models import (
    Agence, Profil,
    CategorieCompte, TypeCreditGestion, TypeCarte, TypePlacement,
    CompteClient, Credit, VenteCarte, VenteTPE, Placement, Credoc,
)


def get_profil(request):
    try:
        return request.user.profil
    except Profil.DoesNotExist:
        return None


class AgenceRestrictedAdmin(admin.ModelAdmin):
    """
    Pour les modèles KPI liés à une agence (CompteClient, Credit, VenteCarte,
    VenteTPE, Placement, Credoc).

    - superuser Django ou profil super_admin : voit et modifie tout.
    - profil chef_agence : ne voit/modifie/supprime QUE les objets de sa
      propre agence ; le champ "Agence" du formulaire est verrouillé dessus.
    - profil employe (ou pas de profil) : aucun accès via /admin/.
    """

    def get_queryset(self, request):
        qs = super().get_queryset(request)
        if request.user.is_superuser:
            return qs
        profil = get_profil(request)
        if profil is None:
            return qs.none()
        if profil.role == 'super_admin':
            return qs
        if profil.role == 'chef_agence' and profil.agence:
            return qs.filter(agence=profil.agence)
        return qs.none()

    def formfield_for_foreignkey(self, db_field, request, **kwargs):
        if db_field.name == 'agence' and not request.user.is_superuser:
            profil = get_profil(request)
            if profil and profil.role == 'chef_agence' and profil.agence:
                kwargs['queryset'] = Agence.objects.filter(id=profil.agence_id)
        return super().formfield_for_foreignkey(db_field, request, **kwargs)

    def save_model(self, request, obj, form, change):
        if not request.user.is_superuser:
            profil = get_profil(request)
            if profil and profil.role == 'chef_agence' and profil.agence:
                obj.agence = profil.agence
        super().save_model(request, obj, form, change)

    def _acces_objet(self, request, obj):
        if request.user.is_superuser:
            return True
        profil = get_profil(request)
        if profil is None:
            return False
        if profil.role == 'super_admin':
            return True
        if profil.role == 'chef_agence':
            return obj.agence_id == profil.agence_id
        return False

    def has_module_permission(self, request):
        if request.user.is_superuser:
            return True
        profil = get_profil(request)
        return profil is not None and profil.role in ('super_admin', 'chef_agence')

    def has_view_permission(self, request, obj=None):
        if obj is None:
            return self.has_module_permission(request)
        return self._acces_objet(request, obj)

    def has_add_permission(self, request):
        return self.has_module_permission(request)

    def has_change_permission(self, request, obj=None):
        if obj is None:
            return self.has_module_permission(request)
        return self._acces_objet(request, obj)

    def has_delete_permission(self, request, obj=None):
        if obj is None:
            return self.has_module_permission(request)
        return self._acces_objet(request, obj)


class SuperAdminOnlyAdmin(admin.ModelAdmin):
    """
    Pour Agence, Profil, et les tables de référence (CategorieCompte,
    TypeCreditGestion, TypeCarte, TypePlacement) : uniquement visibles/
    modifiables par super_admin (ou superuser Django).
    """

    def has_module_permission(self, request):
        if request.user.is_superuser:
            return True
        profil = get_profil(request)
        return profil is not None and profil.role == 'super_admin'

    def has_view_permission(self, request, obj=None):
        return self.has_module_permission(request)

    def has_add_permission(self, request):
        return self.has_module_permission(request)

    def has_change_permission(self, request, obj=None):
        return self.has_module_permission(request)

    def has_delete_permission(self, request, obj=None):
        return self.has_module_permission(request)


# --- Agence & Profil ---

@admin.register(Agence)
class AgenceAdmin(SuperAdminOnlyAdmin):
    list_display = ('code_agence', 'nom')
    search_fields = ('nom', 'code_agence')


@admin.register(Profil)
class ProfilAdmin(SuperAdminOnlyAdmin):
    list_display = ('user', 'agence', 'role', 'compte_actif')
    list_filter = ('role', 'agence')

    def compte_actif(self, obj):
        return obj.user.is_active
    compte_actif.boolean = True
    compte_actif.short_description = "Actif"


# --- Tables de référence ---

@admin.register(CategorieCompte)
class CategorieCompteAdmin(SuperAdminOnlyAdmin):
    list_display = ('code', 'libelle')
    search_fields = ('code', 'libelle')


@admin.register(TypeCreditGestion)
class TypeCreditGestionAdmin(SuperAdminOnlyAdmin):
    list_display = ('code', 'libelle')
    search_fields = ('code', 'libelle')


@admin.register(TypeCarte)
class TypeCarteAdmin(SuperAdminOnlyAdmin):
    list_display = ('code', 'libelle')
    search_fields = ('code', 'libelle')


@admin.register(TypePlacement)
class TypePlacementAdmin(SuperAdminOnlyAdmin):
    list_display = ('code', 'libelle')
    search_fields = ('code', 'libelle')


# --- Modèles de données KPI (restreints par agence) ---

@admin.register(CompteClient)
class CompteClientAdmin(AgenceRestrictedAdmin):
    list_display = ('agence', 'categorie', 'annee', 'nombre', 'montant')
    list_filter = ('agence', 'categorie', 'annee')


@admin.register(Credit)
class CreditAdmin(AgenceRestrictedAdmin):
    list_display = ('agence', 'categorie', 'sous_type', 'annee', 'nombre', 'montant')
    list_filter = ('agence', 'categorie', 'annee')


@admin.register(VenteCarte)
class VenteCarteAdmin(AgenceRestrictedAdmin):
    list_display = ('agence', 'type_carte', 'annee', 'nombre', 'montant')
    list_filter = ('agence', 'type_carte', 'annee')


@admin.register(VenteTPE)
class VenteTPEAdmin(AgenceRestrictedAdmin):
    list_display = ('agence', 'annee', 'nombre', 'montant')
    list_filter = ('agence', 'annee')


@admin.register(Placement)
class PlacementAdmin(AgenceRestrictedAdmin):
    list_display = ('agence', 'type_placement', 'annee', 'nombre', 'montant')
    list_filter = ('agence', 'type_placement', 'annee')


@admin.register(Credoc)
class CredocAdmin(AgenceRestrictedAdmin):
    list_display = ('agence', 'sens', 'montant', 'date')
    list_filter = ('agence', 'sens')
    date_hierarchy = 'date'


# --- Restreindre Utilisateurs / Groupes (fournis par Django) à super_admin uniquement ---
admin.site.unregister(User)
admin.site.unregister(Group)


@admin.register(User)
class RestrictedUserAdmin(UserAdmin):
    def has_module_permission(self, request):
        if request.user.is_superuser:
            return True
        profil = get_profil(request)
        return profil is not None and profil.role == 'super_admin'

    def has_view_permission(self, request, obj=None):
        return self.has_module_permission(request)

    def has_add_permission(self, request):
        return self.has_module_permission(request)

    def has_change_permission(self, request, obj=None):
        return self.has_module_permission(request)

    def has_delete_permission(self, request, obj=None):
        return self.has_module_permission(request)


@admin.register(Group)
class RestrictedGroupAdmin(admin.ModelAdmin):
    def has_module_permission(self, request):
        if request.user.is_superuser:
            return True
        profil = get_profil(request)
        return profil is not None and profil.role == 'super_admin'

    def has_view_permission(self, request, obj=None):
        return self.has_module_permission(request)

    def has_add_permission(self, request):
        return self.has_module_permission(request)

    def has_change_permission(self, request, obj=None):
        return self.has_module_permission(request)

    def has_delete_permission(self, request, obj=None):
        return self.has_module_permission(request)