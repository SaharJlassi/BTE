from django import forms
from .models import CompteClient, Credit, VenteCarte, VenteTPE, Placement, Credoc


class CompteClientForm(forms.ModelForm):
    class Meta:
        model = CompteClient
        exclude = ['agence']


class CreditForm(forms.ModelForm):
    class Meta:
        model = Credit
        exclude = ['agence']

    def clean(self):
        cleaned_data = super().clean()
        categorie = cleaned_data.get('categorie')
        sous_type = cleaned_data.get('sous_type')

        if categorie == 'gestion' and not sous_type:
            self.add_error('sous_type', "Le sous-type est obligatoire pour un crédit de gestion.")
        if categorie != 'gestion' and sous_type:
            self.add_error('sous_type', "Le sous-type ne doit être renseigné que pour un crédit de gestion.")

        return cleaned_data


class VenteCarteForm(forms.ModelForm):
    class Meta:
        model = VenteCarte
        exclude = ['agence']


class VenteTPEForm(forms.ModelForm):
    class Meta:
        model = VenteTPE
        exclude = ['agence']


class PlacementForm(forms.ModelForm):
    class Meta:
        model = Placement
        exclude = ['agence']


class CredocForm(forms.ModelForm):
    class Meta:
        model = Credoc
        exclude = ['agence']
        widgets = {
            'date': forms.DateInput(attrs={'type': 'date'}),
        }