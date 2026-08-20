# KPI BTE — Tableau de bord des indicateurs d'agences

Application web de suivi des indicateurs de performance des 32 agences
de la Banque de Tunisie et des Émirats.

Projet réalisé dans le cadre d'un stage.

## Fonctionnalités

- Trois niveaux d'accès : super administrateur, chef d'agence, employé
- Import de fichiers Excel avec répartition automatique par agence
- Export Excel des données consolidées
- Tableau de bord avec filtres par année et par agence
- Analyse du réseau : segmentation, détection d'agences atypiques, profils d'activité
- Analyse de performance : comparaison du réalisé au niveau attendu

## Technologies

Django 6.0, SQLite, scikit-learn, openpyxl, Chart.js, Bootstrap 5

## Installation

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python manage.py migrate
python manage.py populate_ref
python manage.py createsuperuser
python manage.py runserver
```

## Note

Les fichiers de données ne sont pas versionnés pour des raisons de confidentialité.