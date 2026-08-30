from django.urls import path
from django.views.generic import RedirectView
from . import views

urlpatterns = [
    path('', RedirectView.as_view(pattern_name='dashboard', permanent=False), name='home'),
    path('dashboard/', views.dashboard, name='dashboard'),

    path('import/', views.importer_excel, name='importer_excel'),
    path('export/', views.exporter_excel, name='exporter_excel'),

    path('comptes/', views.gerer_comptes, name='gerer_comptes'),
    path('comptes/creer/', views.creer_compte, name='creer_compte'),
    path('comptes/suspendre/<int:profil_id>/', views.suspendre_compte, name='suspendre_compte'),
    path('comptes/supprimer/<int:profil_id>/', views.supprimer_compte, name='supprimer_compte'),

    path('kpi/<str:type_kpi>/', views.liste_kpi, name='liste_kpi'),
    path('kpi/<str:type_kpi>/ajouter/', views.ajouter_kpi, name='ajouter_kpi'),
    path('kpi/<str:type_kpi>/modifier/<int:pk>/', views.modifier_kpi, name='modifier_kpi'),
    path('kpi/<str:type_kpi>/supprimer/<int:pk>/', views.supprimer_kpi, name='supprimer_kpi'),
    path('analyse/', views.analyse_reseau, name='analyse_reseau'),
    path('performance/', views.performance_reseau, name='performance_reseau'),
    path('communautes/', views.communautes_reseau, name='communautes_reseau'),
]