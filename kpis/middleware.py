class NoCacheMiddleware:
    """
    Empêche le navigateur de mettre en cache les pages de l'application.
    Ainsi, après déconnexion, le bouton "Précédent" force une nouvelle
    requête au serveur — qui redirige vers /login/ si la session a expiré,
    au lieu d'afficher une page sensible depuis le cache du navigateur.
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)
        response['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response['Pragma'] = 'no-cache'
        response['Expires'] = '0'
        return response