from django import template

register = template.Library()


@register.filter
def dictkey(dictionnaire, cle):
    """Permet d'accéder à d[cle] depuis un template Django."""
    if not dictionnaire:
        return ''
    return dictionnaire.get(cle) or dictionnaire.get(str(cle), '')