import gettext
import os

locale_dir = os.path.join(os.path.dirname(__file__), "locales")
lang = gettext.translation('messages', localedir=locale_dir, languages=['zh'])
lang.install()

_ = lang.gettext

