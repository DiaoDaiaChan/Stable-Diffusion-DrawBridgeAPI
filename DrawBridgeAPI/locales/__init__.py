import os
import gettext
import locale

locale_dir = os.path.join(os.path.dirname(__file__), '../locales')
lang = os.getenv('LANG', None)
language, _ = locale.getdefaultlocale()

lang = gettext.translation(
    'messages', localedir=locale_dir, languages=[lang or language], fallback=True
)

lang.install()

_ = lang.gettext
