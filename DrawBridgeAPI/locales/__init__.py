import os
import gettext

locale_dir = os.path.join(os.path.dirname(__file__))

lang = gettext.translation('messages', localedir=locale_dir, languages=['zh'], fallback=True)
lang.install()

_ = lang.gettext
i18n = _