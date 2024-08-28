class DrawBridgeAPIException(Exception):

    class DBAPIExceptions(Exception):
        pass

    class TokenExpired(DBAPIExceptions):
        def __init__(self, message="Token expired."):
            self.message = message
            super().__init__(self.message)

    class NeedRecaptcha(DBAPIExceptions):
        def __init__(self, message="Need Recaptcha."):
            self.message = message
            super().__init__(self.message)

