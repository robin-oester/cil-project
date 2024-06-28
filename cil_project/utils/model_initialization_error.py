class ModelInitializationError(Exception):
    """
    Custom error if a model cannot be initialized.
    """

    def __init__(self, parameter: str, message: str):
        super().__init__(f"{parameter}: {message}")
        self.parameter = parameter
        self.message = message
