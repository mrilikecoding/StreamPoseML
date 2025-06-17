import pickle


class TrainedModel:
    """Encapsulates a trained model.

    This instance needs a model and an transformer that shapes data for the model's predict function.
    """

    def __init__(self):
        self.model = None
        self.model_data = None
        self.data_transformer = None
        self.notes = "There are no notes for this model saved."

    def transform_data(self, data: any) -> any:
        if self.data_transformer is not None and hasattr("transform"):
            return self.data_transformer.transform(data)
        else:
            raise ValueError(
                "No transformer is set on this model, or the transformer does not have a 'transform' function"
            )

    def predict(self, data: any) -> any:
        """Runs predict on the instance's trained model.

        Args:
            data: any
                The date to be used for the prediction
        Returns:
            the result of the model's prediction
        """
        if self.model is not None and hasattr(self.model, "predict"):
            return self.model.predict(data)
        else:
            raise ValueError(
                "No model is loaded, or the loaded model does not have a 'predict' function"
            )

    def set_data_transformer(self, transformer):
        """Transform passed sequence data into the right format for the loaded model"""
        self.data_transformer = transformer

    def set_model(self, model: any, model_data: any, notes: str = None) -> None:
        """Sets the passed model as this instance's mdoel if it has a predict method

        Args:
            model: any
                the model to be called by this instance. must implement "predict"
            notes: str
                details that describe the model
        Returns:
            None
        Raises:
            ValueError if predict is not implemented
        """
        if hasattr(model, "predict"):
            self.model = model
            self.model_data = model_data
            self.notes = notes
        else:
            raise ValueError("The loaded model does not implement a 'predict' function")
