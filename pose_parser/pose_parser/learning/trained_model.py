import pickle


class TrainedModel:
    """Encapsulates a trained model"""

    def __init__(self):
        self.model = None
        self.data_transformer = None

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
        self.transformer = transformer

    def load_trained_model(self, location: str) -> bool:
        try:
            with open(location, "rb") as file:
                model = pickle.load(file)

            if hasattr(model, "predict"):
                self.model = model
                return True
            else:
                raise ValueError(
                    "The loaded model does not implement a 'predict' function"
                )

        except Exception as e:
            print(f"Error loading the model: {e}")
            return False
