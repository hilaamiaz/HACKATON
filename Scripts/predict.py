import torch
import joblib
from base_model import BaseModel
from model import ModelArchitecture

class Model(BaseModel):
    def __init__(self):
        # Initialize the architecture required by model.py
        self.model = ModelArchitecture()
        self.model.eval() # Set to evaluation mode

    def load(self, weights_path: str) -> None:
        """Loads the model weights saved via joblib."""
        state_dict = joblib.load(weights_path)
        self.model.load_state_dict(state_dict)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs inference and returns a 1D tensor of predicted class indices.
        """
        with torch.no_grad():
            logits = self.model(x)
            # Get the index of the max logit (the predicted class 0-19)
            predictions = torch.argmax(logits, dim=1)
        return predictions