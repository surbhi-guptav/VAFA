import sys

import torch
from assisstants.exception.exception import AssisstantException
from assisstants.logging.logger import logging

from assisstants.loader.model_loader import ModelLoader


class TextClassifier:
    def classify(self, text, model=None, tokenizer=None, return_prob: bool = False):
        try:
            logging.info("Text Classification Started")

            # allow caller to pass model/tokenizer (used by app cache)
            model = model or ModelLoader.get_model()
            tokenizer = tokenizer or ModelLoader.get_tokenizer()
            device = ModelLoader._init_device()

            inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=64)

            # move tensors to device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            # compute probabilities and obtain confidence
            probs = torch.softmax(outputs.logits, dim=1)
            max_prob, pred_idx = torch.max(probs, dim=1)
            prediction = pred_idx.item()
            confidence = float(max_prob.item())

            categories = ["Name", "Phone Number", "Amount", "Account Number"]

            predicted_label = categories[prediction]

            logging.info(f"Text Classification Completed: {predicted_label} (conf={confidence:.3f})")

            if return_prob:
                return predicted_label, confidence
            return predicted_label
        except Exception as e:
            raise AssisstantException(e, sys)