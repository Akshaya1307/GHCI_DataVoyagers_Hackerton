import joblib
from preprocess import preprocess
from rules import RuleEngine
import numpy as np


class Pipeline:
    """
    Hybrid Rule-Based + ML Pipeline
    """

    def __init__(
        self,
        model_path='model.pkl',
        vec_path='vectorizer.pkl',
        rules_path='rules.yaml'
    ):
        # Load model and vectorizer
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vec_path)

        # Load rule engine
        self.rule_engine = RuleEngine(rules_path)

        # Identify if model supports probability
        self.has_proba = hasattr(self.model, "predict_proba")

    # ------------------------------------------------------
    # INTERNAL TOKEN EXPLAINER (replaces explain.py)
    # ------------------------------------------------------
    def explain_tfidf(self, text: str, top_n: int = 5):
        """
        Returns:
        - tokens
        - weights
        - sorted list of (token, weight)
        """
        x = self.vectorizer.transform([text])
        feature_names = self.vectorizer.get_feature_names_out()

        if hasattr(self.model, "coef_"):
            # Linear models only
            weights = self.model.coef_
            class_idx = 0  # We'll override later in classify()

            token_weights = x.toarray()[0] * weights[class_idx]
            top_indices = np.argsort(token_weights)[::-1][:top_n]

            top_feats = [
                (feature_names[i], token_weights[i])
                for i in top_indices
                if token_weights[i] > 0
            ]
            return top_feats
        else:
            return []

    # ------------------------------------------------------
    # ML PREDICTOR
    # ------------------------------------------------------
    def ml_predict(self, cleaned_text: str):
        """
        Returns:
        - predicted label
        - confidence (if prob available)
        """
        x = self.vectorizer.transform([cleaned_text])

        if self.has_proba:
            probs = self.model.predict_proba(x)[0]
            idx = np.argmax(probs)
            return self.model.classes_[idx], float(probs[idx])
        else:
            pred = self.model.predict(x)[0]
            return pred, None

    # ------------------------------------------------------
    # MAIN CLASSIFY METHOD (Hybrid Decision)
    # ------------------------------------------------------
    def classify(self, raw_text: str, low_conf_thresh: float = 0.40):
        cleaned_text, mask_info = preprocess(raw_text)

        # -------------------------
        # 1. RULE-BASED CHECK
        # -------------------------
        rule_cat, rule_details = self.rule_engine.predict(cleaned_text)
        if rule_cat:
            return {
                'category': rule_cat,
                'confidence': 1.0,
                'source': 'Rule-based',
                'cleaned': cleaned_text,
                'explanation': rule_details,
                'top_features': []
            }

        # -------------------------
        # 2. ML PREDICTION
        # -------------------------
        pred_label, confidence = self.ml_predict(cleaned_text)

        # If low confidence
        status = "Uncertain" if (confidence is not None and confidence < low_conf_thresh) else "High Confidence"

        # -------------------------
        # 3. FEATURE EXPLANATION
        # -------------------------
        top_feats = self.explain_tfidf(cleaned_text)

        # -------------------------
        # Final Response
        # -------------------------
        return {
            'category': pred_label,
            'confidence': confidence,
            'source': 'ML-based',
            'status': status,
            'cleaned': cleaned_text,
            'top_features': top_feats,
            'explanation': None
        }


if __name__ == "__main__":
    p = Pipeline()
    print(p.classify("Paid to hp petrol pump 1200 via upi"))
