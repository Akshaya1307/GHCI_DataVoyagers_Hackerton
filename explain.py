# explain.py
from typing import List, Tuple
import joblib
import numpy as np

def explain_tfidf(clf_path='model.pkl', vec_path='vectorizer.pkl', text='') -> Tuple[str, float, List[Tuple[str, float]]]:
    """
    Returns: (predicted_class, probability, top_features)
    top_features is a list of (feature_name, weight)
    """
    clf = joblib.load(clf_path)
    vec = joblib.load(vec_path)

    x = vec.transform([text])
    pred = clf.predict(x)[0]
    prob = None
    if hasattr(clf, 'predict_proba'):
        prob = float(max(clf.predict_proba(x)[0]))

    # For LogisticRegression get top positive weights for predicted class
    top_feats = []
    if hasattr(clf, 'coef_'):
        classes = list(clf.classes_)
        if pred in classes:
            idx = classes.index(pred)
            coefs = clf.coef_[idx]
            feature_names = vec.get_feature_names_out()
            topn_ids = np.argsort(coefs)[-5:][::-1]
            top_feats = [(feature_names[i], float(coefs[i])) for i in topn_ids]

    return pred, prob, top_feats

if __name__ == '__main__':
    p, prob, feats = explain_tfidf(text='hp petrol pump 1250')
    print('Pred:', p, 'Prob:', prob)
    print('Top features:', feats)
