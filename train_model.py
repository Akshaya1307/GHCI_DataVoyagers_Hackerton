import argparse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import joblib
from preprocess import preprocess


def load_data(path: str):
    df = pd.read_csv(path)
    df = df.dropna()
    return df


def prepare(df: pd.DataFrame):
    texts = []
    for t in df['transaction_text'].tolist():
        cleaned, _ = preprocess(t)
        texts.append(cleaned)
    return texts, df['label'].tolist()


def train(input_csv: str, output_dir: str = './'):
    print("Loading data...")
    df = load_data(input_csv)

    print("Preprocessing...")
    X_texts, y = prepare(df)

    print("Vectorizing...")
    vect = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=12000,
        min_df=2,
        sublinear_tf=True
    )
    X = vect.fit_transform(X_texts)

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=list(set(y)),
        y=y
    )
    class_weights_dict = {
        c: w for c, w in zip(list(set(y)), class_weights)
    }

    print("Training model...")
    clf = LogisticRegression(
        max_iter=2000,
        class_weight=class_weights_dict,
        n_jobs=-1
    )
    clf.fit(X, y)

    preds = clf.predict(X)
    print("\n=== TRAINING REPORT ===")
    print(classification_report(y, preds))

    print("Saving...")
    joblib.dump(clf, output_dir + 'model.pkl')
    joblib.dump(vect, output_dir + 'vectorizer.pkl')

    print(f"\nMODEL SAVED IN {output_dir}")
    print("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output_dir', default='./')
    args = parser.parse_args()

    train(args.input, args.output_dir)
