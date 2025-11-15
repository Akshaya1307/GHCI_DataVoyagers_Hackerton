# 🛡️ Privacy-First On-Device Transaction Classifier

A fully offline, privacy-focused machine learning system that classifies UPI and bank transaction text into categories such as Shopping, Dining, Fuel, EMI/Loan, Utilities, Subscriptions, Wallet, Transfers, ATM, and more.
Combines **PII masking**, **rule-based matching**, and **TF-IDF + Logistic Regression** for high accuracy.
Features a beautiful **dark-mode Streamlit UI** with CSV upload and downloadable results.

---

## ✨ Features

* 🔐 **100% Offline — No Cloud Usage**
* 🧠 **Hybrid Rule + ML Engine**
* 🧽 **Automatic PII Masking** (UPI IDs, phone, email, card numbers, names)
* 🌓 **Premium Dark Mode UI**
* 📁 **CSV Upload & CSV Output**
* 🏷️ **Category Badges & Confidence Scores**
* 🔍 **Token-Level ML Explanation**

---

## 🧠 Tech Stack

* Python
* Scikit-Learn
* Pandas
* Streamlit
* YAML Rules Engine
* TF-IDF Vectorizer

---

## 🚀 How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Your app opens at:
`http://localhost:8501`

---

## 📂 Project Structure

```
├── app.py                # Streamlit UI
├── pipeline.py           # Hybrid ML + rules pipeline
├── preprocess.py         # PII masking + cleaning
├── rules.py              # Rule engine
├── rules.yaml            # Merchant patterns
├── explain.py            # ML token explanation
├── model.pkl             # Trained ML model
├── vectorizer.pkl        # TF-IDF vectorizer
├── requirements.txt
└── README.md
```

---

## 📌 Short Description (50 words)

A privacy-first, offline machine learning system that classifies financial transactions using a hybrid rules + ML approach. It includes PII masking, rule detection, TF-IDF vectorization, logistic regression classification, CSV upload support, and a premium dark-mode Streamlit UI for secure, fast, and accurate transaction categorization.

---

## 🏆 Highlights

* Real FinTech-style interface
* Zero data leakage
* High-quality modular code
* Suitable for academic, project, and production demos
