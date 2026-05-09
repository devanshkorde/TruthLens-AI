"""
TruthLens AI — Model Training Script
======================================
Run this ONCE to train the TF-IDF + Logistic Regression text classifier.
Uses a synthetic + rule-based dataset for demonstration.

Usage: python train_model.py

For production: replace the dataset with Kaggle LIAR or FakeNewsNet dataset.
"""

import os
import sys
import random
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

sys.path.insert(0, str(Path(__file__).parent.parent))

OUTPUT_DIR = Path(__file__).parent

# ──────────────────────────────────────────────
# Synthetic Training Data
# ──────────────────────────────────────────────

FAKE_SAMPLES = [
    "SHOCKING: You won't believe what the government is hiding from you right now!",
    "BREAKING: Scientists EXPOSED for lying about vaccines — share before deleted!",
    "The DEEP STATE has been controlling elections for decades — wake up people!",
    "URGENT ALERT: 5G towers spreading disease, doctors don't want you to know!",
    "EXPOSED: The media is covering up the TRUTH about this deadly conspiracy!",
    "Share this NOW before it gets censored — the government is lying to you!",
    "BOMBSHELL: Celebrity reveals shocking secret about COVID that will terrify you!",
    "ALERT: Bill Gates planning to microchip entire population through vaccines!",
    "This VIDEO will SHOCK you — mainstream media is hiding the real story!",
    "BREAKING: Top doctor exposes the truth about cancer cure being suppressed!",
    "The elite don't want you to know this — share before they delete it!",
    "EXPOSED: NASA faking moon landing proven by new evidence — shocking truth!",
    "WARNING: New law will destroy your freedom — act now before it's too late!",
    "SECRET DOCUMENT leaked proving that climate change is a massive hoax!",
    "Nakli khabar: Sarkar chhupa rahi hai yeh sach, share karo turant!",
    "VIRAL: What they're not telling you about the new COVID variant — exposed!",
    "100% PROOF: Elections are being stolen and no one is talking about it!",
    "MIND-BLOWING: Ancient texts predicted this crisis — mainstream media silent!",
    "They're poisoning our water supply! Doctors are paid to keep quiet!",
    "BREAKING: Whistleblower exposes massive government fraud — share immediately!",
    "You've been lied to your whole life about this common food — doctors shocked!",
    "OUTRAGEOUS: Politicians caught on tape planning to destroy the country!",
    "This miracle cure is being suppressed by Big Pharma — share to save lives!",
    "ALERT: Famous actor reveals the TRUTH about Hollywood that will shock you!",
    "The TRUTH about 9/11 they don't want you to know — share before censored!",
    "SHOCKING REVELATION: Top scientist admits climate data has been fabricated!",
    "EXPOSED: How the banking elite control every government on the planet!",
    "NEW EVIDENCE: Moon landing was staged — NASA finally caught lying!",
    "MUST READ: The real reason why so many people are dying — it's not what you think!",
    "ALERT: Secret ingredient in tap water causing brain damage — experts hiding truth!",
]

REAL_SAMPLES = [
    "The Reserve Bank of India raised interest rates by 25 basis points on Wednesday, according to official statements.",
    "A study published in the New England Journal of Medicine found that the vaccine reduced hospitalizations by 89%.",
    "Prime Minister Modi addressed parliament on Thursday to discuss the new infrastructure development plan.",
    "Researchers at IIT Delhi have developed a new water purification technology, reports The Hindu.",
    "The Supreme Court ruled on the constitutional validity of the new data protection bill today.",
    "India's GDP grew by 6.2% in the third quarter, according to data released by the Ministry of Statistics.",
    "Scientists have discovered a new exoplanet in the habitable zone, publishing findings in Nature journal.",
    "The Election Commission announced the schedule for upcoming state assembly elections.",
    "Floods in Assam have displaced over 100,000 people, NDRF teams deployed for rescue operations.",
    "WHO released updated guidelines on antibiotic use to combat growing resistance, recommending...",
    "The Union Budget 2024 allocates ₹10 lakh crore for infrastructure development, Finance Minister said.",
    "NASA's Mars rover has found evidence of ancient water systems on the Martian surface.",
    "India and the US signed a trade agreement covering digital services and technology exports.",
    "Bengaluru police arrested three suspects in connection with the bank fraud case involving ₹50 crore.",
    "The new education policy aims to increase enrollment in higher education to 50% by 2035.",
    "According to UNICEF data, India has significantly reduced child mortality rates in the past decade.",
    "The government launched a new portal for farmers to access crop insurance schemes directly.",
    "Mumbai's new coastal road project is expected to reduce travel time by 45 minutes during peak hours.",
    "IISc researchers published a breakthrough in quantum computing in the journal Science.",
    "The Comptroller and Auditor General released its annual audit report highlighting financial irregularities.",
    "The National Highway Authority completed 10,000 km of new road construction this fiscal year.",
    "India ranked 40th in the Global Innovation Index 2024, improving by 6 positions from last year.",
    "The Ministry of Health issued updated protocols for dengue prevention ahead of monsoon season.",
    "The World Bank approved a $500 million loan for India's renewable energy expansion program.",
    "SEBI introduced new regulations for algorithmic trading to ensure market stability.",
    "The Indian Space Research Organisation successfully launched three satellites in a single mission.",
    "Puducherry government announced new welfare schemes for fishermen ahead of the cyclone season.",
    "The Comptroller general found that 87% of MGNREGA funds were properly accounted for in 2023.",
    "Medical experts from AIIMS published research on effective management of post-COVID symptoms.",
    "India's exports increased by 12% year-on-year in March 2024, driven by engineering goods.",
]

# ──────────────────────────────────────────────
# Build dataset
# ──────────────────────────────────────────────

texts = FAKE_SAMPLES + REAL_SAMPLES
labels = [1] * len(FAKE_SAMPLES) + [0] * len(REAL_SAMPLES)  # 1=FAKE, 0=REAL

print(f"[Train] Dataset: {len(texts)} samples ({len(FAKE_SAMPLES)} fake, {len(REAL_SAMPLES)} real)")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

# TF-IDF Vectorization
print("[Train] Fitting TF-IDF vectorizer...")
vectorizer = TfidfVectorizer(
    ngram_range=(1, 3),
    max_features=5000,
    min_df=1,
    sublinear_tf=True,
    strip_accents="unicode",
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Logistic Regression
print("[Train] Training Logistic Regression classifier...")
classifier = LogisticRegression(
    C=1.5,
    max_iter=500,
    class_weight="balanced",
    random_state=42,
)
classifier.fit(X_train_vec, y_train)

# Evaluate
y_pred = classifier.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)
print(f"\n[Train] Test Accuracy: {acc:.2%}")
print(classification_report(y_test, y_pred, target_names=["REAL", "FAKE"]))

# Save models
vectorizer_path = OUTPUT_DIR / "tfidf_vectorizer.pkl"
classifier_path = OUTPUT_DIR / "text_classifier.pkl"

joblib.dump(vectorizer, vectorizer_path)
joblib.dump(classifier, classifier_path)

print(f"\n[Train] Models saved successfully!")
print(f"  Vectorizer: {vectorizer_path}")
print(f"  Classifier: {classifier_path}")
print("\n[Train] Run the FastAPI server now - text model is ready!")
