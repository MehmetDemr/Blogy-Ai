# train_tagger.py
# Çoklu etiketli (multi-label) metin sınıflandırma modeli eğitimi
# TF-IDF + Logistic Regression (One-vs-Rest) yaklaşımı kullanılır

import json
from pathlib import Path

# Metin vektörleştirme ve çoklu etiketleme araçları
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

# Sınıflandırıcılar
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

# Pipeline ve değerlendirme yardımcıları
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Metin temizleme için regex
import re

# Model kaydetmek/yüklemek için
import joblib


# Sistemde desteklenen SABİT etiket listesi
# Model yalnızca bu etiketleri öğrenir
ALL_TAGS = [
    "Gündem",
    "Spor",
    "Teknoloji",
    "Manzara",
    "Tarihi Eser",
    "Yemek",
    "Görülecek Yer",
    "Mekan Önerisi",
    "Öğretici",
    "Ders",
]


# JSONL formatındaki dataset'i yükler

def load_dataset(path: str):
    texts = []   # Metinler
    labels = []  # Etiket listeleri

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            obj = json.loads(line)

            txt = obj["text"]
            lbls = obj["labels"]

            # Sadece izin verilen etiketleri tut
            lbls = [l for l in lbls if l in ALL_TAGS]

            # Etiketsiz veri kullanma
            if not lbls:
                continue

            texts.append(txt)
            labels.append(lbls)

    return texts, labels


# Basit metin temizleme fonksiyonu
def simple_preprocess(text: str) -> str:
    text = text.lower()  # Küçük harfe çevir
    text = re.sub(r"http\S+", " ", text)  # URL temizle
    text = re.sub(
        r"[^a-zçğıöşüA-ZÇĞİÖŞÜ0-9\s]", " ", text
    )  # Noktalama temizle
    text = re.sub(r"\s+", " ", text).strip()  # Fazla boşlukları düzelt
    return text


# Ana eğitim fonksiyonu
def main():
    data_path = "dataset.jsonl"

    # Dataset yüklenir
    X, Y = load_dataset(data_path)
    print(f"Toplam örnek: {len(X)}")

    # Tüm metinler önceden temizlenir
    X_preprocessed = [simple_preprocess(text) for text in X]

    # Çoklu etiketleri binary formata çevirir
    # Örn: ["Spor", "Gündem"] → [1,1,0,0,...]
    mlb = MultiLabelBinarizer(classes=ALL_TAGS)
    Y_bin = mlb.fit_transform(Y)

    # Eğitim / test ayrımı
    X_train, X_test, y_train, y_test = train_test_split(
        X_preprocessed,
        Y_bin,
        test_size=0.2,
        random_state=42
    )

    # Model Pipeline
    # 1) TF-IDF (1-gram + 2-gram)
    # 2) One-vs-Rest Logistic Regression
    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=20000,
                    min_df=2,
                    lowercase=False,  
                ),
            ),
            (
                "clf",
                OneVsRestClassifier(
                    LogisticRegression(
                        max_iter=200,
                        n_jobs=-1
                    )
                ),
            ),
        ]
    )

    # Model eğitimi
    print("Model eğitiliyor...")
    pipeline.fit(X_train, y_train)

    # Test seti değerlendirmesi
    print("Test seti değerlendirme:")
    y_pred = pipeline.predict(X_test)

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=mlb.classes_,
            zero_division=0
        )
    )

    # Model ve label encoder kaydedilir
    out_dir = Path("models")
    out_dir.mkdir(exist_ok=True)

    joblib.dump(pipeline, out_dir / "tag_model.joblib")
    joblib.dump(mlb, out_dir / "tag_mlb.joblib")

    print("✅ Model models/ içine kaydedildi (joblib).")


# Script doğrudan çalıştırılırsa main çağrılır
if __name__ == "__main__":
    main()
