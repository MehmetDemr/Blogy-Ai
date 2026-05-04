import csv
import json
from pathlib import Path

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


def convert(input_path="raw_dataset.txt", output_path="dataset.jsonl"):
    in_path = Path(input_path)
    out_path = Path(output_path)

    if not in_path.exists():
        print(f"Girdi dosyası bulunamadı: {in_path}")
        return

    # 1) VARSA ESKİ JSONL'İ OKU, TEXT'LERE GÖRE SET OLUŞTUR
    existing_texts = set()
    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    txt = obj.get("text")
                    if txt:
                        existing_texts.add(txt)
                except json.JSONDecodeError:
                    continue

    print(f"Mevcut dataset.jsonl içindeki örnek sayısı: {len(existing_texts)}")

    count_in = 0
    count_new = 0

    # 2) INPUT TXT'Yİ OKU, YENİ OLANLARI APPEND ET
    with in_path.open("r", encoding="utf-8") as fin, out_path.open(
        "a", encoding="utf-8"
    ) as fout:
        reader = csv.reader(fin, delimiter=",", quotechar='"')

        for row in reader:
            if not row:
                continue

            # Header satırını atla
            first = row[0].strip()
            if first.lower().startswith("no"):
                continue

            if len(row) < 3:
                print("Satır atlandı (3 sütun yok):", ", ".join(row))
                continue

            count_in += 1

            _id = row[0].strip()
            text = row[1].strip()
            labels_raw = row[2].strip()

            # Eğer bu text zaten dataset.jsonl'de varsa, atla
            if text in existing_texts:
                continue

            cleaned = labels_raw.replace('""', '"')
            try:
                labels = json.loads(cleaned)
            except json.JSONDecodeError:
                print("Label parse edilemedi, satır atlandı:", labels_raw)
                continue

            labels = [l for l in labels if l in ALL_TAGS]
            if not labels:
                print(f"Etiketsiz satır atlandı (id={_id})")
                continue

            obj = {
                "text": text,
                "labels": labels,
            }

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            existing_texts.add(text)
            count_new += 1

    print("==== Tamamlandı ====")
    print("Girdi satırı (okunan):", count_in)
    print("Yeni eklenen örnek:", count_new)
    print("Toplam örnek (tahmini):", len(existing_texts))
    print("Çıktı dosyası:", out_path.resolve())


if __name__ == "__main__":
    convert()
