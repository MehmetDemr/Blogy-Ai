import json
from collections import Counter

DATA_PATH = "dataset.jsonl"  # senin dosyan

def main():
    label_counter = Counter()
    cooc_counter = Counter()
    total = 0

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            labels = obj.get("labels", [])
            labels = [l.strip() for l in labels]

            if not labels:
                continue

            total += 1
            # tek tek say
            label_counter.update(labels)
            # birlikte geçen etiket çiftlerini say
            uniq = sorted(set(labels))
            for i in range(len(uniq)):
                for j in range(i + 1, len(uniq)):
                    cooc_counter[(uniq[i], uniq[j])] += 1

    print("Toplam örnek:", total)
    print("\n=== Etiket Frekansları ===")
    for lbl, cnt in label_counter.most_common():
        print(f"{lbl:15s} -> {cnt}")

    print("\n=== En çok birlikte görülen etiket çiftleri (top 15) ===")
    for (a, b), cnt in cooc_counter.most_common(15):
        print(f"{a} + {b:15s} -> {cnt}")

if __name__ == "__main__":
    main()
