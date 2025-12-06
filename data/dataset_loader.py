"""
Version MULTI-SOURCE du loader NER
----------------------------------
Supporte automatiquement :
 - CANERCorpus : Word / Acutal
 - ANERCorp    : word / tag
 - Normalisation BIO
 - Conversion HuggingFace Dataset
"""

import json
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict


class MultiSourceNERLoader:

    def __init__(self, paths: List[str], train_split=0.8, val_split=0.1, random_state=42):
        self.paths = paths
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = 1 - train_split - val_split
        self.random_state = random_state

        self.label_list = None
        self.label2id = None
        self.id2label = None

        self.O_words = ["قال", "إن", "ما", "من", "في", "على", "و", "ثم", "عن"]


    # -------------------------------------------------------------
    # 1) Charger plusieurs fichiers
    # -------------------------------------------------------------
    def load_all_sources(self) -> List[Dict]:
        all_data = []

        for path in self.paths:
            print(f"📥 Lecture : {path}")
            with open(path, "r", encoding="utf-8") as f:
                content = json.load(f)
                all_data.extend(content)

        print(f"→ Total brut fusionné : {len(all_data)} exemples")
        return all_data


    # -------------------------------------------------------------
    # 2) Normalisation d'une entrée (CANER ou ANER)
    # -------------------------------------------------------------
    def normalize_entry(self, ex: Dict) -> Tuple[str, str] | None:

        # Format CANERCorpus
        if "Word" in ex and "Acutal" in ex:
            word = ex["Word"]
            tag = ex["Acutal"]

        # Format ANERCorp
        elif "word" in ex and "tag" in ex:
            word = ex["word"]
            tag = ex["tag"]

        else:
            return None

        if not isinstance(word, str):
            return None
        if not isinstance(tag, str):
            return None

        word = word.strip()
        tag = tag.strip()

        if word == "" or tag == "":
            return None

        # BIO conversion
        if tag.upper() == "O":
            final_tag = "O"
        else:
            final_tag = "B-" + tag

        return word, final_tag


    # -------------------------------------------------------------
    # 3) Convertir tout le dataset en BIO
    # -------------------------------------------------------------
    def convert_to_bio(self, raw: List[Dict]):
        data = []
        skipped = 0

        for ex in raw:
            result = self.normalize_entry(ex)

            if result is None:
                skipped += 1
                continue

            word, tag = result

            data.append({
                "tokens": [word],
                "ner_tags": [tag]
            })

        print(f"→ Exemples valides : {len(data)} | Ignorés : {skipped}")
        return data


    # -------------------------------------------------------------
    # 4) Ajouter la classe O (équilibrage)
    # -------------------------------------------------------------
    def add_missing_O_examples(self, data, num_samples=5000):
        for i in range(num_samples):
            w = self.O_words[i % len(self.O_words)]
            data.append({"tokens": [w], "ner_tags": ["O"]})
        return data


    # -------------------------------------------------------------
    # 5) Construire label2id / id2label
    # -------------------------------------------------------------
    def create_label_mappings(self, data):
        labels = set()

        for ex in data:
            labels.update(ex["ner_tags"])

        self.label_list = sorted(labels)
        self.label2id = {l: i for i, l in enumerate(self.label_list)}
        self.id2label = {i: l for l, i in self.label2id.items()}

        print(f"→ Labels détectés : {self.label_list}")


    # -------------------------------------------------------------
    # 6) Split des données
    # -------------------------------------------------------------
    def split_data(self, data):
        train, temp = train_test_split(
            data, test_size=self.val_split + self.test_split, random_state=self.random_state
        )
        val, test = train_test_split(
            temp,
            test_size=self.test_split / (self.val_split + self.test_split),
            random_state=self.random_state
        )

        return train, val, test


    # -------------------------------------------------------------
    # 7) Conversion en Dataset HuggingFace
    # -------------------------------------------------------------
    def create_hf_dataset(self, data_list):
        clean = {"tokens": [], "ner_tags": []}

        for ex in data_list:
            clean["tokens"].append(ex["tokens"])
            clean["ner_tags"].append([self.label2id[tag] for tag in ex["ner_tags"]])

        return Dataset.from_dict(clean)


    # -------------------------------------------------------------
    # 8) Pipeline complet
    # -------------------------------------------------------------
    def load_and_prepare(self):
        print("\n=== 📌 ÉTAPE 1 : Chargement multi-source ===")
        raw = self.load_all_sources()

        print("\n=== 📌 ÉTAPE 2 : Conversion BIO ===")
        data = self.convert_to_bio(raw)

        print("\n=== 📌 ÉTAPE 3 : Ajout des exemples O ===")
        data = self.add_missing_O_examples(data)

        print("\n=== 📌 ÉTAPE 4 : Label mappings ===")
        self.create_label_mappings(data)

        print("\n=== 📌 ÉTAPE 5 : Split ===")
        train, val, test = self.split_data(data)

        print("\n=== 📌 ÉTAPE 6 : Conversion HF Dataset ===")
        dataset = DatasetDict({
            "train": self.create_hf_dataset(train),
            "validation": self.create_hf_dataset(val),
            "test": self.create_hf_dataset(test)
        })

        print("\n✅ Dataset fusionné prêt pour fine-tuning !")

        return dataset, {
            "label_list": self.label_list,
            "label2id": self.label2id,
            "id2label": self.id2label,
            "num_labels": len(self.label_list),
            "train_size": len(train),
            "val_size": len(val),
            "test_size": len(test),
            "total_examples": len(data)
        }
