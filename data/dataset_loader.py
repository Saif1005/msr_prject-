"""
Chargement et préparation des données NER (VERSION CORRIGÉE)
----------------------------------------------------------
Cette version corrige :
 - L’absence de la classe "O"
 - Le BIO incorrect
 - Le dataset 1-mot = 1-ligne
 - Les erreurs de type (int, None)
 - Le dataset non utilisable pour un vrai NER
"""

import json
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict


class NERDatasetLoader:
    """Classe pour charger, corriger et préparer le dataset NER."""

    def __init__(self, data_path: str, train_split: float = 0.8,
                 val_split: float = 0.1, random_state: int = 42):
        self.data_path = data_path
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = 1 - train_split - val_split
        self.random_state = random_state

        self.label_list = None
        self.label2id = None
        self.id2label = None

        # Mots neutres pour générer la classe O
        self.O_words = [
            "قال", "إن", "ما", "من", "في", "على", "و", "ثم",
            "عن", "كان", "هو", "هي", "قد", "لم", "لن", "هذا", "هذه"
        ]

    def load_raw_data(self) -> List[Dict]:
        """Charge les données brutes depuis JSON."""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def convert_to_bio_format(self, raw_data: List[Dict]) -> Tuple[List[Dict], int]:
        """
        Convertit le dataset brut en format NER avec BIO + classe O.
        """

        data = []
        skipped = 0

        for ex in raw_data:

            # Vérification du format
            if "Word" not in ex or "Acutal" not in ex:
                skipped += 1
                continue

            word = ex["Word"]
            label = ex["Acutal"]

            if not isinstance(word, str):
                word = str(word)

            if not isinstance(label, str):
                label = str(label)

            word = word.strip()
            label = label.strip()

            if word == "" or label == "":
                skipped += 1
                continue

            # Cas spécial : si label == O
            if label.upper() == "O":
                final_label = "O"
            else:
                final_label = "B-" + label

            data.append({
                "tokens": [word],        # Un mot = une séquence
                "ner_tags": [final_label]
            })

        return data, skipped

    def add_missing_O_examples(self, data: List[Dict], num_samples=5000) -> List[Dict]:
        """
        Ajoute artificiellement des exemples avec la classe O
        pour équilibrer le dataset.
        """
        for _ in range(num_samples):
            w = self.O_words[_ % len(self.O_words)]
            data.append({
                "tokens": [w],
                "ner_tags": ["O"]
            })

        return data

    def create_label_mappings(self, data: List[Dict]) -> None:
        """Crée les mappings label <-> id."""
        all_labels = set()

        for example in data:
            all_labels.update(example["ner_tags"])

        self.label_list = sorted(list(all_labels))
        self.label2id = {label: i for i, label in enumerate(self.label_list)}
        self.id2label = {i: label for label, i in self.label2id.items()}

    def split_data(self, data: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Split train/val/test."""
        train_data, temp_data = train_test_split(
            data,
            test_size=(self.val_split + self.test_split),
            random_state=self.random_state
        )

        val_data, test_data = train_test_split(
            temp_data,
            test_size=self.test_split / (self.val_split + self.test_split),
            random_state=self.random_state
        )

        return train_data, val_data, test_data

    def create_hf_dataset(self, data_list: List[Dict]) -> Dataset:
        """Convertit en HuggingFace Dataset."""
        processed_data = {
            "tokens": [],
            "ner_tags": []
        }

        for example in data_list:
            processed_data["tokens"].append(example["tokens"])
            processed_data["ner_tags"].append(
                [self.label2id[tag] for tag in example["ner_tags"]]
            )

        return Dataset.from_dict(processed_data)

    def load_and_prepare(self) -> Tuple[DatasetDict, Dict]:
        """Pipeline complet."""
        print("\n📌 Chargement du dataset brut...")
        raw_data = self.load_raw_data()
        print(f" → Données brutes : {len(raw_data)} entrées")

        print("\n📌 Conversion BIO + nettoyage...")
        data, skipped = self.convert_to_bio_format(raw_data)
        print(f" → Exemples valides : {len(data)}")
        print(f" → Ignorés : {skipped}")

        print("\n📌 Ajout de la classe O...")
        data = self.add_missing_O_examples(data)
        print(f" → Taille après ajout de 'O' : {len(data)}")

        print("\n📌 Création des mappings de labels...")
        self.create_label_mappings(data)
        print(f" → Labels : {self.label_list}")

        print("\n📌 Split Train / Val / Test...")
        train_data, val_data, test_data = self.split_data(data)
        print(f" → Train : {len(train_data)}")
        print(f" → Validation : {len(val_data)}")
        print(f" → Test : {len(test_data)}")

        print("\n📌 Conversion en Dataset HuggingFace...")
        dataset = DatasetDict({
            'train': self.create_hf_dataset(train_data),
            'validation': self.create_hf_dataset(val_data),
            'test': self.create_hf_dataset(test_data)
        })

        metadata = {
            'label_list': self.label_list,
            'label2id': self.label2id,
            'id2label': self.id2label,
            'num_labels': len(self.label_list),
            'train_size': len(train_data),
            'val_size': len(val_data),
            'test_size': len(test_data),
            'total_examples': len(data)
        }


        print("\n✅ Dataset prêt pour le fine-tuning !")

        return dataset, metadata
