"""
Version améliorée d'AraBERTNER :
 - Compatible CANERCorpus + ANERCorp
 - Compatible mapping dynamique label2id / id2label
 - Préparation complète du dataset multi-source
 - Tokenization + alignment intégré
 - Prediction améliorée
"""
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
)
from typing import Dict, List, Tuple
from datasets import DatasetDict
from .singleton import ModelSingleton


class AraBERTNER:
    """Gestion du modèle AraBERT pour NER multi-source"""

    def __init__(
        self,
        model_checkpoint: str,
        num_labels: int,
        id2label: Dict,
        label2id: Dict,
        max_length: int = 128,
    ):
        """
        Args:
            model_checkpoint: nom du modèle HF
            num_labels: nombre total de labels NER
            id2label: mapping id → label
            label2id: mapping label → id
            max_length: longueur max de tokenizer
        """

        self.model_checkpoint = model_checkpoint
        self.num_labels = num_labels
        self.id2label = id2label
        self.label2id = label2id
        self.max_length = max_length

        self.tokenizer = None
        self.model = None

        self._load_model()

    # -------------------------------------------------------------
    # 1️ Chargement modèle + tokenizer
    # -------------------------------------------------------------
    def _load_model(self):
        print(f"\nChargement du modèle en Singleton : {self.model_checkpoint}")

        singleton = ModelSingleton(
            model_checkpoint=self.model_checkpoint,
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id
        )

        # Référencer le modèle dans ta classe
        self.tokenizer = singleton.tokenizer
        self.model = singleton.model


    # -------------------------------------------------------------
    # 2️ Tokenization + alignement labels mots → sous-tokens
    # -------------------------------------------------------------
    def tokenize_and_align_labels(self, examples: Dict) -> Dict:
        tokenized = self.tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            padding=False,
            max_length=self.max_length,
        )

        all_labels = []

        for i, labels in enumerate(examples["ner_tags"]):
            word_ids = tokenized.word_ids(batch_index=i)

            prev = None
            aligned = []

            for w_id in word_ids:
                if w_id is None:
                    aligned.append(-100)
                elif w_id != prev:
                    aligned.append(labels[w_id])
                else:
                    aligned.append(-100)

                prev = w_id

            all_labels.append(aligned)

        tokenized["labels"] = all_labels
        return tokenized

    # -------------------------------------------------------------
    # 3️ Préparation du dataset fusionné (CANER + ANER)
    # -------------------------------------------------------------
    def prepare_dataset(self, dataset: DatasetDict) -> DatasetDict:
        """
        Prépare train/val/test via tokenization + alignement.
        """
        print("\n Préparation du dataset fusionné pour AraBERT...")

        tokenized_dataset = dataset.map(
            self.tokenize_and_align_labels,
            batched=True,
            remove_columns=dataset["train"].column_names,
            desc="Tokenisation NER",
        )

        self.data_collator = DataCollatorForTokenClassification(self.tokenizer)

        print("    Dataset prêt pour l'entraînement AraBERT")
        return tokenized_dataset

    # -------------------------------------------------------------
    # 4️ Prédiction unitaire
    # -------------------------------------------------------------
    def predict(self, text: str) -> List[Tuple[str, str]]:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)

        with torch.no_grad():
            outputs = self.model(**inputs)

        preds = torch.argmax(outputs.logits, dim=2)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        labels = [self.id2label[p.item()] for p in preds[0]]

        # supprimer tokens spéciaux
        result = []
        for tok, lab in zip(tokens, labels):
            if tok not in ["[CLS]", "[SEP]", "[PAD]"]:
                result.append((tok, lab))

        return result

    # -------------------------------------------------------------
    # 5️ Sauvegarde
    # -------------------------------------------------------------
    def save(self, path: str):
        print(f"\n Sauvegarde modèle dans : {path}")
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print("  Sauvegarde terminée")

    # -------------------------------------------------------------
    # 6️ Chargement custom depuis checkpoint
    # -------------------------------------------------------------
    @classmethod
    def load_from_checkpoint(cls, path: str) -> "AraBERTNER":
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForTokenClassification.from_pretrained(path)

        inst = cls.__new__(cls)
        inst.tokenizer = tokenizer
        inst.model = model
        inst.id2label = model.config.id2label
        inst.label2id = model.config.label2id
        inst.num_labels = model.config.num_labels

        return inst
