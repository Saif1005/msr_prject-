"""
Gestion du modèle AraBERT pour NER
"""
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification
)
from typing import Dict, List, Tuple


class AraBERTNER:
    """Classe wrapper pour le modèle AraBERT NER"""
    
    def __init__(self, model_checkpoint: str, num_labels: int,
                 id2label: Dict, label2id: Dict, max_length: int = 128):
        """
        Args:
            model_checkpoint: Nom du modèle pré-entraîné
            num_labels: Nombre de classes NER
            id2label: Mapping id -> label
            label2id: Mapping label -> id
            max_length: Longueur max des séquences
        """
        self.model_checkpoint = model_checkpoint
        self.num_labels = num_labels
        self.id2label = id2label
        self.label2id = label2id
        self.max_length = max_length
        
        self.tokenizer = None
        self.model = None
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Charge le tokenizer et le modèle"""
        print(f"\n🤖 Chargement du modèle: {self.model_checkpoint}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_checkpoint,
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id
        )
        
        print(f"   ✅ Modèle chargé avec {self.num_labels} classes")
    
    def tokenize_and_align_labels(self, examples: Dict) -> Dict:
        """
        Tokenize et aligne les labels avec les subword tokens
        
        Args:
            examples: Batch d'exemples avec tokens et ner_tags
            
        Returns:
            Inputs tokenizés avec labels alignés
        """
        tokenized_inputs = self.tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            padding=False,
            max_length=self.max_length
        )
        
        labels = []
        for i, example_labels in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            aligned_labels = []
            
            for word_idx in word_ids:
                if word_idx is None:
                    aligned_labels.append(-100)
                elif word_idx != previous_word_idx:
                    aligned_labels.append(example_labels[word_idx])
                else:
                    aligned_labels.append(-100)
                
                previous_word_idx = word_idx
            
            labels.append(aligned_labels)
        
        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    
    def predict(self, text: str) -> List[Tuple[str, str]]:
        """
        Prédiction NER sur un texte
        
        Args:
            text: Texte en arabe
            
        Returns:
            Liste de (token, label)
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        predictions = torch.argmax(outputs.logits, dim=2)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        labels_pred = [self.id2label[p.item()] for p in predictions[0]]
        
        return [
            (tok, lab) for tok, lab in zip(tokens, labels_pred)
            if tok not in ["[CLS]", "[SEP]", "[PAD]"]
        ]
    
    def save(self, output_path: str) -> None:
        """Sauvegarde le modèle et le tokenizer"""
        print(f"\n💾 Sauvegarde du modèle: {output_path}")
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        print("   ✅ Modèle sauvegardé")
    
    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str) -> 'AraBERTNER':
        """Charge un modèle depuis un checkpoint"""
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        model = AutoModelForTokenClassification.from_pretrained(checkpoint_path)
        
        instance = cls.__new__(cls)
        instance.tokenizer = tokenizer
        instance.model = model
        instance.id2label = model.config.id2label
        instance.label2id = model.config.label2id
        instance.num_labels = model.config.num_labels
        
        return instance