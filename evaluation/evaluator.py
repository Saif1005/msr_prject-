"""
Évaluation détaillée du modèle NER
"""
import numpy as np
from seqeval.metrics import classification_report
from typing import Dict
from datasets import DatasetDict


class NEREvaluator:
    """Classe pour l'évaluation détaillée"""
    
    def __init__(self, trainer, id2label: Dict):
        """
        Args:
            trainer: Instance Trainer
            id2label: Mapping id -> label
        """
        self.trainer = trainer
        self.id2label = id2label
    
    def generate_classification_report(self, 
                                       tokenized_dataset: DatasetDict) -> str:
        """
        Génère un rapport de classification détaillé
        
        Args:
            tokenized_dataset: Dataset tokenizé
            
        Returns:
            Rapport de classification (string)
        """
        print("\n📈 Génération du rapport de classification...")
        
        predictions, labels, _ = self.trainer.predict(tokenized_dataset["test"])
        predictions = np.argmax(predictions, axis=2)
        
        true_predictions = [
            [self.id2label[p] for (p, l) in zip(pred, label) if l != -100]
            for pred, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.id2label[l] for (p, l) in zip(pred, label) if l != -100]
            for pred, label in zip(predictions, labels)
        ]
        
        report = classification_report(true_labels, true_predictions)
        print(report)
        
        return report
    
    def save_report(self, report: str, output_path: str) -> None:
        """Sauvegarde le rapport"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"   ✅ Rapport sauvegardé: {output_path}")