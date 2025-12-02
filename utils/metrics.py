"""
Métriques d'évaluation pour NER
"""
import numpy as np
import evaluate
from typing import Dict, List


class NERMetrics:
    """Classe pour calculer les métriques NER"""
    
    def __init__(self, id2label: Dict):
        """
        Args:
            id2label: Mapping id -> label
        """
        self.id2label = id2label
        self.seqeval = evaluate.load("seqeval")
    
    def compute_metrics(self, eval_preds) -> Dict:
        """
        Calcule les métriques seqeval
        
        Args:
            eval_preds: Tuple (predictions, labels)
            
        Returns:
            Dict avec precision, recall, f1, accuracy
        """
        predictions, labels = eval_preds
        predictions = np.argmax(predictions, axis=2)
        
        true_predictions = [
            [self.id2label[p] for (p, l) in zip(pred, label) if l != -100]
            for pred, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.id2label[l] for (p, l) in zip(pred, label) if l != -100]
            for pred, label in zip(predictions, labels)
        ]
        
        results = self.seqeval.compute(
            predictions=true_predictions,
            references=true_labels
        )
        
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"]
        }
    
    @staticmethod
    def get_predictions_and_labels(predictions: np.ndarray, 
                                   labels: np.ndarray,
                                   id2label: Dict) -> tuple:
        """
        Extrait les prédictions et labels vrais
        
        Returns:
            Tuple (true_predictions, true_labels)
        """
        predictions = np.argmax(predictions, axis=2)
        
        true_predictions = [
            [id2label[p] for (p, l) in zip(pred, label) if l != -100]
            for pred, label in zip(predictions, labels)
        ]
        true_labels = [
            [id2label[l] for (p, l) in zip(pred, label) if l != -100]
            for pred, label in zip(predictions, labels)
        ]
        
        return true_predictions, true_labels