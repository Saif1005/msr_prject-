"""
Classe pour l'entraînement du modèle NER
"""
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from datasets import DatasetDict
from typing import Dict
import sys
sys.path.append('..')
from models.ner_model import AraBERTNER
from utils.metrics import NERMetrics


class NERTrainer:
    """Classe wrapper pour l'entraînement"""
    
    def __init__(self, model: AraBERTNER, config: 'TrainingConfig'):
        """
        Args:
            model: Instance AraBERTNER
            config: Configuration d'entraînement
        """
        self.model = model
        self.config = config
        
        self.metrics_calculator = NERMetrics(model.id2label)
        self.data_collator = DataCollatorForTokenClassification(
            tokenizer=model.tokenizer
        )
        
        self.trainer = None
        
    def setup_trainer(self, tokenized_dataset: DatasetDict) -> None:
        """Configure le Trainer Hugging Face"""
        
        
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            num_train_epochs=self.config.num_train_epochs,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            save_total_limit=self.config.save_total_limit,
            fp16=self.config.fp16,
            seed=self.config.seed,
            logging_dir="./runs/ner_tensorboard"
        )

        self.trainer = Trainer(
            model=self.model.model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            tokenizer=self.model.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.metrics_calculator.compute_metrics
        )
    
    def train(self) -> None:
        """Lance l'entraînement"""
        print("\n🚀 Démarrage de l'entraînement...")
        self.trainer.train()
        print("   ✅ Entraînement terminé")
    
    def evaluate(self, dataset_split: str = "test") -> Dict:
        """
        Évalue le modèle
        
        Args:
            dataset_split: 'validation' ou 'test'
            
        Returns:
            Dictionnaire des métriques
        """
        print(f"\n📊 Évaluation sur {dataset_split}...")
        results = self.trainer.evaluate()
        print(f"   Results: {results}")
        return results