"""
Configuration centrale pour le fine-tuning AraBERT
"""
from dataclasses import dataclass, field
from typing import Optional
import torch

@dataclass
class ModelConfig:
    """Configuration du modèle"""
    model_checkpoint: str = "aubmindlab/bert-base-arabertv02"
    max_length: int = 128
    
@dataclass
class TrainingConfig:
    """Configuration de l'entraînement"""
    output_dir: str = "./arabert-ner-hadith"
    learning_rate: float = 3e-5
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 32
    num_train_epochs: int = 4
    warmup_steps: int = 500
    logging_steps: int = 100
    save_steps: int = 1000
    eval_steps: int = 1000
    save_total_limit: int = 2
    fp16: bool = torch.cuda.is_available()
    seed: int = 42
    
@dataclass
class DataConfig:
    """Configuration des données"""
    data_path: str = "/mnt/c/Users/saifa/msr_project/CANERCorpus_utf8.json"
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    random_state: int = 42
    
@dataclass
class Config:
    """Configuration globale"""
    model: ModelConfig = field(default_factory=ModelConfig)  # 🔧 FIX
    training: TrainingConfig = field(default_factory=TrainingConfig)  # 🔧 FIX
    data: DataConfig = field(default_factory=DataConfig)  # 🔧 FIX
    final_model_path: str = "./arabert-ner-hadith-final"