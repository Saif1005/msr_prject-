"""
Script principal pour le fine-tuning AraBERT NER
"""
from config.config import Config
from data.dataset_loader import NERDatasetLoader
from models.ner_model import AraBERTNER
from training.trainer import NERTrainer
from evaluation.evaluator import NEREvaluator
from visualization.plotter import MetricsPlotter  # 🆕
from utils.metrics import NERMetrics  # 🆕
import numpy as np


def main():
    """Pipeline complet de fine-tuning"""
    
    print("="*60)
    print("   ARABERT FINE-TUNING — CLASSICAL ARABIC NER")
    print("="*60)
    
    # 1. Chargement de la configuration
    config = Config()
    
    # 2. Chargement et préparation des données
   # 2. Chargement et préparation des données
    print("\n=== Préparation du dataset NER ===")

    dataset_loader = NERDatasetLoader(
        data_path=config.data.data_path,
        train_split=config.data.train_split,
        val_split=config.data.val_split,
        random_state=config.data.random_state
    )

    dataset, metadata = dataset_loader.load_and_prepare()

    print("\n=== Informations Metadata ===")
    print(f"Labels        : {metadata['label_list']}")
    print(f"Nombre labels : {metadata['num_labels']}")
    print(f"Train size    : {metadata['train_size']}")
    print(f"Val size      : {metadata['val_size']}")
    print(f"Test size     : {metadata['test_size']}")

    
    # 3. Chargement du modèle AraBERT
    model = AraBERTNER(
        model_checkpoint=config.model.model_checkpoint,
        num_labels=metadata['num_labels'],
        id2label=metadata['id2label'],
        label2id=metadata['label2id'],
        max_length=config.model.max_length
    )
    
    # 4. Tokenization
    print("\n🔤 Tokenization des données...")
    tokenized_dataset = dataset.map(
        model.tokenize_and_align_labels,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    print("   ✅ Tokenization terminée")
    
    # 5. Configuration du Trainer
    trainer = NERTrainer(model, config.training)
    trainer.setup_trainer(tokenized_dataset)
    
    # 6. Entraînement
    trainer.train()
    
    # 7. Évaluation sur le test set
    test_results = trainer.evaluate(dataset_split="test")
    
    # 8. Rapport de classification détaillé
    evaluator = NEREvaluator(trainer.trainer, metadata['id2label'])
    report = evaluator.generate_classification_report(tokenized_dataset)
    evaluator.save_report(report, "./classification_report.txt")
    
    # 🆕 9. GÉNÉRATION DES VISUALISATIONS
    print("\n" + "="*60)
    print("   GÉNÉRATION DES VISUALISATIONS")
    print("="*60)
    
    plotter = MetricsPlotter(output_dir="./plots")
    
    # Extraire true_labels et pred_labels pour la confusion matrix
    predictions, labels, _ = trainer.trainer.predict(tokenized_dataset["test"])
    true_predictions, true_labels = NERMetrics.get_predictions_and_labels(
        predictions, labels, metadata['id2label']
    )
    
    # Générer tous les plots
    plotter.generate_all_plots(
        trainer=trainer.trainer,
        dataset=dataset,
        metadata=metadata,
        classification_report_str=report,
        true_labels=true_labels,
        pred_labels=true_predictions
    )
    
    # 10. Sauvegarde du modèle final
    model.save(config.final_model_path)
    
    # 11. Test d'inférence
    print("\n🔍 Test d'inférence:")
    test_text = "قال رسول الله صلى الله عليه وسلم"
    predictions = model.predict(test_text)
    for token, label in predictions:
        print(f"   {token}: {label}")
    
    print("\n" + "="*60)
    print("   ✅ PIPELINE TERMINÉ AVEC SUCCÈS")
    print("="*60)


if __name__ == "__main__":
    main()