"""
Script principal pour le fine-tuning AraBERT NER (multi-source)
"""

from config.config import Config
from data.dataset_loader import MultiSourceNERLoader   # ⭐ change ici
from models.ner_model import AraBERTNER
from training.trainer import NERTrainer
from evaluation.evaluator import NEREvaluator
from visualization.plotter import MetricsPlotter
from utils.metrics import NERMetrics
import numpy as np


def main():
    """Pipeline complet de fine-tuning"""

    print("=" * 60)
    print("      ARABERT FINE-TUNING — MULTI-SOURCE NER")
    print("=" * 60)

    # ----------------------------------------------------------
    # 1️⃣ Charger la configuration globale
    # ----------------------------------------------------------
    config = Config()

    # ----------------------------------------------------------
    # 2️⃣ Chargement & préparation du dataset multi-source
    # ----------------------------------------------------------
    print("\n=== Préparation du dataset NER ===")

    dataset_loader = MultiSourceNERLoader(
        paths=config.data.data_paths,
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

    # ----------------------------------------------------------
    # 3️⃣ Chargement du modèle AraBERT
    # ----------------------------------------------------------
    model = AraBERTNER(
        model_checkpoint=config.model.model_checkpoint,
        num_labels=metadata["num_labels"],
        id2label=metadata["id2label"],
        label2id=metadata["label2id"],
        max_length=config.model.max_length
    )

    # ----------------------------------------------------------
    # 4️⃣ Tokenization + alignement labels
    # ----------------------------------------------------------
    print("\n=== Tokenization des données ===")

    tokenized_dataset = model.prepare_dataset(dataset)   # ⭐ use méthode interne
    print("    Tokenization terminée")

    # ----------------------------------------------------------
    # 5️⃣ Configuration du Trainer HF
    # ----------------------------------------------------------
    trainer = NERTrainer(model, config.training)
    trainer.setup_trainer(tokenized_dataset)

    # ----------------------------------------------------------
    # 6️⃣ Entraînement
    # ----------------------------------------------------------
    trainer.train()

    # ----------------------------------------------------------
    # 7️⃣ Évaluation : test set
    # ----------------------------------------------------------
    test_results = trainer.evaluate(dataset_split="test")


    # ----------------------------------------------------------
    # 8️⃣ Rapport Classification (seqeval)
    # ----------------------------------------------------------
    print("\n=== Rapport détaillé ===")

    evaluator = NEREvaluator(trainer.trainer, metadata["id2label"])
    report = evaluator.generate_classification_report(tokenized_dataset)
    evaluator.save_report(report, "./classification_report.txt")

    # ----------------------------------------------------------
    # 9️⃣ Visualisations (courbes, heatmaps, cm)
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("        GÉNÉRATION DES VISUALISATIONS")
    print("=" * 60)

    plotter = MetricsPlotter(output_dir="./plots")

    predictions, labels, _ = trainer.trainer.predict(tokenized_dataset["test"])

    true_predictions, true_labels = NERMetrics.get_predictions_and_labels(
        predictions, labels, metadata["id2label"]
    )

    plotter.generate_all_plots(
        trainer=trainer.trainer,
        dataset=dataset,
        metadata=metadata,
        classification_report_str=report,
        true_labels=true_labels,
        pred_labels=true_predictions
    )

    # ----------------------------------------------------------
    # 🔟 Sauvegarde modèle final
    # ----------------------------------------------------------
    model.save(config.final_model_path)

    print("\n🎉 Pipeline complet terminé avec succès !")
    print(f"📁 Modèle enregistré dans : {config.final_model_path}")


if __name__ == "__main__":
    main()
