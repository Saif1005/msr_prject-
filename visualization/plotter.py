"""
Visualisation des métriques d'entraînement et d'évaluation
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import json


class MetricsPlotter:
    """Classe pour visualiser les métriques NER"""
    
    def __init__(self, output_dir: str = "./plots"):
        """
        Args:
            output_dir: Dossier de sauvegarde des plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Style des plots
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['font.size'] = 10
        
    def plot_training_history(self, trainer, save: bool = True) -> None:
        """
        Plot l'historique d'entraînement (loss, learning rate)
        
        Args:
            trainer: Instance Trainer avec log_history
            save: Sauvegarder le plot
        """
        log_history = trainer.state.log_history
        
        # Extraire les métriques
        train_loss = []
        eval_loss = []
        learning_rates = []
        steps = []
        eval_steps = []
        
        for log in log_history:
            if 'loss' in log:
                train_loss.append(log['loss'])
                steps.append(log['step'])
            if 'eval_loss' in log:
                eval_loss.append(log['eval_loss'])
                eval_steps.append(log['step'])
            if 'learning_rate' in log:
                learning_rates.append(log['learning_rate'])
        
        # Créer la figure avec 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Training & Validation Loss
        ax1.plot(steps, train_loss, label='Train Loss', color='#2E86DE', linewidth=2)
        if eval_loss:
            ax1.plot(eval_steps, eval_loss, label='Validation Loss', 
                    color='#EE5A6F', linewidth=2, marker='o', markersize=5)
        ax1.set_xlabel('Steps', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax1.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Learning Rate
        if learning_rates:
            ax2.plot(steps[:len(learning_rates)], learning_rates, 
                    color='#10AC84', linewidth=2)
            ax2.set_xlabel('Steps', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
            ax2.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / "training_history.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   📊 Training history sauvegardé: {save_path}")
        
        plt.show()
    
    def plot_evaluation_metrics(self, metrics: Dict, save: bool = True) -> None:
        """
        Plot les métriques d'évaluation (Precision, Recall, F1, Accuracy)
        
        Args:
            metrics: Dict avec les métriques
            save: Sauvegarder le plot
        """
        metric_names = ['Precision', 'Recall', 'F1-Score', 'Accuracy']
        metric_values = [
            metrics.get('eval_precision', 0),
            metrics.get('eval_recall', 0),
            metrics.get('eval_f1', 0),
            metrics.get('eval_accuracy', 0)
        ]
        
        # Couleurs
        colors = ['#2E86DE', '#EE5A6F', '#10AC84', '#F79F1F']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(metric_names, metric_values, color=colors, alpha=0.8, edgecolor='black')
        
        # Ajouter les valeurs sur les barres
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}',
                   ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Evaluation Metrics on Test Set', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / "evaluation_metrics.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   📊 Evaluation metrics sauvegardé: {save_path}")
        
        plt.show()
    
    def plot_per_class_metrics(self, classification_report: str, 
                               save: bool = True) -> None:
        """
        Plot les métriques par classe NER
        
        Args:
            classification_report: Rapport de seqeval (string)
            save: Sauvegarder le plot
        """
        # Parser le rapport
        lines = classification_report.strip().split('\n')
        data = []
        
        for line in lines[1:]:  # Skip header
            parts = line.split()
            if len(parts) >= 4 and parts[0] not in ['micro', 'macro', 'weighted']:
                try:
                    entity = parts[0]
                    precision = float(parts[1])
                    recall = float(parts[2])
                    f1 = float(parts[3])
                    data.append({
                        'Entity': entity,
                        'Precision': precision,
                        'Recall': recall,
                        'F1-Score': f1
                    })
                except (ValueError, IndexError):
                    continue
        
        if not data:
            print("   ⚠️  Pas de données par classe à visualiser")
            return
        
        df = pd.DataFrame(data)
        
        # Trier par F1-Score
        df = df.sort_values('F1-Score', ascending=True)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, max(8, len(df) * 0.4)))
        
        x = np.arange(len(df))
        width = 0.25
        
        bars1 = ax.barh(x - width, df['Precision'], width, 
                       label='Precision', color='#2E86DE', alpha=0.8)
        bars2 = ax.barh(x, df['Recall'], width, 
                       label='Recall', color='#EE5A6F', alpha=0.8)
        bars3 = ax.barh(x + width, df['F1-Score'], width, 
                       label='F1-Score', color='#10AC84', alpha=0.8)
        
        ax.set_yticks(x)
        ax.set_yticklabels(df['Entity'])
        ax.set_xlabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Per-Class NER Metrics', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.set_xlim([0, 1.1])
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / "per_class_metrics.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   📊 Per-class metrics sauvegardé: {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix_per_entity(self, true_labels: List[List[str]], 
                                         pred_labels: List[List[str]],
                                         save: bool = True) -> None:
        """
        Plot une matrice de confusion pour les entités NER
        
        Args:
            true_labels: Labels vrais (nested list)
            pred_labels: Labels prédits (nested list)
            save: Sauvegarder le plot
        """
        from sklearn.metrics import confusion_matrix
        
        # Flatten les listes
        y_true = [label for sublist in true_labels for label in sublist]
        y_pred = [label for sublist in pred_labels for label in sublist]
        
        # Extraire les entités uniques
        unique_labels = sorted(list(set(y_true + y_pred)))
        
        # Calculer la matrice de confusion
        cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
        
        # Normaliser
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)  # Remplacer NaN par 0
        
        # Plot
        fig, ax = plt.subplots(figsize=(max(12, len(unique_labels)), 
                                        max(10, len(unique_labels))))
        
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=unique_labels, yticklabels=unique_labels,
                   cbar_kws={'label': 'Normalized Frequency'}, ax=ax)
        
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / "confusion_matrix.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   📊 Confusion matrix sauvegardée: {save_path}")
        
        plt.show()
    
    def plot_label_distribution(self, dataset, label_list: List[str], 
                               save: bool = True) -> None:
        """
        Plot la distribution des labels dans le dataset
        
        Args:
            dataset: DatasetDict Hugging Face
            label_list: Liste des labels
            save: Sauvegarder le plot
        """
        from collections import Counter
        
        # Compter les labels pour train/val/test
        splits = ['train', 'validation', 'test']
        label_counts = {split: Counter() for split in splits}
        
        for split in splits:
            for example in dataset[split]:
                for tag_id in example['ner_tags']:
                    if tag_id != -100:
                        label_counts[split][tag_id] += 1
        
        # Préparer les données
        data = []
        for split in splits:
            for label_id, count in label_counts[split].items():
                data.append({
                    'Split': split.capitalize(),
                    'Label': label_list[label_id],
                    'Count': count
                })
        
        df = pd.DataFrame(data)
        
        # Pivoter pour le plot
        df_pivot = df.pivot(index='Label', columns='Split', values='Count').fillna(0)
        
        # Plot
        fig, ax = plt.subplots(figsize=(14, max(8, len(label_list) * 0.3)))
        
        df_pivot.plot(kind='barh', ax=ax, color=['#2E86DE', '#EE5A6F', '#10AC84'], 
                     alpha=0.8, edgecolor='black')
        
        ax.set_xlabel('Count', fontsize=12, fontweight='bold')
        ax.set_ylabel('NER Label', fontsize=12, fontweight='bold')
        ax.set_title('Label Distribution Across Splits', fontsize=14, fontweight='bold')
        ax.legend(title='Split', fontsize=11)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / "label_distribution.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   📊 Label distribution sauvegardée: {save_path}")
        
        plt.show()
    
    def plot_epoch_metrics(self, trainer, save: bool = True) -> None:
        """
        Plot les métriques par epoch
        
        Args:
            trainer: Instance Trainer
            save: Sauvegarder le plot
        """
        log_history = trainer.state.log_history
        
        # Extraire les métriques par epoch
        epochs_data = []
        
        for log in log_history:
            if 'epoch' in log and 'eval_loss' in log:
                epochs_data.append({
                    'Epoch': log['epoch'],
                    'Validation Loss': log.get('eval_loss', None),
                    'Precision': log.get('eval_precision', None),
                    'Recall': log.get('eval_recall', None),
                    'F1-Score': log.get('eval_f1', None),
                    'Accuracy': log.get('eval_accuracy', None)
                })
        
        if not epochs_data:
            print("   ⚠️  Pas de données d'epoch à visualiser")
            return
        
        df = pd.DataFrame(epochs_data)
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Loss par epoch
        ax1.plot(df['Epoch'], df['Validation Loss'], 
                marker='o', color='#EE5A6F', linewidth=2, markersize=8)
        ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Validation Loss', fontsize=12, fontweight='bold')
        ax1.set_title('Validation Loss per Epoch', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Métriques par epoch
        ax2.plot(df['Epoch'], df['Precision'], marker='o', label='Precision', 
                linewidth=2, markersize=6, color='#2E86DE')
        ax2.plot(df['Epoch'], df['Recall'], marker='s', label='Recall', 
                linewidth=2, markersize=6, color='#EE5A6F')
        ax2.plot(df['Epoch'], df['F1-Score'], marker='^', label='F1-Score', 
                linewidth=2, markersize=6, color='#10AC84')
        ax2.plot(df['Epoch'], df['Accuracy'], marker='d', label='Accuracy', 
                linewidth=2, markersize=6, color='#F79F1F')
        
        ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax2.set_title('Evaluation Metrics per Epoch', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1.1])
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / "epoch_metrics.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   📊 Epoch metrics sauvegardées: {save_path}")
        
        plt.show()
    
    def save_metrics_to_json(self, metrics: Dict, filename: str = "metrics.json") -> None:
        """
        Sauvegarde les métriques en JSON
        
        Args:
            metrics: Dictionnaire des métriques
            filename: Nom du fichier
        """
        save_path = self.output_dir / filename
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        print(f"   💾 Métriques sauvegardées: {save_path}")
    
    def generate_all_plots(self, trainer, dataset, metadata: Dict, 
                          classification_report_str: str,
                          true_labels: List[List[str]], 
                          pred_labels: List[List[str]]) -> None:
        """
        Génère tous les plots d'un coup
        
        Args:
            trainer: Instance Trainer
            dataset: DatasetDict
            metadata: Métadonnées du dataset
            classification_report_str: Rapport de classification
            true_labels: Labels vrais
            pred_labels: Labels prédits
        """
        print("\n📊 Génération de toutes les visualisations...")
        
        # 1. Training history
        self.plot_training_history(trainer)
        
        # 2. Epoch metrics
        self.plot_epoch_metrics(trainer)
        
        # 3. Evaluation metrics
        test_results = trainer.evaluate()
        self.plot_evaluation_metrics(test_results)
        
        # 4. Per-class metrics
        self.plot_per_class_metrics(classification_report_str)
        
        # 5. Confusion matrix
        self.plot_confusion_matrix_per_entity(true_labels, pred_labels)
        
        # 6. Label distribution
        self.plot_label_distribution(dataset, metadata['label_list'])
        
        # 7. Save metrics to JSON
        self.save_metrics_to_json(test_results)
        
        print(f"\n✅ Tous les plots sauvegardés dans: {self.output_dir}")