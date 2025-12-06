from transformers import AutoTokenizer, AutoModelForTokenClassification

class ModelSingleton:
    _instance = None

    def __new__(cls, model_checkpoint, num_labels, id2label, label2id):
        # Si le modèle n'est pas encore chargé → on le charge une seule fois
        if cls._instance is None:
            print("\n Singleton: Chargement du modèle AraBERT une seule fois...")

            cls._instance = super().__new__(cls)
            cls._instance.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

            cls._instance.model = AutoModelForTokenClassification.from_pretrained(
                model_checkpoint,
                num_labels=num_labels,
                id2label=id2label,
                label2id=label2id
            )

            print(" Modèle AraBERT chargé et stocké dans Singleton.\n")
        
        # Sinon → on renvoie la même instance déjà chargée
        else:
            print(" Singleton: Modèle déjà chargé → réutilisation.")

        return cls._instance
