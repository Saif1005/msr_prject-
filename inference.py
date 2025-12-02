import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

class ArabicNER:
    def __init__(self, model_path):
        # Charger tokenizer + modèle fine-tuné
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)

        # Charger mapping id2label
        self.id2label = self.model.config.id2label

    def predict(self, text):
        # Préparation du texte
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        # Forward du modèle
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Logits → ids prédits
        predictions = torch.argmax(outputs.logits, dim=2)[0].tolist()

        # Conversion des tokens
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        # Sortie formatée
        result = []
        for tok, pred in zip(tokens, predictions):
            if tok not in ["[CLS]", "[SEP]"]:
                label = self.id2label[pred]
                result.append((tok, label))

        return result


if __name__ == "__main__":
    print("Chargement du modèle fine-tuné...")

    #  PATH vers ton modèle final
    model_path = "./arabert-ner-hadith-final"

    ner_model = ArabicNER(model_path)

    print("\n Test d'inférence:")
    test_text = "قال رسول الله صلى الله عليه وسلم"

    predictions = ner_model.predict(test_text)

    for token, label in predictions:
        print(f"   {token} → {label}")
