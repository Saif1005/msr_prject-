import json

input_file = '/mnt/c/Users/saifa/msr_project/CANERCorpus.json'
output_file = '/mnt/c/Users/saifa/msr_project/CANERCorpus_utf8.json'

# Lire avec cp1256 (encodage Windows arabe)
with open(input_file, 'r', encoding='cp1256') as f:
    data = json.load(f)

# Sauvegarder en UTF-8
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("✅ Fichier converti en UTF-8")
