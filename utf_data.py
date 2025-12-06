import json

input_file = '/mnt/c/Users/saifa/msr_project/ANERCorp_json/train.json'
output_file = '/mnt/c/Users/saifa/msr_project/ANERCorp_utf8_train.json'

data = []

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            data.append(json.loads(line))

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("✅ Conversion réussie !")
