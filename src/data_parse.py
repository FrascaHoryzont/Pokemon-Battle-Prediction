import json
from tqdm import tqdm
from pathlib import Path

def parse_jsonl_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = []
        for line in tqdm(file, desc="Loadiing Battles "):
            line=line.strip()
            if line:  # Check if the line is not empty
                data.append(json.loads(line))

    return data

if __name__ == "__main__":
    file_path = Path("data/train.jsonl") 
    parsed_data = parse_jsonl_file(file_path)
