import json
from tqdm import tqdm
from pathlib import Path

def parse_jsonl_file(file_path):
    """
    Parses a JSONL file and returns the data as a list of Python objects.

    Args:
        file_path (str): The path to the JSON file.
    """
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
    print(f"Battles loaded: {len(parsed_data)}")
    print(f"Player 1: {parsed_data[0]['p1_team_details']}")
    print(f"Player 2: {parsed_data[0]['p2_lead_details']['name']}")