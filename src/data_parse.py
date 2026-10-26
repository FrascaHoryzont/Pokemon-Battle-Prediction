import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# Open the JSONL file and parse each line into a Python dictionary, returning a list of dictionaries.
def parse_jsonl_file(file_path):
  with open(file_path, "r", encoding="utf-8") as file:
    data = []
    for line in tqdm(file, desc="Loading Battles "):
      line = line.strip()
      if line:  # Check if the line is not empty
        data.append(json.loads(line))

  return data

# Load the type chart from a CSV file and return it as a dictionary.
def load_type_chart(csv_path=Path("type_chart_gen2.csv")):
  df = pd.read_csv(csv_path)
  chart = {}
  for _, row in df.iterrows():
    chart[(row["attacking_type"], row["defending_type"])] = float(
        row["multiplier"]
    )
  return chart

# Calculate the damage multiplier based on the attacking type and defending types using the type chart.
def get_type_multiplier(atk_type, def_types, type_chart):
  if not atk_type or atk_type in ["notype", "threequestionmarks"]:
    return 1.0
  mult = 1.0
  for d_type in def_types:
    if d_type and d_type != "notype":
      mult *= type_chart.get((atk_type, d_type), 1.0)
  return mult


if __name__ == "__main__":
  file_path = Path("data/train.jsonl")
  parsed_data = parse_jsonl_file(file_path)
