import os
import json

# Define the path to the JSON file
json_file_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "available_models.json")
)

# Load the JSON file
with open(json_file_path, "r") as file:
    available_models = json.load(file)

OPENAIMODELS = available_models.get("OPENAIMODELS", {})
ANTHROPICMODELS = available_models.get("ANTHROPICMODELS", {})
