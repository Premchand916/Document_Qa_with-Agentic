import pandas as pd

def analyze_excel(file_path, query):

    df = pd.read_excel(file_path)

    result = {}

    if "sum" in query.lower():
        result["sum"] = df.sum(numeric_only=True).to_dict()

    if "average" in query.lower():
        result["average"] = df.mean(numeric_only=True).to_dict()

    if "max" in query.lower():
        result["max"] = df.max(numeric_only=True).to_dict()

    if "min" in query.lower():
        result["min"] = df.min(numeric_only=True).to_dict()

    return result