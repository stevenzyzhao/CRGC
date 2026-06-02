import re
from llm_client import llm_client


def evaluate_mmlu(model_name, instructions, method_fn):
    correct = 0
    total = 0
    for item in instructions:
        result = method_fn(item["instruction"])
        output = result["output"] if isinstance(result, dict) else result
        predicted = extract_choice(output)
        expected = item["answer"]
        if isinstance(expected, int):
            expected = chr(65 + expected)
        if predicted.upper() == str(expected).upper():
            correct += 1
        total += 1
    return correct / total if total > 0 else 0.0


def evaluate_gsm8k(model_name, instructions, method_fn):
    correct = 0
    total = 0
    for item in instructions:
        prompt = f"{item['instruction']}\n\nSolve step by step and give the final numerical answer."
        result = method_fn(prompt)
        output = result["output"] if isinstance(result, dict) else result
        predicted = extract_number(output)
        expected = extract_number(item["answer"])
        if predicted is not None and expected is not None and abs(predicted - expected) < 1e-5:
            correct += 1
        total += 1
    return correct / total if total > 0 else 0.0


def evaluate_bbh(model_name, instructions, method_fn):
    correct = 0
    total = 0
    for item in instructions:
        result = method_fn(item["instruction"])
        output = result["output"] if isinstance(result, dict) else result
        predicted = output.strip().lower()
        expected = item["answer"].strip().lower()
        if expected in predicted or predicted == expected:
            correct += 1
        total += 1
    return correct / total if total > 0 else 0.0


def extract_choice(text):
    text = text.strip()
    match = re.search(r"\b([A-D])\b", text)
    if match:
        return match.group(1)
    if text and text[0] in "ABCD":
        return text[0]
    return ""


def extract_number(text):
    if text is None:
        return None
    text = str(text).replace(",", "")
    numbers = re.findall(r"-?\d+\.?\d*", text)
    if numbers:
        return float(numbers[-1])
    return None
