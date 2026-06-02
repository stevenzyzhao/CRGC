import re
from llm_client import llm_client
from config import EVAL_MODEL, TAU

SATISFACTION_EVAL_PROMPT = """You act as an impartial, deterministic evaluation scoring algorithm. Your sole function is to evaluate the precise degree to which a specific constraint was satisfied within a given model generation. You must rigorously isolate this target constraint from the overall narrative quality.

Target Constraint: {constraint}
Constraint Nature: {constraint_nature}
Model Generation: {output_text}

Evaluate the adherence to the target constraint. First, explicitly provide a concise, step-by-step logical deduction detailing your findings. Second, assign a final satisfaction score. If the constraint nature is 'Binary', you must output strictly 0.0 (complete violation) or 1.0 (perfect adherence). If the nature is 'Continuous', output a precise float value between 0.0 and 1.0 reflecting the exact degree of partial or full satisfaction. Your final line must be exactly formatted as: "SCORE: [float]"."""


def classify_constraint_nature(constraint):
    deterministic_keywords = [
        "exactly", "word count", "words", "characters", "sentences",
        "paragraphs", "bullet", "format", "json", "markdown",
        "enclose", "quotation", "uppercase", "lowercase", "number of",
        "at least", "at most", "no more than", "minimum", "maximum",
        "include the word", "start with", "end with", "contain",
    ]
    constraint_lower = constraint.lower()
    for kw in deterministic_keywords:
        if kw in constraint_lower:
            return "Binary"
    return "Continuous"


def evaluate_constraint_deterministic(constraint, output):
    constraint_lower = constraint.lower()

    word_match = re.search(r"exactly (\d+) words?", constraint_lower)
    if word_match:
        target = int(word_match.group(1))
        actual = len(output.split())
        return 1.0 if actual == target else 0.0

    max_words_match = re.search(r"(?:no more than|at most|within|under|maximum) (\d+) words?", constraint_lower)
    if max_words_match:
        target = int(max_words_match.group(1))
        actual = len(output.split())
        return 1.0 if actual <= target else 0.0

    min_words_match = re.search(r"(?:at least|minimum) (\d+) words?", constraint_lower)
    if min_words_match:
        target = int(min_words_match.group(1))
        actual = len(output.split())
        return 1.0 if actual >= target else 0.0

    keyword_match = re.search(r"include the (?:word|phrase) [\"'](.+?)[\"']", constraint_lower)
    if keyword_match:
        keyword = keyword_match.group(1)
        return 1.0 if keyword.lower() in output.lower() else 0.0

    if "enclose" in constraint_lower and "quotation" in constraint_lower:
        return 1.0 if output.startswith('"') and output.endswith('"') else 0.0

    if "json" in constraint_lower and "format" in constraint_lower:
        try:
            import json
            json.loads(output)
            return 1.0
        except (json.JSONDecodeError, ValueError):
            return 0.0

    if "uppercase" in constraint_lower:
        return 1.0 if output == output.upper() else 0.0

    if "lowercase" in constraint_lower:
        return 1.0 if output == output.lower() else 0.0

    bullet_match = re.search(r"(\d+) bullet", constraint_lower)
    if bullet_match:
        target = int(bullet_match.group(1))
        actual = len(re.findall(r"^[\-\*•]", output, re.MULTILINE))
        return 1.0 if actual == target else 0.0

    paragraph_match = re.search(r"(\d+) paragraphs?", constraint_lower)
    if paragraph_match:
        target = int(paragraph_match.group(1))
        actual = len([p for p in output.split("\n\n") if p.strip()])
        return 1.0 if actual == target else 0.0

    return None


def evaluate_constraint_llm(constraint, output, eval_model=None):
    if eval_model is None:
        eval_model = EVAL_MODEL
    nature = classify_constraint_nature(constraint)
    prompt = SATISFACTION_EVAL_PROMPT.format(
        constraint=constraint,
        constraint_nature=nature,
        output_text=output
    )
    response = llm_client.generate(eval_model, prompt, temperature=0.0)
    score_match = re.search(r"SCORE:\s*([\d.]+)", response)
    if score_match:
        return float(score_match.group(1))
    return 0.0


def evaluate_constraint(constraint, output, eval_model=None):
    deterministic_score = evaluate_constraint_deterministic(constraint, output)
    if deterministic_score is not None:
        return deterministic_score
    return evaluate_constraint_llm(constraint, output, eval_model)


def compute_csr(constraints, output, eval_model=None):
    scores = [evaluate_constraint(c, output, eval_model) for c in constraints]
    satisfied = sum(1 for s in scores if s >= TAU)
    return satisfied / len(constraints) if constraints else 0.0


def compute_wcs(constraints, output, eval_model=None):
    scores = [evaluate_constraint(c, output, eval_model) for c in constraints]
    return sum(scores) / len(scores) if scores else 0.0


def compute_tcr(constraints, output, eval_model=None):
    scores = [evaluate_constraint(c, output, eval_model) for c in constraints]
    return min(scores) if scores else 0.0


def evaluate_all_metrics(constraints, output, eval_model=None):
    scores = [evaluate_constraint(c, output, eval_model) for c in constraints]
    csr = sum(1 for s in scores if s >= TAU) / len(scores) if scores else 0.0
    wcs = sum(scores) / len(scores) if scores else 0.0
    tcr = min(scores) if scores else 0.0
    return {"csr": csr, "wcs": wcs, "tcr": tcr, "scores": scores}
