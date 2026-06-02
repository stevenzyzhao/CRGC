import json
import urllib.request
from datasets import load_dataset


def load_ifeval():
    dataset = load_dataset("google/IFEval", split="train")
    instructions = []
    for item in dataset:
        instructions.append({
            "id": item["key"],
            "instruction": item["prompt"],
            "instruction_id_list": item["instruction_id_list"],
            "kwargs": item["kwargs"],
        })
    return instructions


def load_complexbench():
    url = "https://raw.githubusercontent.com/thu-coai/ComplexBench/main/data/data_final.json"
    response = urllib.request.urlopen(url)
    data = json.loads(response.read().decode())
    instructions = []
    for item in data:
        instruction_text = item.get("instruction_en", item.get("instruction", ""))
        instructions.append({
            "id": item.get("main_id", 0),
            "instruction": instruction_text,
            "constraint_dimensions": item.get("constraint_dimensions", []),
            "scoring_questions": item.get("scoring_questions", []),
        })
    return instructions


def load_followbench():
    dataset = load_dataset("YuxinJiang/FollowBench", split="train")
    instructions = []
    for item in dataset:
        instructions.append({
            "id": item["example_id"],
            "instruction": item["instruction"],
            "category": item["category"],
            "level": item["level"],
            "target": item.get("target", ""),
        })
    return instructions


def load_mmlu(n_samples=200):
    dataset = load_dataset("cais/mmlu", "all", split="test")
    instructions = []
    for i, item in enumerate(dataset):
        if i >= n_samples:
            break
        choices_str = "\n".join(
            f"{chr(65+j)}. {c}" for j, c in enumerate(item["choices"])
        )
        instruction = f"{item['question']}\n\n{choices_str}\n\nAnswer with just the letter."
        instructions.append({
            "id": i,
            "instruction": instruction,
            "subject": item["subject"],
            "answer": item["answer"],
        })
    return instructions


def load_gsm8k(n_samples=200):
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    instructions = []
    for i, item in enumerate(dataset):
        if i >= n_samples:
            break
        answer_text = item["answer"]
        final_answer = answer_text.split("####")[-1].strip() if "####" in answer_text else ""
        instructions.append({
            "id": i,
            "instruction": item["question"],
            "answer": final_answer,
            "full_answer": answer_text,
        })
    return instructions


def load_bbh(n_samples=200):
    tasks = [
        "boolean_expressions", "causal_judgement", "date_understanding",
        "disambiguation_qa", "formal_fallacies", "geometric_shapes",
        "hyperbaton", "logical_deduction_five_objects", "movie_recommendation",
        "multistep_arithmetic_two", "navigate", "object_counting",
        "penguins_in_a_table", "reasoning_about_colored_objects",
        "snarks", "sports_understanding", "temporal_sequences",
        "tracking_shuffled_objects_three_objects", "web_of_lies", "word_sorting",
    ]
    instructions = []
    per_task = max(1, n_samples // len(tasks))
    for task_name in tasks:
        try:
            dataset = load_dataset("lukaemon/bbh", task_name, split="test")
            for i, item in enumerate(dataset):
                if i >= per_task:
                    break
                instructions.append({
                    "id": len(instructions),
                    "instruction": item["input"],
                    "answer": item["target"],
                    "task": task_name,
                })
        except Exception:
            continue
    return instructions[:n_samples]


def load_all_if_datasets():
    return {
        "ifeval": load_ifeval(),
        "complexbench": load_complexbench(),
        "followbench": load_followbench(),
    }


def load_all_reasoning_datasets(n_samples=200):
    return {
        "mmlu": load_mmlu(n_samples),
        "gsm8k": load_gsm8k(n_samples),
        "bbh": load_bbh(n_samples),
    }
