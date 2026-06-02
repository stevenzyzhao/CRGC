from llm_client import llm_client
from config import EXTRACTION_MODEL

EXTRACTION_PROMPT = """You are an expert computational linguist. Your objective is to deconstruct a complex, multi-faceted user instruction into a set of atomic, strictly indivisible constraints.

Input Instruction: {instruction}

Analyze the text and extract all explicit and implicit constraints. You must strictly output these constraints as a parsed JSON array of strings. Do not modify the original semantic intent. You must categorically separate quantitative bounds (e.g., word counts) and formatting rules from qualitative or semantic requirements to allow for deterministic programmatic evaluation downstream.

Output format: a JSON array of strings, e.g. ["constraint 1", "constraint 2", ...]"""

TASK_EXTRACTION_PROMPT = """Given the following instruction, identify the core task objective (the main thing the user wants accomplished, separate from any constraints on how to do it).

Instruction: {instruction}

Output only the core task objective as a single sentence."""


def extract_constraints(instruction):
    prompt = EXTRACTION_PROMPT.format(instruction=instruction)
    constraints = llm_client.generate_json(EXTRACTION_MODEL, prompt)
    return constraints


def extract_task(instruction):
    prompt = TASK_EXTRACTION_PROMPT.format(instruction=instruction)
    task = llm_client.generate(EXTRACTION_MODEL, prompt, temperature=0.0)
    return task.strip()


def decompose_instruction(instruction):
    task = extract_task(instruction)
    constraints = extract_constraints(instruction)
    return task, constraints
