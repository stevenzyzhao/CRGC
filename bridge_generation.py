from llm_client import llm_client
from config import BRIDGE_GENERATION_MODEL, BRIDGE_TEMPERATURE

RECONCILIATION_PROMPT = """You are an AI reasoning optimization engine. Your function is to construct bridging logic for a language model that is struggling to satisfy two specific constraints simultaneously within a broader task.

Core Task Objective: {task}
Primary Constraint α: {constraint_i}
Primary Constraint β: {constraint_j}
Computed Relationship Vector: Interfering

Synthesize a single, highly concise natural language bridge (maximum two sentences) that provides an explicit strategic compromise to satisfy both constraints without violation. Under no circumstances should you introduce novel constraints. Output only the bridge constraint text."""

CONNECTION_PROMPT = """You are an AI reasoning optimization engine. Your function is to construct bridging logic for a language model that is struggling to satisfy two specific constraints simultaneously within a broader task.

Core Task Objective: {task}
Primary Constraint α: {constraint_i}
Primary Constraint β: {constraint_j}
Computed Relationship Vector: Independent

Synthesize a logical connective (maximum two sentences) that forces the model to address both constraints within the same narrative sequence. Under no circumstances should you introduce novel constraints. Output only the bridge constraint text."""


def generate_bridge_constraint(task, constraint_i, constraint_j, relationship_type):
    if relationship_type == "interfering":
        prompt = RECONCILIATION_PROMPT.format(
            task=task, constraint_i=constraint_i, constraint_j=constraint_j
        )
    else:
        prompt = CONNECTION_PROMPT.format(
            task=task, constraint_i=constraint_i, constraint_j=constraint_j
        )

    bridge = llm_client.generate(
        BRIDGE_GENERATION_MODEL, prompt, temperature=BRIDGE_TEMPERATURE
    )
    return bridge.strip()


def generate_all_bridges(task, constraints, problematic_edges):
    bridges = []
    for u, v, rel_type, weight in problematic_edges:
        c_i_idx = int(u[1:]) if u.startswith("C") else None
        c_j_idx = int(v[1:]) if v.startswith("C") else None

        if c_i_idx is None or c_j_idx is None:
            continue

        constraint_i = constraints[c_i_idx]
        constraint_j = constraints[c_j_idx]

        bridge = generate_bridge_constraint(task, constraint_i, constraint_j, rel_type)
        bridges.append({
            "source": constraint_i,
            "target": constraint_j,
            "relationship": rel_type,
            "weight": weight,
            "bridge": bridge,
        })
    return bridges


def enhance_instruction(instruction, bridges):
    if not bridges:
        return instruction

    bridge_text = "\n".join(
        f"- {b['bridge']}" for b in bridges
    )
    enhanced = (
        f"{instruction}\n\n"
        f"Additional guidance to help satisfy all requirements:\n{bridge_text}"
    )
    return enhanced
