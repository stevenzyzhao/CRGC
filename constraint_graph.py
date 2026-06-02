import numpy as np
import networkx as nx
from llm_client import llm_client
from config import N_SAMPLES, DELTA, TAU
from evaluation import evaluate_constraint


def compute_edge_weights(model_name, instruction, task, constraints):
    n = len(constraints)
    satisfaction_scores = np.zeros((N_SAMPLES, n))

    full_prompt = instruction
    outputs = llm_client.generate(model_name, full_prompt, temperature=0.7, n=N_SAMPLES)
    if isinstance(outputs, str):
        outputs = [outputs]

    for sample_idx, output in enumerate(outputs):
        for c_idx, constraint in enumerate(constraints):
            score = evaluate_constraint(constraint, output)
            satisfaction_scores[sample_idx, c_idx] = score

    baseline_satisfaction = np.mean(satisfaction_scores, axis=0)

    weight_matrix = np.zeros((n, n))
    for i in range(n):
        satisfied_mask = satisfaction_scores[:, i] >= TAU
        if np.sum(satisfied_mask) < 2:
            for j in range(n):
                if i != j:
                    weight_matrix[i][j] = 0.0
            continue
        for j in range(n):
            if i == j:
                continue
            conditional_satisfaction = np.mean(satisfaction_scores[satisfied_mask, j])
            if baseline_satisfaction[j] > 0:
                weight_matrix[i][j] = (baseline_satisfaction[j] - conditional_satisfaction) / baseline_satisfaction[j]
            else:
                weight_matrix[i][j] = 0.0

    return weight_matrix, baseline_satisfaction


def build_initial_graph(task, constraints, weight_matrix):
    n = len(constraints)
    G = nx.DiGraph()

    G.add_node("T", label=task, node_type="task")
    for i, c in enumerate(constraints):
        G.add_node(f"C{i}", label=c, node_type="constraint")

    for i in range(n):
        G.add_edge("T", f"C{i}", weight=0.0)

    for i in range(n):
        for j in range(n):
            if i != j:
                G.add_edge(f"C{i}", f"C{j}", weight=weight_matrix[i][j])

    return G


def compute_minimum_spanning_arborescence(G):
    try:
        arb = nx.minimum_spanning_arborescence(G, attr="weight")
        return arb
    except nx.NetworkXException:
        return None


def detect_problematic_edges(arborescence, delta=DELTA):
    problematic_edges = []
    for u, v, data in arborescence.edges(data=True):
        w = data.get("weight", 0.0)
        if w > delta:
            problematic_edges.append((u, v, "interfering", w))
        elif w >= -delta and w <= delta and u != "T":
            problematic_edges.append((u, v, "independent", w))
    return problematic_edges


def construct_constraint_graph(model_name, instruction, task, constraints):
    weight_matrix, baseline_satisfaction = compute_edge_weights(
        model_name, instruction, task, constraints
    )

    G = build_initial_graph(task, constraints, weight_matrix)
    arborescence = compute_minimum_spanning_arborescence(G)

    if arborescence is None:
        problematic_edges = []
        for i in range(len(constraints)):
            for j in range(len(constraints)):
                if i != j and weight_matrix[i][j] > DELTA:
                    problematic_edges.append(
                        (f"C{i}", f"C{j}", "interfering", weight_matrix[i][j])
                    )
        return G, None, problematic_edges, weight_matrix

    problematic_edges = detect_problematic_edges(arborescence)
    return G, arborescence, problematic_edges, weight_matrix
