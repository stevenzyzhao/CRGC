from constraint_extraction import decompose_instruction
from constraint_graph import construct_constraint_graph
from bridge_generation import generate_all_bridges, enhance_instruction
from evaluation import evaluate_all_metrics
from llm_client import llm_client
from config import TAU


class CRGC:
    def __init__(self, generation_model, eval_model=None):
        self.generation_model = generation_model
        self.eval_model = eval_model

    def run(self, instruction):
        task, constraints = decompose_instruction(instruction)

        if len(constraints) <= 1:
            output = llm_client.generate(self.generation_model, instruction, temperature=0.0)
            metrics = evaluate_all_metrics(constraints, output, self.eval_model)
            return {
                "instruction": instruction,
                "task": task,
                "constraints": constraints,
                "bridges": [],
                "enhanced_instruction": instruction,
                "output": output,
                "metrics": metrics,
            }

        _, arborescence, problematic_edges, weight_matrix = construct_constraint_graph(
            self.generation_model, instruction, task, constraints
        )

        bridges = generate_all_bridges(task, constraints, problematic_edges)
        enhanced_instruction = enhance_instruction(instruction, bridges)
        output = llm_client.generate(self.generation_model, enhanced_instruction, temperature=0.0)
        metrics = evaluate_all_metrics(constraints, output, self.eval_model)
        metrics["turns_used"] = 1

        return {
            "instruction": instruction,
            "task": task,
            "constraints": constraints,
            "problematic_edges": problematic_edges,
            "bridges": bridges,
            "enhanced_instruction": enhanced_instruction,
            "output": output,
            "metrics": metrics,
            "weight_matrix": weight_matrix.tolist(),
        }


class StandardPrompting:
    def __init__(self, generation_model, eval_model=None):
        self.generation_model = generation_model
        self.eval_model = eval_model

    def run(self, instruction):
        task, constraints = decompose_instruction(instruction)
        output = llm_client.generate(self.generation_model, instruction, temperature=0.0)
        metrics = evaluate_all_metrics(constraints, output, self.eval_model)
        return {
            "instruction": instruction,
            "task": task,
            "constraints": constraints,
            "output": output,
            "metrics": metrics,
        }


class ChainOfThought:
    def __init__(self, generation_model, eval_model=None):
        self.generation_model = generation_model
        self.eval_model = eval_model

    def run(self, instruction):
        task, constraints = decompose_instruction(instruction)
        cot_prompt = (
            f"{instruction}\n\n"
            f"Let's think step by step about how to satisfy all requirements in this instruction."
        )
        output = llm_client.generate(self.generation_model, cot_prompt, temperature=0.0)
        metrics = evaluate_all_metrics(constraints, output, self.eval_model)
        return {
            "instruction": instruction,
            "task": task,
            "constraints": constraints,
            "output": output,
            "metrics": metrics,
        }


class SelfReflection:
    def __init__(self, generation_model, eval_model=None):
        self.generation_model = generation_model
        self.eval_model = eval_model

    def run(self, instruction):
        task, constraints = decompose_instruction(instruction)
        initial_output = llm_client.generate(self.generation_model, instruction, temperature=0.0)
        reflection_prompt = (
            f"Original instruction: {instruction}\n\n"
            f"Your initial response:\n{initial_output}\n\n"
            f"Please review your response against all requirements in the instruction. "
            f"If any requirements are not fully satisfied, provide an improved response "
            f"that better satisfies all constraints. If all requirements are met, "
            f"reproduce your response unchanged."
        )
        output = llm_client.generate(self.generation_model, reflection_prompt, temperature=0.0)
        metrics = evaluate_all_metrics(constraints, output, self.eval_model)
        return {
            "instruction": instruction,
            "task": task,
            "constraints": constraints,
            "output": output,
            "metrics": metrics,
        }


class SelfConsistency:
    def __init__(self, generation_model, eval_model=None, n_samples=5):
        self.generation_model = generation_model
        self.eval_model = eval_model
        self.n_samples = n_samples

    def run(self, instruction):
        task, constraints = decompose_instruction(instruction)
        outputs = llm_client.generate(
            self.generation_model, instruction, temperature=0.7, n=self.n_samples
        )
        if isinstance(outputs, str):
            outputs = [outputs]

        best_output = None
        best_score = -1
        for output in outputs:
            metrics = evaluate_all_metrics(constraints, output, self.eval_model)
            score = metrics["wcs"]
            if score > best_score:
                best_score = score
                best_output = output

        metrics = evaluate_all_metrics(constraints, best_output, self.eval_model)
        return {
            "instruction": instruction,
            "task": task,
            "constraints": constraints,
            "output": best_output,
            "metrics": metrics,
        }


class ConstraintOptimizationPrompting:
    def __init__(self, generation_model, eval_model=None):
        self.generation_model = generation_model
        self.eval_model = eval_model

    def run(self, instruction):
        task, constraints = decompose_instruction(instruction)
        constraint_list = "\n".join(f"- {c}" for c in constraints)
        cop_prompt = (
            f"{instruction}\n\n"
            f"You must satisfy ALL of the following constraints:\n{constraint_list}\n\n"
            f"Optimize your response to maximize satisfaction of every constraint listed above."
        )
        output = llm_client.generate(self.generation_model, cop_prompt, temperature=0.0)
        metrics = evaluate_all_metrics(constraints, output, self.eval_model)
        return {
            "instruction": instruction,
            "task": task,
            "constraints": constraints,
            "output": output,
            "metrics": metrics,
        }


class ThinkToThink:
    def __init__(self, generation_model, eval_model=None, n_samples=5):
        self.generation_model = generation_model
        self.eval_model = eval_model
        self.n_samples = n_samples

    def run(self, instruction):
        task, constraints = decompose_instruction(instruction)
        samples = llm_client.generate(
            self.generation_model, instruction, temperature=0.7, n=self.n_samples
        )
        if isinstance(samples, str):
            samples = [samples]

        sample_text = "\n---\n".join(
            f"Sample {i+1}:\n{s}" for i, s in enumerate(samples)
        )
        t2_prompt = (
            f"Original instruction: {instruction}\n\n"
            f"Here are multiple attempts at answering this instruction:\n{sample_text}\n\n"
            f"Analyze the strengths and weaknesses of each attempt with respect to "
            f"satisfying all requirements. Then produce an optimal response that combines "
            f"the best aspects of all attempts to fully satisfy every constraint."
        )
        output = llm_client.generate(self.generation_model, t2_prompt, temperature=0.0)
        metrics = evaluate_all_metrics(constraints, output, self.eval_model)
        return {
            "instruction": instruction,
            "task": task,
            "constraints": constraints,
            "output": output,
            "metrics": metrics,
        }


class DeCRIM:
    def __init__(self, generation_model, eval_model=None, max_turns=6):
        self.generation_model = generation_model
        self.eval_model = eval_model
        self.max_turns = max_turns

    def run(self, instruction):
        task, constraints = decompose_instruction(instruction)
        output = llm_client.generate(self.generation_model, instruction, temperature=0.0)
        turns_used = 1

        for turn in range(1, self.max_turns):
            metrics = evaluate_all_metrics(constraints, output, self.eval_model)
            if metrics["tcr"] >= TAU:
                break

            violated = [
                constraints[i] for i, s in enumerate(metrics["scores"]) if s < TAU
            ]
            violation_text = "\n".join(f"- {v}" for v in violated)
            refine_prompt = (
                f"Original instruction: {instruction}\n\n"
                f"Your previous response:\n{output}\n\n"
                f"The following constraints were NOT satisfied:\n{violation_text}\n\n"
                f"Please produce a corrected response that satisfies ALL constraints, "
                f"paying special attention to the violated ones listed above."
            )
            output = llm_client.generate(self.generation_model, refine_prompt, temperature=0.0)
            turns_used += 1

        metrics = evaluate_all_metrics(constraints, output, self.eval_model)
        metrics["turns_used"] = turns_used
        return {
            "instruction": instruction,
            "task": task,
            "constraints": constraints,
            "output": output,
            "metrics": metrics,
        }
