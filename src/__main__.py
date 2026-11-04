from .decoder import ConstraintEngine
from .vocab import TokenVocab
from .schemas import FunctionDefinition, PromptItem, FunctionCallOutput
from llm_sdk import Small_LLM_Model  # type: ignore
import argparse
import json
import sys
from pathlib import Path
from pydantic import TypeAdapter, ValidationError
from typing import Any


def load_json_file(filepath: str, adapter: TypeAdapter) -> Any:
    """Safely load and validate a JSON file against a Pydantic TypeAdapter."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        return adapter.validate_python(raw_data)
    except Exception as e:
        print(f"Error loading '{filepath}': {e}")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Call Me Maybe: LLM Function Calling Engine"
    )
    parser.add_argument(
        "-functions_definition", "--functions_definition", type=str,
        default="data/input/functions_definition.json")
    parser.add_argument("--input", type=str,
                        default="data/input/function_calling_tests.json")
    parser.add_argument("--output", type=str,
                        default="data/output/function_calling_results.json")
    args = parser.parse_args()

    print("Loading definitions and prompts...")
    functions = load_json_file(
        args.functions_definition, TypeAdapter(list[FunctionDefinition]))
    prompts = load_json_file(args.input, TypeAdapter(list[PromptItem]))

    print("Initializing LLM...")
    llm = Small_LLM_Model(model_name="Qwen/Qwen3-0.6B")

    vocab_path = llm.get_path_to_vocab_file()
    vocab = TokenVocab.load_from_file(vocab_path)
    engine = ConstraintEngine(vocab, functions)

    results = []

    print(f"\nProcessing {len(prompts)} prompts...")
    for idx, prompt_item in enumerate(prompts):
        test_prompt = prompt_item.prompt
        escaped_prompt = json.dumps(test_prompt)[1:-1]

        # 1. Build the semantic context
        full_prompt = (
            "Select a function to answer the query.\n\nAvailable functions:\n"
        )
        for f in functions:
            full_prompt += f"- {f.name}: {f.description}\n"
        full_prompt += f"\nQuery: {test_prompt}\n\nResult:\n"

        input_ids = llm.encode(full_prompt)[0].tolist()
        prefix = f'{{"prompt": "{escaped_prompt}", "name": "'
        generated_text = prefix
        input_ids += llm.encode(prefix)[0].tolist()

        max_tokens = 150

        print(f"[{idx + 1}/{len(prompts)}] Generating for: '{test_prompt}'",
              end="", flush=True)

        for step in range(max_tokens):
            logits = llm.get_logits_from_input_ids(input_ids)
            indexed_logits = list(enumerate(logits))
            indexed_logits.sort(key=lambda x: x[1], reverse=True)

            best_token_id = None
            best_token_str = ""

            for token_id, logit_score in indexed_logits:
                # Skip out-of-vocabulary fallback IDs
                if token_id not in vocab.id_to_str:
                    continue

                token_str = vocab.id_to_str[token_id]

                # Check if this single token survives the engine's mask
                test_mask = engine.mask_logits(
                    logits=[0.0] * len(logits),
                    current_generation=generated_text,
                    original_prompt=escaped_prompt,
                    single_token_test=(token_id, token_str)
                )

                if test_mask:
                    best_token_id = token_id
                    best_token_str = token_str
                    break  # Stop immediately! We found the best valid token.

            if best_token_id is None:
                print("No Valid token sys is exiting")
                break

            generated_text += best_token_str
            input_ids.append(best_token_id)

            if generated_text.endswith("}}"):
                break

        print(" -> Done.")

        try:
            # Validate output and append to our results array
            final_obj = FunctionCallOutput.model_validate_json(generated_text)
            results.append(final_obj.model_dump())
        except ValidationError as e:
            print(
                "\nError: Model produced invalid output"
                f"for prompt '{test_prompt}': {e}")
            sys.exit(1)

    # Safely write the output file
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSuccess! All {len(results)} results written to '{args.output}'.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user"
              " (Ctrl+C). Exiting gracefully...")
        sys.exit(130)
