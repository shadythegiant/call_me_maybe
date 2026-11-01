import argparse
import json
import sys
from pathlib import Path
from pydantic import TypeAdapter, ValidationError

from llm_sdk import Small_LLM_Model
from .schemas import FunctionDefinition, PromptItem, FunctionCallOutput
from .vocab import TokenVocab
from .decoder import ConstraintEngine


def load_json_file(filepath: str, adapter: TypeAdapter) -> list:
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
        description="Call Me Maybe: LLM Function Calling Engine")
    parser.add_argument("-functions_definition", "--functions_definition", type=str,
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

        full_prompt = "Select a function to answer the query.\n\nAvailable functions:\n"
        for f in functions:
            full_prompt += f"- {f.name}: {f.description}\n"
        full_prompt += f"\nQuery: {test_prompt}\n\nResult:\n"

        input_ids = llm.encode(full_prompt)[0].tolist()
        generated_text = ""
        max_tokens = 150

        print(f"[{idx + 1}/{len(prompts)}] Generating for: '{test_prompt}'",
              end="", flush=True)

        for step in range(max_tokens):
            logits = llm.get_logits_from_input_ids(input_ids)
            masked_logits = engine.mask_logits(
                logits, generated_text, escaped_prompt)

            best_token_id = masked_logits.index(max(masked_logits))
            best_token_str = vocab.id_to_str[best_token_id]

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
                f"\nError: Model produced invalid output for prompt '{test_prompt}': {e}")
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
        print("Programm Interrupted by user")
        sys.exit(130)
