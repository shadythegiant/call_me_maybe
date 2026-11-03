*This project has been created as part of the 42 curriculum by azahidi.*

## Description
Call Me Maybe is a specialized inference engine that enables small language models (like `Qwen/Qwen3-0.6B`) to perform highly accurate function calling. Standard LLMs often fail to produce strict, schema-compliant JSON, suffering from hallucinations, conversational filler, and syntax errors. This project solves that problem using **constrained decoding**: intercepting the model's generation process at the mathematical level to guarantee 100% syntactically valid and schema-compliant JSON output, eliminating the need for post-generation text repair.

## Instructions
**Installation:**
This project uses `uv` for dependency management. To install the required packages and synchronize the environment:
```bash
make install
# or manually:
uv sync
```

**Execution:**
To run the engine using the default input files:
```bash
make run
# or manually:
uv run python -m src
```

You can also specify custom paths for the definitions, inputs, and outputs:
```bash
uv run python -m src --functions_definition path/to/defs.json --input path/to/tests.json --output path/to/results.json
```

## Resources
* **Hugging Face Transformers Documentation:** Used to understand BPE (Byte-Pair Encoding) tokenization and vocabulary mappings.
* **Python `re` Module:** Utilized for building the sliding-window regex state machine for string, number, boolean, and integer parsing.
* **Pydantic Documentation:** Referenced for strict schema validation and runtime type enforcement (`TypeAdapter`, `ConfigDict(extra="forbid")`).
* **AI Usage:** AI (Gemini) was utilized as a technical thought partner and code reviewer. It assisted in diagnosing PyTorch CPU thrashing issues, designing the "Sort and Check" fast-path logit optimization, and simulating 42 peer-evaluation defense questions to deepen my understanding of the underlying theoretical computer science concepts.

## Algorithm Explanation
The constrained decoding approach operates as a token-by-token state machine during inference:
1. **Template Generation:** Before generation begins, the engine translates Pydantic function definitions into sequential blocks of required literals (e.g., `{"prompt": "`) and dynamic value types (e.g., `integer`, `string`).
2. **Logit Extraction:** At each step, the LLM predicts the next token, outputting an array of probabilities (logits) for all 151,936 vocabulary tokens.
3. **Regex State Machine:** The engine appends a candidate token to the current generation string and evaluates it against the expected template. A combination of `startswith()` checks for literals and compiled regexes (`RE_INTEGER`, `RE_BOOLEAN`, etc.) verifies if the token creates a valid JSON prefix.
4. **Logit Masking:** Tokens that violate the template or JSON syntax are assigned a logit score of negative infinity ($-\infty$).
5. **Token Selection:** The highest-scoring token that survives the mask is selected, appended to the context, and fed back into the LLM until the JSON object is properly closed.

## Design Decisions
* **The "Sort & Check" Fast-Path:** Instead of running the regex state machine on all 151,000+ tokens at every step (which creates a massive CPU bottleneck), the engine sorts the logits from highest to lowest. It evaluates tokens one by one and stops at the very first valid token. This reduces evaluations from millions to just a few per prompt.
* **Prefix Pre-filling:** The first ~50 characters of the JSON output (`{"prompt": "...", "name": "`) are entirely deterministic. Instead of forcing the LLM to generate these one by one, they are encoded and appended directly to the `input_ids` before generation begins, saving massive amounts of compute time.
* **CPU Thread Optimization:** Multi-core CPUs suffer from severe context-switching overhead when running small LLMs at a batch size of 1. By explicitly setting `torch.set_num_threads(4)`, the engine prevents CPU thrashing.

## Performance Analysis
* **Accuracy:** Successfully extracts correct function names and maps arguments with high precision, easily surpassing the >90% requirement.
* **Reliability:** The state machine guarantees 100% JSON validity. The output is consistently parseable and strictly adheres to the provided schemas.
* **Speed:** Thanks to the "Sort and Check" algorithm, prefix pre-filling, and PyTorch thread limiting, the engine easily processes the entire test suite in under 5 minutes on standard CPU hardware.

## Challenges Faced
* **Vocabulary Ambiguity (BPE):** Byte-Pair Encoding stores spaces as `Ġ` and newlines as `Ċ`. The engine initially rejected valid spaces because it compared raw vocabulary strings against standard spaces. This was solved by explicitly decoding these characters in `vocab.py` during initialization.
* **Hardware Bottlenecks:** Evaluating 150,000 tokens against a regex machine for every generation step caused timeouts. This was solved by implementing the "Sort & Check" optimization and pre-compiling the Python regular expressions.
* **Schema Edge Cases:** The engine initially crashed when encountering `integer` or `boolean` types in private tests. The state machine was extended to handle all distinct Pydantic types accurately.

## Testing Strategy
The implementation was validated using a multi-phase strategy:
1. **Pydantic Hardening:** Used `ConfigDict(extra="forbid")` to ensure the program rejects malformed input files immediately.
2. **Hidden Test Suite (Moulinette):** The engine was tested against the private Moulinette suite, successfully processing edge cases like integers, floats, and booleans without crashing.
3. **Output Validation:** The final generated string is parsed through a strict `FunctionCallOutput` Pydantic model before being written to disk, ensuring no invalid JSON can ever be saved.

## Example Usage
**Input Prompt:**
"Calculate compound interest on 1234567.89 at 0.0375 rate for 23 years"

**Generated Output:**
```json
[
  {
    "prompt": "Calculate compound interest on 1234567.89 at 0.0375 rate for 23 years",
    "name": "fn_calculate_compound_interest",
    "parameters": {
      "principal": 1234567.89,
      "rate": 0.0375,
      "years": 23
    }
  }
]
```