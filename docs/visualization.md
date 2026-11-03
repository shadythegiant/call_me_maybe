# Call Me Maybe: Visual Architecture

This document is the visual map of the project. Read it alongside
[`project_mastery.md`](./project_mastery.md), which explains how to study,
run, debug, and extend the system.

## 1. The one-sentence model

**Call Me Maybe uses a small causal language model to choose a function and
generate its JSON arguments, while a token-level constraint engine prevents
the generated text from leaving the JSON shape described by the available
function definitions.**

The project is therefore closer to a **constrained function-call generator**
than to a complete tool-execution framework:

```mermaid
flowchart LR
    U[User prompt] --> A[src.__main__.py]
    D[functions_definition.json] --> A
    A --> L[Small_LLM_Model]
    L --> C[ConstraintEngine]
    C --> O[Validated JSON output]
    O --> W[function_calling_results.json]
    W -. optional grading .-> M[moulinette]
```

The current program does **not** locate and invoke Python functions after
generation. The `moulinette` can invoke reference functions while grading, but
that is a separate evaluation path.

## 2. Repository map

```mermaid
flowchart TB
    ROOT[CallMe_Maybe] --> SRC[src/]
    ROOT --> SDK[llm_sdk/]
    ROOT --> DATA[data/]
    ROOT --> MOUL[moulinette/]
    ROOT --> DOCS[docs/]

    SRC --> MAIN[src/__main__.py<br/>CLI and generation loop]
    SRC --> DEC[src/decoder.py<br/>template matching and masking]
    SRC --> SCH[src/schemas.py<br/>Pydantic input/output contracts]
    SRC --> VOC[src/vocab.py<br/>token ID to text mapping]

    SDK --> MODEL[llm_sdk/__init__.py<br/>Hugging Face model wrapper]

    DATA --> DEFINITIONS[data/input/functions_definition.json]
    DATA --> PROMPTS[data/input/function_calling_tests.json]
    DATA --> RESULTS[data/output/function_calling_results.json]

    MOUL --> EXDEF[functions_definition.py<br/>reference functions and tests]
    MOUL --> EXTRACT[extract_functions_infos.py<br/>Python to JSON metadata]
    MOUL --> GENERATE[generate_tests_and_corrections.py<br/>exercise files]
    MOUL --> GRADE[__main__.py<br/>student answer grading]
    MOUL --> FORMAT[output_formatter.py<br/>terminal presentation]

    DOCS --> EXISTING[algorithm and line-by-line explanations]
    DOCS --> THIS[visualization.md]
    DOCS --> MASTER[project_mastery.md]
```

### Responsibilities at a glance

| Area | Responsibility | Does not do |
|---|---|---|
| `src/__main__.py` | Load data, build prompts, run generation, validate and save results | Execute the selected function |
| `src/decoder.py` | Decide whether a candidate token keeps a valid JSON/template prefix | Decide whether a function is authorized in a real application |
| `src/schemas.py` | Validate input definitions, prompts, and final output shape | Validate the semantic correctness of every argument |
| `src/vocab.py` | Normalize model vocabulary text such as BPE space markers | Tokenize the prompt itself |
| `llm_sdk` | Load the tokenizer/model and return next-token logits | Apply project-specific JSON constraints |
| `moulinette` | Produce exercises and compare answers to reference functions | Participate in the main inference loop |

## 3. End-to-end execution

```mermaid
sequenceDiagram
    autonumber
    participant CLI as src.__main__
    participant JSON as Input JSON
    participant SDK as Small_LLM_Model
    participant VOC as TokenVocab
    participant ENG as ConstraintEngine
    participant OUT as Output JSON

    CLI->>JSON: Load function definitions
    JSON-->>CLI: list[FunctionDefinition]
    CLI->>JSON: Load prompt tests
    JSON-->>CLI: list[PromptItem]
    CLI->>SDK: Load Qwen/Qwen3-0.6B
    SDK-->>CLI: tokenizer/model wrapper
    CLI->>SDK: Get vocabulary path
    SDK-->>CLI: vocab.json path
    CLI->>VOC: Build id_to_str mapping
    VOC-->>CLI: token text lookup
    CLI->>ENG: Create engine(vocab, functions)

    loop For each prompt
        CLI->>SDK: Encode semantic context
        SDK-->>CLI: input token IDs
        CLI->>SDK: Encode deterministic JSON prefix
        SDK-->>CLI: prefix token IDs
        loop Until JSON closes or max_tokens is reached
            CLI->>SDK: Get next-token logits
            SDK-->>CLI: scores for vocabulary
            CLI->>CLI: Sort token IDs by score
            CLI->>ENG: Test highest-scoring candidate
            ENG->>ENG: Match tentative text to templates
            ENG-->>CLI: allowed / rejected
            CLI->>CLI: Append first allowed token
        end
        CLI->>JSON: Validate complete JSON with Pydantic
    end
    CLI->>OUT: Write validated result list
```

## 4. Input and output contracts

The definition file describes **metadata**, not executable code. A definition
has a name, description, parameter types, and return type:

```mermaid
classDiagram
    class FunctionDefinition {
        +str name
        +str description
        +dict parameters
        +TypeSchema returns
    }
    class TypeSchema {
        +ParameterType type
    }
    class PromptItem {
        +str prompt
    }
    class FunctionCallOutput {
        +str prompt
        +str name
        +dict parameters
    }
    FunctionDefinition "1" --> "*" TypeSchema : parameter schemas
    PromptItem ..> FunctionDefinition : prompt is interpreted using
    FunctionCallOutput ..> FunctionDefinition : name and parameters follow
```

The generated object is conceptually:

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": {
    "a": 2,
    "b": 3
  }
}
```

`returns` is part of the input contract and is useful to a future execution
layer, but the current constrained template focuses on the function name and
parameters. The current output also does not contain a computed return value.

## 5. What happens inside one generation step?

At every step, the model produces one logit for every vocabulary token. The
engine does not blindly accept the model's best token. It tries candidates in
descending logit order and accepts the first one that keeps at least one valid
function template possible.

```mermaid
flowchart TD
    S[Current input_ids and generated_text] --> L[Model returns next-token logits]
    L --> SORT[Sort token IDs by score descending]
    SORT --> C{Candidate token in vocabulary?}
    C -- No --> NEXT[Try next candidate]
    C -- Yes --> T[Append candidate tentatively]
    T --> MATCH[Compare tentative text with valid templates]
    MATCH --> OK{Any template survives?}
    OK -- No --> NEXT
    OK -- Yes --> ACCEPT[Accept token]
    ACCEPT --> CLOSE{Text ends with }}?}
    CLOSE -- No --> S
    CLOSE -- Yes --> DONE[Validate final object and save]
```

This is the project's **sort-and-check fast path**. It avoids running the
constraint check on every vocabulary token when a high-probability valid token
is found early.

## 6. Template construction

For a selected function, `_get_expected_template` creates an ordered list of
literal and typed blocks. For example, a function with `a: number` and
`b: number` becomes conceptually:

```mermaid
flowchart LR
    A["{\"prompt\": \"...\", \"name\": \"fn_add_numbers\", \"parameters\": {"]
    B["\"a\": "]
    C[number]
    D[", \"b\": "]
    E[number]
    F["}}"]
    A --> B --> C --> D --> E --> F
```

Literal blocks must match character-for-character. Typed blocks are checked by
regular expressions:

| Schema type | Constraint used by the engine | Meaning |
|---|---|---|
| `string` | `RE_STRING` | Starts with a quote and follows JSON string escaping rules |
| `number` | `RE_NUMBER` | Optional sign, digits, and optional decimal portion |
| `integer` | `RE_INTEGER` | Optional sign and integer digits |
| `boolean` | `RE_BOOLEAN` | Prefixes of `true` or `false`, then completion |

The engine tracks a cursor `(block_index, character_index)` through this
template:

```mermaid
stateDiagram-v2
    [*] --> LiteralPrefix
    LiteralPrefix --> FunctionName: literal prefix complete
    FunctionName --> ParameterLiteral: name identifies a function
    ParameterLiteral --> TypedValue: literal complete
    TypedValue --> ParameterLiteral: value complete and more params
    ParameterLiteral --> ClosingLiteral: all params complete
    ClosingLiteral --> [*]: "}}" complete
    TypedValue --> DeadBranch: invalid character or invalid continuation
    LiteralPrefix --> DeadBranch: mismatch
    DeadBranch --> [*]
```

Before the model loop starts, `src/__main__.py` pre-fills the deterministic
prefix through `generated_text` and `input_ids`. The model is therefore used
mainly for the function name and argument values, not for repeatedly emitting
known punctuation.

## 7. The two meanings of “function calling”

The pasted architecture describes a useful production pattern, but this
repository implements only the first half of it:

```mermaid
flowchart LR
    subgraph Current["Current repository"]
        P[Prompt] --> SELECT[Model chooses function]
        SELECT --> ARG[Model generates arguments]
        ARG --> VALIDATE[Constrained JSON + Pydantic validation]
        VALIDATE --> FILE[Write structured call]
    end

    subgraph Future["Full tool-calling application"]
        FILE2[Structured call] --> LOOKUP[Approved registry lookup]
        LOOKUP --> AUTH[Permission and policy checks]
        AUTH --> EXEC[Execute Python function]
        EXEC --> RESULT[Return result to model/user]
    end

    FILE -. handoff .-> FILE2
```

The safe extension point is a registry whose keys come from validated
definitions:

```python
FUNCTION_REGISTRY = {
    "fn_add_numbers": fn_add_numbers,
    "fn_greet": fn_greet,
}
```

The registry must remain separate from JSON. JSON can describe a tool, but it
must never be treated as Python code. Before execution, a future layer should
validate the selected name, argument types, permissions, and execution errors.

## 8. Reliability layers

No single layer solves every failure mode:

```mermaid
flowchart TB
    L1[Language model<br/>semantic choice]
    L2[Token vocabulary normalization<br/>BPE text interpretation]
    L3[ConstraintEngine<br/>valid prefix and schema-shaped JSON]
    L4[Pydantic adapters<br/>input and final-object validation]
    L5[Registry and authorization<br/>approved executable functions]
    L6[Business validation and execution handling]

    L1 --> L2 --> L3 --> L4 --> L5 --> L6
```

Constrained decoding protects syntax and allowed structure. It does not prove
that the model chose the semantically correct function, that a value is safe,
or that a function should be allowed to run.

## 9. Performance story

```mermaid
flowchart TD
    BASE[Naive approach:<br/>check every vocabulary token fully] --> COST[Many regex/template checks per step]
    PREFILL[Prefix pre-fill] --> LESS1[Fewer generated steps]
    SORT[Sort logits and stop at first valid token] --> LESS2[Fewer checks in each step]
    REGEX[Compile regexes once] --> LESS3[Lower repeated setup cost]
    THREADS[Limit CPU thread oversubscription] --> LESS4[Less context-switching overhead]
    LESS1 --> TOTAL[Lower end-to-end inference cost]
    LESS2 --> TOTAL
    LESS3 --> TOTAL
    LESS4 --> TOTAL
```

The expensive part is model inference plus repeated candidate checks. The
optimizations reduce work; they do not change the correctness contract.

## 10. Exercise and grading pipeline

```mermaid
flowchart LR
    REF[Reference Python functions] --> META[extract_functions_infos.py]
    REF --> TESTGEN[generate_tests_and_corrections.py]
    META --> DEF[functions_definition.json]
    TESTGEN --> TESTS[function_calling_tests.json]
    TESTGEN --> CORR[corrections JSON]
    DEF --> STUDENT[src/__main__.py]
    TESTS --> STUDENT
    STUDENT --> ANSWERS[student results]
    ANSWERS --> GRADE[moulinette grade_student_answers]
    CORR --> GRADE
    REF --> GRADE
    GRADE --> SCORE[score and diagnostics]
```

Public exercises are exposed to the student. Private exercises are retained
by the grading side and test additional types and edge cases such as integers
and booleans. This is why the metadata format is intentionally broader than
the small public sample.

## 11. Reading order

1. Read [`project_mastery.md`](./project_mastery.md) for the learning path.
2. Read [`src/schemas.py`](./src/schemas.py) to learn the data contract.
3. Read [`src/__main__.py`](./src/__main__.py) for orchestration.
4. Read [`src/decoder.py`](./src/decoder.py) for the state machine.
5. Read [`src/vocab.py`](./src/vocab.py) and [`llm_sdk/llm_sdk/__init__.py`](./llm_sdk/llm_sdk/__init__.py) for token/model mechanics.
6. Read the existing line-by-line notes in [`docs/`](./docs/).

