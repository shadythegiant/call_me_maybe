import math
import re
from typing import List

from .vocab import TokenVocab
from .schemas import FunctionDefinition

# Pre-compiled regexes for maximum speed
RE_NUMBER = re.compile(r'^-?(?:0|[1-9]\d*)?(?:\.\d*)?')
RE_STRING = re.compile(r'^"([^"\\]|\\.)*("?)')


class ConstraintEngine:
    def __init__(self, vocab: TokenVocab, functions: list[FunctionDefinition]):
        self.vocab = vocab
        self.functions = functions

    def _get_expected_template(
        self, func: FunctionDefinition, original_prompt: str
    ) -> list[tuple[str, str]]:
        """Breaks the exact required JSON structure into an ordered sequence."""
        seq = []
        seq.append(
            ("literal", f'{{"prompt": "{original_prompt}", "name": "{func.name}", "parameters": {{'))

        params = list(func.parameters.items())
        for i, (p_name, p_schema) in enumerate(params):
            if i > 0:
                seq.append(("literal", f', "{p_name}": '))
            else:
                seq.append(("literal", f'"{p_name}": '))
            seq.append((p_schema.type, ""))

        seq.append(("literal", "}}"))
        return seq

    def _is_valid_prefix(self, text: str, template: list[tuple[str, str]]) -> bool:
        """Evaluates if 'text' is a valid partial generation of the template."""
        text_idx = 0

        for block_type, literal_val in template:
            if text_idx >= len(text):
                return True

            remaining_text = text[text_idx:]

            if block_type == "literal":
                if literal_val.startswith(remaining_text):
                    return True
                elif remaining_text.startswith(literal_val):
                    text_idx += len(literal_val)
                else:
                    return False

            elif block_type == "number":
                match = RE_NUMBER.match(remaining_text)
                num_str = match.group(0) if match else ""

                if not num_str:
                    return False
                if num_str == remaining_text:
                    return True
                text_idx += len(num_str)

            elif block_type == "string":
                if not remaining_text.startswith('"'):
                    return False
                match = RE_STRING.match(remaining_text)
                str_part = match.group(0) if match else ""

                if str_part == remaining_text:
                    return True
                if not str_part.endswith('"') or len(str_part) == 1:
                    return False
                text_idx += len(str_part)

        return text_idx >= len(text)

    def mask_logits(
        self, logits: List[float], current_generation: str, original_prompt: str
    ) -> List[float]:
        """Mask illegal tokens using fast-paths and template parsing."""

        # SPEEDUP 1: Memory Pre-allocation
        # Instead of copying the list and running 'if i not in vocab' 150k times,
        # we instantly allocate an array of negative infinity.
        masked_logits = [-math.inf] * len(logits)

        base_prefix = f'{{"prompt": "{original_prompt}", "name": "'

        # SPEEDUP 2: The Fast-Path
        # If we are typing the base prompt (which is >50% of the tokens),
        # skip all Regex templates and use ultra-fast C-level string checking.
        if len(current_generation) < len(base_prefix):
            expected = base_prefix[len(current_generation):]

            for token_id, token_str in self.vocab.id_to_str.items():
                if expected.startswith(token_str) or token_str.startswith(expected):
                    # Restore the valid logit score
                    masked_logits[token_id] = logits[token_id]
            return masked_logits

        # --- STANDARD PHASE: Schema & Parameter Enforcement ---
        valid_templates = []
        match = re.search(r'"name":\s*"([^"]+)"', current_generation)
        if match:
            chosen_name = match.group(1)
            for func in self.functions:
                if func.name == chosen_name:
                    valid_templates.append(
                        self._get_expected_template(func, original_prompt))
                    break
        else:
            for func in self.functions:
                valid_templates.append(
                    self._get_expected_template(func, original_prompt))

        if not valid_templates:
            return masked_logits

        # The Heavy Loop (now only running for the final few parameter tokens)
        for token_id, token_str in self.vocab.id_to_str.items():
            tentative_generation = current_generation + token_str

            for template in valid_templates:
                if self._is_valid_prefix(tentative_generation, template):
                    masked_logits[token_id] = logits[token_id]
                    break

        return masked_logits
