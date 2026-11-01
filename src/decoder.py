import math
import re
from typing import List

from .vocab import TokenVocab
from .schemas import FunctionDefinition

# PRE-COMPILE REGEXES ONCE FOR SPEED
# Translates the regex strings into a C-level state machine once,
# preventing massive overhead during the token evaluation loop.
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

        # 1. The mandatory JSON start
        seq.append(
            ("literal", f'{{"prompt": "{original_prompt}", "name": "{func.name}", "parameters": {{'))

        # 2. The parameters in order based on the function schema
        params = list(func.parameters.items())
        for i, (p_name, p_schema) in enumerate(params):
            if i > 0:
                seq.append(("literal", f', "{p_name}": '))
            else:
                seq.append(("literal", f'"{p_name}": '))

            seq.append((p_schema.type, ""))

        # 3. The closure
        seq.append(("literal", "}}"))
        return seq

    def _get_cursor_state(self, text: str, template: list[tuple[str, str]]):
        """Returns the exact state (block_index, char_index) of the generated text."""
        text_idx = 0

        for i, (block_type, literal_val) in enumerate(template):
            # If we've successfully verified all text, return the current block state
            if text_idx >= len(text):
                return (i, 0)

            remaining_text = text[text_idx:]

            if block_type == "literal":
                if literal_val.startswith(remaining_text):
                    return (i, len(remaining_text))
                elif remaining_text.startswith(literal_val):
                    text_idx += len(literal_val)
                else:
                    return None  # Branch dead: Mismatch

            elif block_type == "number":
                match = RE_NUMBER.match(remaining_text)
                num_str = match.group(0) if match else ""

                if not num_str:
                    return None  # Branch dead: Invalid character for a number

                if num_str == remaining_text:
                    return (i, len(num_str))

                # Number finished, advance the cursor past it
                text_idx += len(num_str)

            elif block_type == "string":
                if not remaining_text.startswith('"'):
                    return None

                match = RE_STRING.match(remaining_text)
                str_part = match.group(0) if match else ""

                if str_part == remaining_text:
                    return (i, len(str_part))

                # If the string isn't closed, but there's more text, it's invalid
                if not str_part.endswith('"') or len(str_part) == 1:
                    return None

                # String finished, advance the cursor past it
                text_idx += len(str_part)

        # If we successfully parsed the whole template and the text ran out
        if text_idx >= len(text):
            return (len(template), 0)

        return None

    def mask_logits(
        self, logits: List[float], current_generation: str, original_prompt: str,
        single_token_test: tuple[int, str] | None = None
    ) -> bool | List[float]:
        """
        Evaluates valid templates. 
        If single_token_test is provided, returns a boolean indicating if that specific token is allowed.
        Otherwise, returns a fully masked logit array for the entire vocabulary.
        """
        # --- 1. Identify Valid Templates ---
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
            # We are still generating the name; all templates are valid
            for func in self.functions:
                valid_templates.append(
                    self._get_expected_template(func, original_prompt))

        if not valid_templates:
            return False if single_token_test else [-math.inf] * len(logits)

        # --- 2. FAST PATH: "Sort & Check" Optimization ---
        # Used when __main__.py evaluates one token at a time starting from the highest logit
        if single_token_test:
            token_id, token_str = single_token_test
            tentative_generation = current_generation + token_str

            for template in valid_templates:
                if self._get_cursor_state(tentative_generation, template) is not None:
                    return True
            return False

        # --- 3. SLOW PATH: Full Vocabulary Masking (Fallback) ---
        # Only used if the generation loop does not pass single_token_test
        masked_logits = [-math.inf] * len(logits)

        for token_id, token_str in self.vocab.id_to_str.items():
            tentative_generation = current_generation + token_str

            is_valid = False
            for template in valid_templates:
                if self._get_cursor_state(tentative_generation, template) is not None:
                    is_valid = True
                    break

            if is_valid:
                masked_logits[token_id] = logits[token_id]

        return masked_logits
