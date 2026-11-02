import json
from pydantic import BaseModel, ConfigDict


class TokenVocab(BaseModel):
    """Stores the reversed vocabulary mapping from the LLM."""

    model_config = ConfigDict(extra="forbid")

    # Maps integer token IDs to their string representations
    id_to_str: dict[int, str]

    @classmethod
    def load_from_file(cls, vocab_path: str) -> "TokenVocab":
        """Load and clean the Hugging Face vocab.json file."""
        with open(vocab_path, "r", encoding="utf-8") as f:
            raw_vocab: dict[str, int] = json.load(f)

        id_to_str: dict[int, str] = {}
        for token_str, token_id in raw_vocab.items():
            # In BPE, 'Ġ' represents a leading space and 'Ċ'
            # represents a newline
            clean_token = token_str.replace("Ġ", " ").replace("Ċ", "\n")
            id_to_str[token_id] = clean_token

        return cls(id_to_str=id_to_str)
