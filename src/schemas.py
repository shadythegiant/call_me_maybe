from __future__ import annotations

from typing import Literal, TypeAlias, Any

from pydantic import BaseModel, ConfigDict, Field


ParameterType: TypeAlias = Literal[
    "string",
    "number",
    "integer",
    "boolean",
]


class TypeSchema(BaseModel):
    """Describe the JSON-compatible type produced or accepted by a function."""
    model_config = ConfigDict(extra="forbid")
    type: ParameterType


class FunctionDefinition(BaseModel):
    """Describe a callable function and its input and output types."""
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters: dict[str, TypeSchema]
    returns: TypeSchema


class PromptItem(BaseModel):
    """Represent a natural language request from the input JSON."""
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=1)


class FunctionCallOutput(BaseModel):
    """Represent the final structured result written to disk."""
    model_config = ConfigDict(extra="forbid")
    prompt: str
    name: str
    parameters: dict[str, Any]
