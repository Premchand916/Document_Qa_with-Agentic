"""Bounded public request contracts. Identity is never accepted in request JSON."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StringConstraints, field_validator


Text = Annotated[str, StringConstraints(strict=True, strip_whitespace=True, min_length=1, max_length=8000)]
FileType = Literal["pdf", "pptx", "xlsx", "xlsm", "csv", "tsv", "txt", "md", "json", "jpg", "jpeg", "png", "tiff", "tif", "bmp", "webp"]


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["user", "assistant"]
    content: Text


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: Text
    conversation_id: Annotated[str, StringConstraints(strict=True, pattern=r"^[a-f0-9]{32}$")] | None = None
    idempotency_key: Annotated[str, StringConstraints(strict=True, min_length=8, max_length=128)] | None = None
    chat_history: list[Message] = Field(default_factory=list, max_length=40)
    answer_mode: Literal["Auto", "Executive Brief", "Analyst Deep Dive", "Comparison Matrix", "Action Plan", "Risk Review", "Data Highlights", "Research Synthesis"] = "Auto"
    audience: Literal["General", "Leadership", "Analyst", "Product", "Operations", "Sales"] = "General"
    response_depth: Literal["Fast", "Balanced", "Comprehensive"] = "Balanced"
    use_web_search: StrictBool = False
    uploaded_file_types: list[FileType] = Field(default_factory=list, max_length=10)

    @field_validator("uploaded_file_types", mode="before")
    @classmethod
    def normalize_file_types(cls, value):
        if isinstance(value, list):
            return [item.lower() if isinstance(item, str) else item for item in value]
        return value

    @field_validator("chat_history")
    @classmethod
    def bound_history(cls, messages: list[Message]) -> list[Message]:
        if sum(len(message.content) for message in messages) > 32000:
            raise ValueError("History exceeds the character budget")
        return messages


class ProviderRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: Literal["gemini", "ollama"]
