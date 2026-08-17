
from pydantic import BaseModel
from typing import Literal, Any

Confidence = Literal["high", "medium", "low", "insufficient_data"]

class ToolResult(BaseModel):
    value: Any
    confidence: Confidence
    explanation: str
    metadata: dict