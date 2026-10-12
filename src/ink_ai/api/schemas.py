from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    prompt: str = Field(..., examples=["Merhaba, sen kimsin?"])
    max_tokens: int = Field(default=512, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)

class ChatResponse(BaseModel):
    response: str
    status: str = "success"