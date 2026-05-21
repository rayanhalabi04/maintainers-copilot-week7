from typing import Literal

from pydantic import BaseModel, Field


Role = Literal["user", "admin"]


class RegisterRequest(BaseModel):
    email: str = Field(..., min_length=3)
    password: str = Field(..., min_length=6)
    role: Role = "user"


class LoginRequest(BaseModel):
    email: str = Field(..., min_length=3)
    password: str = Field(..., min_length=1)


class AuthToken(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: Role
    email: str


class CurrentUser(BaseModel):
    email: str
    role: Role
