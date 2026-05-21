from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import get_auth_service, get_current_user, require_admin
from app.domain.auth import AuthToken, CurrentUser, LoginRequest, RegisterRequest
from app.services.auth_service import (
    AuthService,
    DuplicateUserError,
    InvalidCredentialsError,
)


router = APIRouter(tags=["auth"])


@router.post("/auth/register", response_model=CurrentUser)
def register_user(
    request: RegisterRequest,
    auth_service: AuthService = Depends(get_auth_service),
) -> CurrentUser:
    try:
        return auth_service.register_user(
            email=request.email,
            password=request.password,
            role=request.role,
        )
    except DuplicateUserError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="User already exists.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Role must be user or admin.",
        ) from exc


@router.post("/auth/login", response_model=AuthToken)
def login_user(
    request: LoginRequest,
    auth_service: AuthService = Depends(get_auth_service),
) -> AuthToken:
    try:
        user = auth_service.authenticate_user(request.email, request.password)
    except InvalidCredentialsError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        ) from exc
    return AuthToken(
        access_token=auth_service.create_access_token(user),
        role=user.role,
        email=user.email,
    )


@router.get("/auth/me", response_model=CurrentUser)
def read_current_user(current_user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
    return current_user


@router.get("/admin/ping")
def admin_ping(current_user: CurrentUser = Depends(require_admin)) -> dict[str, str]:
    return {"status": "ok", "role": current_user.role}
