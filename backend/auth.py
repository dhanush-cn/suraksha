import os
import time
import json
import base64
import hmac
import hashlib
from typing import Dict, Any, Optional
from fastapi import HTTPException, Security, Depends, status
from fastapi.security import OAuth2PasswordBearer

SECRET_KEY = os.getenv("JWT_SECRET", "rockfallguard_super_secret_jwt_key_2026_pit_safety")
ALGORITHM = "HS256"
TOKEN_EXPIRE_SECONDS = 86400 * 7 # 7 Days

# Standard OAuth2 Bearer Token Scheme (Points to /api/auth/login)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)

def base64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b'=').decode('utf-8')

def base64url_decode(data: str) -> bytes:
    padding = 4 - (len(data) % 4)
    if padding != 4:
        data += '=' * padding
    return base64.urlsafe_b64decode(data.encode('utf-8'))

def create_access_token(payload: Dict[str, Any]) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    header_json = json.dumps(header, separators=(',', ':')).encode('utf-8')
    header_b64 = base64url_encode(header_json)
    
    payload_copy = payload.copy()
    payload_copy["exp"] = int(time.time()) + TOKEN_EXPIRE_SECONDS
    payload_json = json.dumps(payload_copy, separators=(',', ':')).encode('utf-8')
    payload_b64 = base64url_encode(payload_json)
    
    signing_input = f"{header_b64}.{payload_b64}".encode('utf-8')
    signature = hmac.new(SECRET_KEY.encode('utf-8'), signing_input, hashlib.sha256).digest()
    signature_b64 = base64url_encode(signature)
    
    return f"{header_b64}.{payload_b64}.{signature_b64}"

def verify_access_token(token: str) -> Dict[str, Any]:
    try:
        parts = token.split('.')
        if len(parts) != 3:
            raise ValueError("Invalid token format")
            
        header_b64, payload_b64, signature_b64 = parts
        signing_input = f"{header_b64}.{payload_b64}".encode('utf-8')
        expected_sig = hmac.new(SECRET_KEY.encode('utf-8'), signing_input, hashlib.sha256).digest()
        actual_sig = base64url_decode(signature_b64)
        
        if not hmac.compare_digest(expected_sig, actual_sig):
            raise ValueError("Invalid token signature")
            
        payload = json.loads(base64url_decode(payload_b64).decode('utf-8'))
        if payload.get("exp", 0) < time.time():
            raise ValueError("Token expired")
            
        return payload
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"OAuth2 Authentication Failed: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )

# FastAPI Dependency for extracting authenticated user context via OAuth2 / JWT
def get_current_user(token: Optional[str] = Depends(oauth2_scheme)) -> Dict[str, Any]:
    if not token:
        # Default unauthenticated fallback for initial page load / guest preview
        return {
            "id": 0,
            "username": "guest_operator",
            "role": "admin",
            "mine_id": None,
            "company_name": "Global Mining Admin (Demo Preview)"
        }
    user_payload = verify_access_token(token)
    return user_payload

# RBAC Tenant Validation Helper
def enforce_tenant_access(user: Dict[str, Any], target_mine_id: int):
    role = user.get("role", "user")
    user_mine_id = user.get("mine_id")
    
    if role == "admin":
        return True # Admin bypasses tenant isolation
        
    if role == "user":
        if user_mine_id is not None and int(user_mine_id) != int(target_mine_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Tenant Access Denied: User '{user.get('username')}' is restricted to Mine ID {user_mine_id} and cannot access Mine ID {target_mine_id}."
            )
        return True
        
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Unauthorized user role.")

def enforce_admin_only(user: Dict[str, Any]):
    if user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin Privilege Required: Only Admin users can perform this operation."
        )
    return True
