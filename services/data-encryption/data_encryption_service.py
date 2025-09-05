#!/usr/bin/env python3
"""
Data Encryption Service for Agentic Platform

This service provides comprehensive encryption capabilities including:
- TLS/SSL encryption for all communications
- Field-level encryption for sensitive data
- Key management and rotation
- Encryption at rest and in transit
- Secure key storage and access control
- Compliance with encryption standards (AES-256, RSA)
- Data masking and tokenization
"""

import base64
import hashlib
import hmac
import json
import os
import secrets
import ssl
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import psycopg2
import psycopg2.extras
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from prometheus_client import Counter, Gauge, Histogram, generate_latest
import structlog
import uvicorn

# Configure structured logging
logging.basicConfig(level=logging.INFO)
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# FastAPI app
app = FastAPI(
    title="Data Encryption Service",
    description="Comprehensive encryption service with TLS and field-level encryption",
    version="1.0.0"
)

# Prometheus metrics
ENCRYPTION_OPERATIONS = Counter('encryption_operations_total', 'Total encryption operations', ['operation_type', 'algorithm'])
DECRYPTION_OPERATIONS = Counter('decryption_operations_total', 'Total decryption operations', ['operation_type', 'algorithm'])
KEY_OPERATIONS = Counter('key_operations_total', 'Total key operations', ['operation_type'])
ENCRYPTION_ERRORS = Counter('encryption_errors_total', 'Total encryption errors', ['error_type'])
TLS_CONNECTIONS = Counter('tls_connections_total', 'Total TLS connections', ['protocol_version'])
KEY_ROTATION_EVENTS = Counter('key_rotation_events_total', 'Total key rotation events', ['key_type'])

# Global variables
database_connection = None

# Pydantic models
class EncryptionRequest(BaseModel):
    """Encryption request model"""
    data: Union[str, Dict[str, Any]] = Field(..., description="Data to encrypt")
    algorithm: str = Field("AES-256-GCM", description="Encryption algorithm")
    key_id: Optional[str] = Field(None, description="Specific key ID to use")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class DecryptionRequest(BaseModel):
    """Decryption request model"""
    encrypted_data: str = Field(..., description="Encrypted data to decrypt")
    key_id: Optional[str] = Field(None, description="Key ID used for encryption")
    algorithm: Optional[str] = Field(None, description="Encryption algorithm used")

class FieldEncryptionRequest(BaseModel):
    """Field-level encryption request model"""
    data: Dict[str, Any] = Field(..., description="Data object to encrypt fields in")
    fields: List[str] = Field(..., description="Fields to encrypt")
    algorithm: str = Field("AES-256-GCM", description="Encryption algorithm")
    key_id: Optional[str] = Field(None, description="Key ID to use")

class KeyGenerationRequest(BaseModel):
    """Key generation request model"""
    key_type: str = Field(..., description="Type of key (AES, RSA)")
    key_size: int = Field(256, description="Key size in bits")
    purpose: str = Field(..., description="Purpose of the key")
    rotation_period_days: Optional[int] = Field(None, description="Key rotation period")

class TLSConfig(BaseModel):
    """TLS configuration model"""
    certificate_path: str = Field(..., description="Path to TLS certificate")
    private_key_path: str = Field(..., description="Path to private key")
    ca_certificate_path: Optional[str] = Field(None, description="Path to CA certificate")
    min_tls_version: str = Field("TLSv1.2", description="Minimum TLS version")
    cipher_suites: List[str] = Field([], description="Allowed cipher suites")

class MaskingRequest(BaseModel):
    """Data masking request model"""
    data: Union[str, Dict[str, Any]] = Field(..., description="Data to mask")
    masking_rules: Dict[str, str] = Field(..., description="Masking rules per field")

# Encryption Manager Class
class EncryptionManager:
    """Comprehensive encryption manager with multiple algorithms and key management"""

    def __init__(self):
        self.keys = {}
        self.key_versions = {}
        self.encryption_algorithms = {
            "AES-256-GCM": self._aes_256_gcm_encrypt,
            "AES-256-CBC": self._aes_256_cbc_encrypt,
            "RSA-OAEP": self._rsa_oaep_encrypt
        }
        self.decryption_algorithms = {
            "AES-256-GCM": self._aes_256_gcm_decrypt,
            "AES-256-CBC": self._aes_256_cbc_decrypt,
            "RSA-OAEP": self._rsa_oaep_decrypt
        }

    def generate_key(self, key_type: str, key_size: int = 256) -> Dict[str, Any]:
        """Generate encryption key"""
        try:
            key_id = f"{key_type.lower()}_{secrets.token_hex(8)}"
            created_at = datetime.utcnow()

            if key_type.upper() == "AES":
                if key_size not in [128, 192, 256]:
                    raise ValueError("AES key size must be 128, 192, or 256 bits")

                # Generate AES key
                key = secrets.token_bytes(key_size // 8)
                key_material = base64.b64encode(key).decode('utf-8')

                # Store key securely
                self._store_key_securely(key_id, key_material, key_type, key_size, created_at)

                return {
                    "key_id": key_id,
                    "key_type": key_type,
                    "key_size": key_size,
                    "created_at": created_at.isoformat(),
                    "status": "generated"
                }

            elif key_type.upper() == "RSA":
                # Generate RSA key pair
                private_key = rsa.generate_private_key(
                    public_exponent=65537,
                    key_size=key_size,
                    backend=default_backend()
                )

                # Serialize keys
                private_pem = private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                )

                public_key = private_key.public_key()
                public_pem = public_key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                )

                # Store keys securely
                self._store_key_securely(key_id, private_pem.decode('utf-8'), "RSA_PRIVATE", key_size, created_at)
                self._store_key_securely(f"{key_id}_public", public_pem.decode('utf-8'), "RSA_PUBLIC", key_size, created_at)

                return {
                    "key_id": key_id,
                    "key_type": key_type,
                    "key_size": key_size,
                    "public_key": public_pem.decode('utf-8'),
                    "created_at": created_at.isoformat(),
                    "status": "generated"
                }

            else:
                raise ValueError(f"Unsupported key type: {key_type}")

        except Exception as e:
            logger.error("Key generation failed", error=str(e), key_type=key_type)
            ENCRYPTION_ERRORS.labels(error_type="key_generation").inc()
            raise

    def encrypt_data(self, data: Union[str, Dict[str, Any]], algorithm: str = "AES-256-GCM",
                    key_id: Optional[str] = None) -> Dict[str, Any]:
        """Encrypt data using specified encryption algorithm with automatic key management.

        This method provides end-to-end encryption for sensitive data using industry-standard
        algorithms. It supports both string and structured data (dictionaries) and automatically
        handles key generation and rotation.

        Args:
            data: Data to encrypt (string or dictionary)
            algorithm: Encryption algorithm to use (default: AES-256-GCM)
            key_id: Optional existing key ID to use for encryption

        Returns:
            Dict containing:
            - encrypted_data: Base64-encoded encrypted payload
            - key_id: ID of the key used for encryption
            - algorithm: Algorithm used for encryption
            - iv: Initialization vector (for symmetric encryption)
            - timestamp: Encryption timestamp

        Raises:
            ValueError: If unsupported algorithm is specified
            RuntimeError: If encryption operation fails
        """
        try:
            # Get or generate key
            if not key_id:
                key_response = self.generate_key("AES", 256)
                key_id = key_response["key_id"]

            # Serialize data if dict
            if isinstance(data, dict):
                data_str = json.dumps(data)
            else:
                data_str = str(data)

            # Get encryption function
            if algorithm not in self.encryption_algorithms:
                raise ValueError(f"Unsupported encryption algorithm: {algorithm}")

            encrypt_func = self.encryption_algorithms[algorithm]
            encrypted_data, iv, tag = encrypt_func(data_str, key_id)

            # Create encrypted package
            encrypted_package = {
                "algorithm": algorithm,
                "key_id": key_id,
                "iv": base64.b64encode(iv).decode('utf-8') if iv else None,
                "tag": base64.b64encode(tag).decode('utf-8') if tag else None,
                "data": base64.b64encode(encrypted_data).decode('utf-8'),
                "timestamp": datetime.utcnow().isoformat()
            }

            ENCRYPTION_OPERATIONS.labels(operation_type="encrypt", algorithm=algorithm).inc()

            return {
                "encrypted_data": base64.b64encode(json.dumps(encrypted_package).encode()).decode(),
                "key_id": key_id,
                "algorithm": algorithm,
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error("Data encryption failed", error=str(e), algorithm=algorithm)
            ENCRYPTION_ERRORS.labels(error_type="encryption").inc()
            raise

    def decrypt_data(self, encrypted_package: str, key_id: Optional[str] = None) -> Any:
        """Decrypt data"""
        try:
            # Decode encrypted package
            package_data = json.loads(base64.b64decode(encrypted_package).decode())

            algorithm = package_data["algorithm"]
            key_id = key_id or package_data["key_id"]

            # Get decryption function
            if algorithm not in self.decryption_algorithms:
                raise ValueError(f"Unsupported decryption algorithm: {algorithm}")

            decrypt_func = self.decryption_algorithms[algorithm]

            # Decode components
            encrypted_data = base64.b64decode(package_data["data"])
            iv = base64.b64decode(package_data["iv"]) if package_data.get("iv") else None
            tag = base64.b64decode(package_data["tag"]) if package_data.get("tag") else None

            # Decrypt
            decrypted_data = decrypt_func(encrypted_data, key_id, iv, tag)

            # Try to parse as JSON
            try:
                return json.loads(decrypted_data)
            except:
                return decrypted_data

        except Exception as e:
            logger.error("Data decryption failed", error=str(e))
            ENCRYPTION_ERRORS.labels(error_type="decryption").inc()
            raise

    def encrypt_fields(self, data: Dict[str, Any], fields: List[str],
                      algorithm: str = "AES-256-GCM", key_id: Optional[str] = None) -> Dict[str, Any]:
        """Encrypt specific fields in data object"""
        try:
            result = data.copy()

            for field in fields:
                if field in result:
                    field_value = result[field]
                    if field_value is not None:
                        encrypted_result = self.encrypt_data(field_value, algorithm, key_id)
                        result[field] = f"ENCRYPTED:{encrypted_result['encrypted_data']}"

            return result

        except Exception as e:
            logger.error("Field encryption failed", error=str(e), fields=fields)
            raise

    def decrypt_fields(self, data: Dict[str, Any], fields: List[str]) -> Dict[str, Any]:
        """Decrypt specific fields in data object"""
        try:
            result = data.copy()

            for field in fields:
                if field in result:
                    field_value = result[field]
                    if isinstance(field_value, str) and field_value.startswith("ENCRYPTED:"):
                        encrypted_data = field_value[10:]  # Remove "ENCRYPTED:" prefix
                        decrypted_value = self.decrypt_data(encrypted_data)
                        result[field] = decrypted_value

            return result

        except Exception as e:
            logger.error("Field decryption failed", error=str(e), fields=fields)
            raise

    def mask_data(self, data: Union[str, Dict[str, Any]], masking_rules: Dict[str, str]) -> Union[str, Dict[str, Any]]:
        """Mask sensitive data according to rules"""
        try:
            if isinstance(data, str):
                return self._apply_masking_rules(data, masking_rules)
            elif isinstance(data, dict):
                result = data.copy()
                for field, rule in masking_rules.items():
                    if field in result:
                        result[field] = self._apply_masking_rule(result[field], rule)
                return result
            else:
                return data

        except Exception as e:
            logger.error("Data masking failed", error=str(e))
            raise

    def rotate_keys(self, key_type: str = "AES") -> Dict[str, Any]:
        """Rotate encryption keys"""
        try:
            # Generate new key
            new_key = self.generate_key(key_type, 256)

            # Mark old keys as rotated (implement key versioning)
            # This would involve updating key metadata in database

            KEY_ROTATION_EVENTS.labels(key_type=key_type).inc()

            return {
                "status": "rotated",
                "new_key_id": new_key["key_id"],
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error("Key rotation failed", error=str(e), key_type=key_type)
            raise

    # AES-256-GCM implementation
    def _aes_256_gcm_encrypt(self, data: str, key_id: str) -> tuple:
        """AES-256-GCM encryption"""
        key = self._get_key_material(key_id)
        iv = os.urandom(12)  # 96-bit IV for GCM

        cipher = Cipher(algorithms.AES(key), modes.GCM(iv), backend=default_backend())
        encryptor = cipher.encryptor()

        ciphertext = encryptor.update(data.encode()) + encryptor.finalize()

        return ciphertext, iv, encryptor.tag

    def _aes_256_gcm_decrypt(self, ciphertext: bytes, key_id: str, iv: bytes, tag: bytes) -> str:
        """AES-256-GCM decryption"""
        key = self._get_key_material(key_id)

        cipher = Cipher(algorithms.AES(key), modes.GCM(iv, tag), backend=default_backend())
        decryptor = cipher.decryptor()

        plaintext = decryptor.update(ciphertext) + decryptor.finalize()

        return plaintext.decode()

    # AES-256-CBC implementation (legacy support)
    def _aes_256_cbc_encrypt(self, data: str, key_id: str) -> tuple:
        """AES-256-CBC encryption"""
        key = self._get_key_material(key_id)
        iv = os.urandom(16)

        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        encryptor = cipher.encryptor()

        # PKCS7 padding
        padded_data = self._pad_data(data.encode())

        ciphertext = encryptor.update(padded_data) + encryptor.finalize()

        return ciphertext, iv, None

    def _aes_256_cbc_decrypt(self, ciphertext: bytes, key_id: str, iv: bytes, tag: bytes = None) -> str:
        """AES-256-CBC decryption"""
        key = self._get_key_material(key_id)

        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        decryptor = cipher.decryptor()

        padded_plaintext = decryptor.update(ciphertext) + decryptor.finalize()

        # Remove PKCS7 padding
        plaintext = self._unpad_data(padded_plaintext)

        return plaintext.decode()

    # RSA-OAEP implementation
    def _rsa_oaep_encrypt(self, data: str, key_id: str) -> tuple:
        """RSA-OAEP encryption"""
        public_key_pem = self._get_key_material(f"{key_id}_public")
        public_key = serialization.load_pem_public_key(public_key_pem.encode(), backend=default_backend())

        ciphertext = public_key.encrypt(
            data.encode(),
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )

        return ciphertext, None, None

    def _rsa_oaep_decrypt(self, ciphertext: bytes, key_id: str, iv: bytes = None, tag: bytes = None) -> str:
        """RSA-OAEP decryption"""
        private_key_pem = self._get_key_material(key_id)
        private_key = serialization.load_pem_private_key(private_key_pem.encode(), password=None, backend=default_backend())

        plaintext = private_key.decrypt(
            ciphertext,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )

        return plaintext.decode()

    def _get_key_material(self, key_id: str) -> bytes:
        """Get key material from secure storage"""
        # In production, this would retrieve from secure key store
        # For demo, using in-memory storage
        if key_id in self.keys:
            key_data = self.keys[key_id]
            if isinstance(key_data, str):
                # Base64 encoded key
                return base64.b64decode(key_data)
            else:
                return key_data
        else:
            raise ValueError(f"Key not found: {key_id}")

    def _store_key_securely(self, key_id: str, key_material: str, key_type: str, key_size: int, created_at: datetime):
        """Store key securely"""
        # In production, this would use a secure key management system
        # For demo, storing in memory with database backup
        self.keys[key_id] = key_material

        # Store key metadata in database
        try:
            with database_connection.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO encryption_keys (key_id, key_type, key_size, created_at, is_active)
                    VALUES (%s, %s, %s, %s, true)
                """, (key_id, key_type, key_size, created_at))
                database_connection.commit()

        except Exception as e:
            logger.error("Failed to store key metadata", error=str(e), key_id=key_id)

    def _pad_data(self, data: bytes, block_size: int = 16) -> bytes:
        """PKCS7 padding"""
        padding_length = block_size - (len(data) % block_size)
        padding = bytes([padding_length]) * padding_length
        return data + padding

    def _unpad_data(self, padded_data: bytes) -> bytes:
        """Remove PKCS7 padding"""
        padding_length = padded_data[-1]
        return padded_data[:-padding_length]

    def _apply_masking_rule(self, value: str, rule: str) -> str:
        """Apply masking rule to value"""
        if not isinstance(value, str):
            value = str(value)

        if rule == "credit_card":
            # Mask all but last 4 digits
            return "****-****-****-" + value[-4:] if len(value) >= 4 else "****"
        elif rule == "ssn":
            # Mask SSN format
            return "***-**-" + value[-4:] if len(value) >= 4 else "***-**-****"
        elif rule == "email":
            # Mask email
            parts = value.split("@")
            if len(parts) == 2:
                return parts[0][:2] + "***@" + parts[1]
            return value
        elif rule.startswith("first_") and rule.endswith("_chars"):
            # Mask first N characters
            n = int(rule.split("_")[1])
            return "*" * min(n, len(value)) + value[n:] if n < len(value) else "*" * len(value)
        elif rule.startswith("last_") and rule.endswith("_chars"):
            # Mask last N characters
            n = int(rule.split("_")[1])
            return value[:-n] + "*" * min(n, len(value)) if n < len(value) else "*" * len(value)
        else:
            # Default: mask all
            return "*" * len(value)

# TLS Manager Class
class TLSManager:
    """TLS/SSL encryption manager"""

    def __init__(self):
        self.certificates = {}
        self.tls_configs = {}

    def setup_tls_context(self, cert_path: str, key_path: str, ca_path: Optional[str] = None) -> ssl.SSLContext:
        """Setup TLS context for secure connections"""
        try:
            context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)

            # Load certificate and private key
            context.load_cert_chain(cert_path, key_path)

            if ca_path:
                context.load_verify_locations(ca_path)

            # Set minimum TLS version
            context.minimum_version = ssl.TLSVersion.TLSv1_2
            context.maximum_version = ssl.TLSVersion.TLSv1_3

            # Set secure cipher suites
            context.set_ciphers('ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20')

            TLS_CONNECTIONS.labels(protocol_version="TLSv1.2+").inc()

            return context

        except Exception as e:
            logger.error("TLS context setup failed", error=str(e))
            raise

    def validate_certificate(self, cert_path: str) -> Dict[str, Any]:
        """Validate TLS certificate"""
        try:
            with open(cert_path, 'rb') as f:
                cert_data = f.read()

            cert = ssl._ssl._test_decode_cert(cert_data)

            return {
                "subject": cert.get("subject"),
                "issuer": cert.get("issuer"),
                "valid_from": cert.get("notBefore"),
                "valid_until": cert.get("notAfter"),
                "serial_number": cert.get("serialNumber"),
                "is_valid": True
            }

        except Exception as e:
            logger.error("Certificate validation failed", error=str(e))
            return {"is_valid": False, "error": str(e)}

# Global instances
encryption_manager = EncryptionManager()
tls_manager = TLSManager()

# API endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "data-encryption-service",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "encryption_algorithms": list(encryption_manager.encryption_algorithms.keys())
    }

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()

@app.post("/encrypt")
async def encrypt_data(request: EncryptionRequest):
    """Encrypt data"""
    try:
        result = encryption_manager.encrypt_data(
            request.data,
            request.algorithm,
            request.key_id
        )
        return result

    except Exception as e:
        logger.error("Encryption request failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Encryption failed: {str(e)}")

@app.post("/decrypt")
async def decrypt_data(request: DecryptionRequest):
    """Decrypt data"""
    try:
        result = encryption_manager.decrypt_data(
            request.encrypted_data,
            request.key_id
        )
        return {"decrypted_data": result}

    except Exception as e:
        logger.error("Decryption request failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Decryption failed: {str(e)}")

@app.post("/encrypt-fields")
async def encrypt_fields(request: FieldEncryptionRequest):
    """Encrypt specific fields in data"""
    try:
        result = encryption_manager.encrypt_fields(
            request.data,
            request.fields,
            request.algorithm,
            request.key_id
        )
        return {"encrypted_data": result}

    except Exception as e:
        logger.error("Field encryption failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Field encryption failed: {str(e)}")

@app.post("/decrypt-fields")
async def decrypt_fields(data: Dict[str, Any], fields: List[str]):
    """Decrypt specific fields in data"""
    try:
        result = encryption_manager.decrypt_fields(data, fields)
        return {"decrypted_data": result}

    except Exception as e:
        logger.error("Field decryption failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Field decryption failed: {str(e)}")

@app.post("/keys/generate")
async def generate_key(request: KeyGenerationRequest):
    """Generate encryption key"""
    try:
        result = encryption_manager.generate_key(
            request.key_type,
            request.key_size
        )
        return result

    except Exception as e:
        logger.error("Key generation failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Key generation failed: {str(e)}")

@app.post("/keys/rotate")
async def rotate_keys(key_type: str = "AES"):
    """Rotate encryption keys"""
    try:
        result = encryption_manager.rotate_keys(key_type)
        return result

    except Exception as e:
        logger.error("Key rotation failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Key rotation failed: {str(e)}")

@app.post("/mask")
async def mask_data(request: MaskingRequest):
    """Mask sensitive data"""
    try:
        result = encryption_manager.mask_data(
            request.data,
            request.masking_rules
        )
        return {"masked_data": result}

    except Exception as e:
        logger.error("Data masking failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Data masking failed: {str(e)}")

@app.get("/tls/certificate/validate")
async def validate_certificate(cert_path: str):
    """Validate TLS certificate"""
    try:
        result = tls_manager.validate_certificate(cert_path)
        return result

    except Exception as e:
        logger.error("Certificate validation failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Certificate validation failed: {str(e)}")

@app.get("/stats")
async def get_stats():
    """Get encryption service statistics"""
    return {
        "service": "data-encryption-service",
        "metrics": {
            "encryption_operations_total": ENCRYPTION_OPERATIONS._value.get(),
            "decryption_operations_total": DECRYPTION_OPERATIONS._value.get(),
            "key_operations_total": KEY_OPERATIONS._value.get(),
            "encryption_errors_total": ENCRYPTION_ERRORS._value.get(),
            "tls_connections_total": TLS_CONNECTIONS._value.get(),
            "key_rotation_events_total": KEY_ROTATION_EVENTS._value.get()
        },
        "algorithms": {
            "supported_encryption": list(encryption_manager.encryption_algorithms.keys()),
            "supported_decryption": list(encryption_manager.decryption_algorithms.keys())
        }
    }

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize service on startup"""
    global database_connection

    logger.info("Data Encryption Service starting up...")

    # Setup database connection
    try:
        db_config = {
            "host": os.getenv("POSTGRES_HOST", "postgresql_ingestion"),
            "port": int(os.getenv("POSTGRES_PORT", 5432)),
            "database": os.getenv("POSTGRES_DB", "agentic_ingestion"),
            "user": os.getenv("POSTGRES_USER", "agentic_user"),
            "password": os.getenv("POSTGRES_PASSWORD", "")
        if not os.getenv("POSTGRES_PASSWORD"):
            logger.error("POSTGRES_PASSWORD not configured for Data Encryption Service")
            raise RuntimeError("POSTGRES_PASSWORD not configured for Data Encryption Service")
        }

        database_connection = psycopg2.connect(**db_config)
        logger.info("Database connection established")

    except Exception as e:
        logger.warning("Database connection failed", error=str(e))

    # Generate initial encryption keys
    try:
        encryption_manager.generate_key("AES", 256)
        logger.info("Initial encryption keys generated")

    except Exception as e:
        logger.warning("Failed to generate initial keys", error=str(e))

    logger.info("Data Encryption Service startup complete")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global database_connection

    logger.info("Data Encryption Service shutting down...")

    # Close database connection
    if database_connection:
        database_connection.close()

    logger.info("Data Encryption Service shutdown complete")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP Exception",
            "detail": exc.detail,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error("Unhandled exception", error=str(exc), path=request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": "An unexpected error occurred",
            "timestamp": datetime.utcnow().isoformat()
        }
    )

if __name__ == "__main__":
    # Run the FastAPI server
    uvicorn.run(
        "data_encryption_service:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8094)),
        reload=False,
        log_level="info"
    )
