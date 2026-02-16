"""
API Key Validation Utility
"""
import os
from typing import Optional, Tuple


class APIKeyValidator:
    """Validates and checks for required API keys."""
    
    @staticmethod
    def check_google_api_key() -> Tuple[bool, Optional[str]]:
        """
        Check if Google API key is present in environment.
        
        Returns:
            Tuple of (is_valid, api_key)
            - is_valid: True if key exists and is not empty
            - api_key: The API key value or None
        """
        api_key = os.getenv("GOOGLE_API_KEY")
        
        if api_key and api_key.strip():
            return True, api_key
        return False, None
    
    @staticmethod
    def check_langsmith_api_key() -> Tuple[bool, Optional[str]]:
        """
        Check if LangSmith API key is present in environment (optional).
        
        Returns:
            Tuple of (is_valid, api_key)
            - is_valid: True if key exists and is not empty
            - api_key: The API key value or None
        """
        api_key = os.getenv("LANGSMITH_API_KEY")
        
        if api_key and api_key.strip():
            return True, api_key
        return False, None
    
    @staticmethod
    def validate_all_keys() -> dict:
        """
        Validate all required and optional API keys.
        
        Returns:
            Dict with validation results:
            {
                'google': {'valid': bool, 'key': str or None, 'required': True},
                'langsmith': {'valid': bool, 'key': str or None, 'required': False}
            }
        """
        google_valid, google_key = APIKeyValidator.check_google_api_key()
        langsmith_valid, langsmith_key = APIKeyValidator.check_langsmith_api_key()
        
        return {
            'google': {
                'valid': google_valid,
                'key': google_key,
                'required': True,
                'name': 'Google API Key (Gemini)'
            },
            'langsmith': {
                'valid': langsmith_valid,
                'key': langsmith_key,
                'required': False,
                'name': 'LangSmith API Key (Optional)'
            }
        }
    
    @staticmethod
    def get_validation_message() -> str:
        """
        Get a user-friendly validation message for all API keys.
        
        Returns:
            Formatted string with validation status
        """
        results = APIKeyValidator.validate_all_keys()
        messages = []
        
        for key_type, info in results.items():
            status = "✅" if info['valid'] else "❌"
            required = "(Required)" if info['required'] else "(Optional)"
            messages.append(f"{status} {info['name']} {required}")
        
        return "\n".join(messages)
    
    @staticmethod
    def raise_if_missing_required() -> None:
        """
        Raise an error if required API keys are missing.
        
        Raises:
            ValueError: If Google API key is missing
        """
        google_valid, _ = APIKeyValidator.check_google_api_key()
        
        if not google_valid:
            raise ValueError(
                "GOOGLE_API_KEY is required but not found in environment.\n"
                "Please set it in your .env file:\n"
                "GOOGLE_API_KEY=your_api_key_here"
            )


# Convenience functions for quick checks
def is_google_api_key_present() -> bool:
    """Quick check if Google API key is present."""
    valid, _ = APIKeyValidator.check_google_api_key()
    return valid


def get_google_api_key() -> Optional[str]:
    """Get Google API key if present, None otherwise."""
    _, key = APIKeyValidator.check_google_api_key()
    return key


def is_langsmith_enabled() -> bool:
    """Check if LangSmith is enabled (API key present)."""
    valid, _ = APIKeyValidator.check_langsmith_api_key()
    return valid
