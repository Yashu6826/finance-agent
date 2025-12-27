import os
import logging
from typing import Optional
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

logger.info("🔧 Loading configuration...")

# Optional import: frontend (Streamlit) deployment may not include backend LLM deps
try:
    from langchain_google_genai import GoogleGenerativeAI, ChatGoogleGenerativeAI  # type: ignore
    _GENAI_AVAILABLE = True
    logger.info("✅ langchain-google-genai package imported successfully")
except ModuleNotFoundError as e:
    GoogleGenerativeAI = None  # type: ignore
    ChatGoogleGenerativeAI = None  # type: ignore
    _GENAI_AVAILABLE = False
    logger.error(f"❌ langchain-google-genai not found: {e}")
except Exception as e:
    GoogleGenerativeAI = None  # type: ignore
    ChatGoogleGenerativeAI = None  # type: ignore
    _GENAI_AVAILABLE = False
    logger.error(f"❌ Error importing langchain-google-genai: {e}")

load_dotenv()
logger.info("✅ Environment variables loaded from .env file")


class ConfigurationError(Exception):
    pass


def get_google_api_key() -> str:
    """Get Google API key from environment variables."""
    logger.debug("📝 Attempting to retrieve GOOGLE_API_KEY...")
    api_key = os.getenv('GOOGLE_API_KEY')

    if not api_key:
        logger.error("❌ GOOGLE_API_KEY environment variable not found")
        raise ConfigurationError(
            "Google API key not found. Please set GOOGLE_API_KEY environment variable."
        )

    if api_key == "your_actual_google_api_key_here":
        logger.error("❌ GOOGLE_API_KEY is still a placeholder value")
        raise ConfigurationError(
            "Google API key is still set to placeholder value. Please configure with your actual API key."
        )

    logger.info(f"✅ GOOGLE_API_KEY found (length: {len(api_key)} chars)")
    logger.debug(f"   Key prefix: {api_key[:10]}..." if len(api_key) > 10 else f"   Key: {api_key}")
    return api_key


def get_llm(model: str = "gemini-2.5-flash-lite",
           temperature: float = 0.3,
           max_output_tokens: int = 2048):  # return type conditional
    """Get configured Google Generative AI LLM instance.

    This function is safe to call even if langchain-google-genai is not installed.
    In that case it raises a ConfigurationError with a clear message instead of
    causing an immediate ModuleNotFoundError during import of this module.
    """
    logger.info(f"🚀 Initializing GoogleGenerativeAI with model: {model}")
    
    if not _GENAI_AVAILABLE or GoogleGenerativeAI is None:  # type: ignore
        logger.error("❌ GoogleGenerativeAI not available - dependency issue")
        raise ConfigurationError(
            "langchain-google-genai dependency not installed. Install backend requirements or add 'langchain-google-genai' to requirements.txt for LLM features."
        )

    try:
        logger.debug("📝 Retrieving Google API key...")
        api_key = get_google_api_key()
        
        logger.debug(f"🔌 Creating GoogleGenerativeAI instance with settings:")
        logger.debug(f"   - model: {model}")
        logger.debug(f"   - temperature: {temperature}")
        logger.debug(f"   - max_output_tokens: {max_output_tokens}")
        
        llm = GoogleGenerativeAI(  # type: ignore
            model=model,
            google_api_key=api_key,
            temperature=temperature,
            max_output_tokens=max_output_tokens
        )
        logger.info("✅ GoogleGenerativeAI initialized successfully")
        return llm
        
    except ConfigurationError as e:
        logger.error(f"❌ Configuration error: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Error initializing GoogleGenerativeAI: {type(e).__name__}: {e}")
        logger.error(f"   Full error details: {str(e)}")
        import traceback
        logger.error(f"   Traceback: {traceback.format_exc()}")
        raise


def get_chat_llm(model: str = "gemini-2.5-flash-lite",
                temperature: float = 0.1,
                max_output_tokens: int = 1024):  # return type conditional
    """Get configured Google Generative AI Chat LLM instance.

    Raises ConfigurationError with guidance if dependency missing.
    """
    logger.info(f"🚀 Initializing ChatGoogleGenerativeAI with model: {model}")
    
    if not _GENAI_AVAILABLE or ChatGoogleGenerativeAI is None:  # type: ignore
        logger.error("❌ ChatGoogleGenerativeAI not available - dependency issue")
        raise ConfigurationError(
            "langchain-google-genai dependency not installed. Install backend requirements or add 'langchain-google-genai' to requirements.txt for chat LLM features."
        )

    try:
        logger.debug("📝 Retrieving Google API key...")
        api_key = get_google_api_key()
        
        logger.debug(f"🔌 Creating ChatGoogleGenerativeAI instance with settings:")
        logger.debug(f"   - model: {model}")
        logger.debug(f"   - temperature: {temperature}")
        logger.debug(f"   - max_output_tokens: {max_output_tokens}")
        logger.debug(f"   - max_retries: 3")
        logger.debug(f"   - timeout: 120")
        
        llm = ChatGoogleGenerativeAI(  # type: ignore
            model=model,
            google_api_key=api_key,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            max_retries=3,
            timeout=120
        )
        logger.info("✅ ChatGoogleGenerativeAI initialized successfully")
        return llm
        
    except ConfigurationError as e:
        logger.error(f"❌ Configuration error: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Error initializing ChatGoogleGenerativeAI: {type(e).__name__}: {e}")
        logger.error(f"   Full error details: {str(e)}")
        import traceback
        logger.error(f"   Traceback: {traceback.format_exc()}")
        raise


def get_newsapi_key() -> str:
    """Get NewsAPI key from environment variables."""
    api_key = os.getenv('NEWSAPI_KEY')

    if not api_key:
        raise ConfigurationError(
            "NewsAPI key not found. Please set NEWSAPI_KEY environment variable."
        )

    if api_key == "your_actual_newsapi_key_here":
        raise ConfigurationError(
            "NewsAPI key is still set to placeholder value. Please configure with your actual API key."
        )

    return api_key


def validate_configuration() -> bool:
    """Validate all required configuration is present and valid."""
    logger.info("🔍 Validating configuration...")
    try:
        logger.debug("   - Checking GOOGLE_API_KEY...")
        get_google_api_key()
        logger.info("   ✅ GOOGLE_API_KEY valid")
        
        logger.debug("   - Checking NEWSAPI_KEY...")
        get_newsapi_key()
        logger.info("   ✅ NEWSAPI_KEY valid")
        
        logger.info("✅ Configuration validation passed")
        return True
    except Exception as e:
        logger.error(f"❌ Configuration validation failed: {e}")
        raise


DEFAULT_MODEL = "gemini-2.5-flash-lite"
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS = 2048
DEFAULT_CHAT_TEMPERATURE = 0.1
DEFAULT_CHAT_MAX_TOKENS = 1024