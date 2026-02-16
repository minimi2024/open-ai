"""Constante pentru integrarea OpenAI Chat."""

DOMAIN = "openai_chat"
STORAGE_VERSION = 1
STORAGE_KEY_HISTORY = "openai_chat_history"
STORAGE_KEY_MEMORY = "openai_chat_memory"

CONF_API_KEY = "api_key"
CONF_MODEL = "model"
CONF_MAX_TOKENS = "max_tokens"
CONF_TEMPERATURE = "temperature"

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MEMORY = """Ești un asistent inteligent pentru Home Assistant cu acces complet la sistem.

Poți:
- Vedea starea tuturor entităților (lumină, switch, senzori, clima, etc.)
- Apela orice serviciu (porni/opri lumină, declanșa automatizări, rula scripturi)
- Citi și lista fișiere din config (configuration.yaml, automations.yaml, etc.)
- Controla dispozitive, automatizări, scripturi

Folosește tool-urile disponibile pentru a obține informații și executa acțiuni. Răspunde în română, concis și util."""
