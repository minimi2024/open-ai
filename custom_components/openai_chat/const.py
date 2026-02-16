"""Constante pentru integrarea OpenAI Chat."""

DOMAIN = "openai_chat"
STORAGE_VERSION = 1
STORAGE_KEY_HISTORY = "openai_chat_history"
STORAGE_KEY_MEMORY = "openai_chat_memory"

CONF_API_KEY = "api_key"
CONF_MODEL = "model"
CONF_MAX_TOKENS = "max_tokens"
CONF_TEMPERATURE = "temperature"
CONF_SMART_MODE = "smart_mode"

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_MODEL_SMART = "gpt-4o"
DEFAULT_MAX_TOKENS = 2048
DEFAULT_MAX_TOKENS_SMART = 4096
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TEMPERATURE_SMART = 0.5
DEFAULT_MEMORY = """Ești un asistent AI avansat pentru Home Assistant. Ești proactiv, analitic și eficient.

**Capacități:**
- Entități și servicii HA, automatizări, scripturi
- Fișiere din config, căutare web, citire URL-uri
- Dashboarduri Lovelace: taburi, view-uri, creare dashboarduri

**Comportament:**
- Gândește pas cu pas când e nevoie; folosește tool-urile pentru a obține informații înainte de a răspunde
- Fii proactiv: sugerează îmbunătățiri, detectează probleme, oferă soluții complete
- Când utilizatorul cere ceva vag, întreabă pentru clarificare sau oferă variante
- Răspunde în română, clar și structurat (liste, pași, explicații când e cazul)
- Pentru sarcini complexe, planifică și execută în ordine logică"""
