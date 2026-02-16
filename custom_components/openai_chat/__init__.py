"""Integrare OpenAI Chat cu istoric persistent și memorie editabilă."""

from __future__ import annotations

import json
import logging
import os
import re

import openai
from homeassistant.components.http import StaticPathConfig
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_API_KEY
from homeassistant.core import HomeAssistant
from homeassistant.helpers import storage
from homeassistant.helpers.httpx_client import get_async_client

from .const import (
    CONF_SMART_MODE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MAX_TOKENS_SMART,
    DEFAULT_MEMORY,
    DEFAULT_MODEL,
    DEFAULT_MODEL_SMART,
    DEFAULT_TEMPERATURE,
    DEFAULT_TEMPERATURE_SMART,
    DOMAIN,
    STORAGE_KEY_HISTORY,
    STORAGE_KEY_MEMORY,
    STORAGE_VERSION,
)
from .http import (
    OpenAIChatView,
    OpenAIConversationsView,
    OpenAIHistoryView,
    OpenAIMemoryView,
)
from .tools import OPENAI_TOOLS, execute_tool

_LOGGER = logging.getLogger(__name__)


async def async_setup(hass: HomeAssistant, config: dict) -> bool:
    """Configurare inițială a integrării."""
    hass.data.setdefault(DOMAIN, {})

    # Înregistrare path static pentru panel
    frontend_path = os.path.join(os.path.dirname(__file__), "frontend")
    if os.path.isdir(frontend_path):
        await hass.http.async_register_static_paths(
            [StaticPathConfig("/api/openai_chat/static", frontend_path, False)]
        )

    # Înregistrare rute HTTP (coordinator-ul se obține din hass.data)
    hass.http.register_view(OpenAIChatView())
    hass.http.register_view(OpenAIHistoryView())
    hass.http.register_view(OpenAIMemoryView())
    hass.http.register_view(OpenAIConversationsView())

    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Configurare integrare din Config Entry."""
    client = openai.AsyncOpenAI(
        api_key=entry.data[CONF_API_KEY],
        http_client=get_async_client(hass),
    )

    coordinator = OpenAIChatCoordinator(hass, client, entry)
    hass.data[DOMAIN][entry.entry_id] = coordinator

    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Descărcare integrare."""
    hass.data[DOMAIN].pop(entry.entry_id, None)
    return True


class OpenAIChatCoordinator:
    """Coordinator pentru chat OpenAI cu istoric și memorie."""

    def __init__(
        self,
        hass: HomeAssistant,
        client: openai.AsyncOpenAI,
        entry: ConfigEntry,
    ) -> None:
        """Inițializare coordinator."""
        self.hass = hass
        self.client = client
        self.entry = entry
        self._history_store = storage.Store(
            hass, STORAGE_VERSION, f"{DOMAIN}_{entry.entry_id}_{STORAGE_KEY_HISTORY}"
        )
        self._memory_store = storage.Store(
            hass, STORAGE_VERSION, f"{DOMAIN}_{entry.entry_id}_{STORAGE_KEY_MEMORY}"
        )

    @property
    def model(self) -> str:
        """Model OpenAI configurat."""
        if self.entry.options.get(CONF_SMART_MODE):
            model = self.entry.options.get("model", DEFAULT_MODEL_SMART)
        else:
            model = self.entry.options.get("model", DEFAULT_MODEL)
        return str(model).strip()

    @property
    def max_tokens(self) -> int:
        """Max tokens configurat."""
        if self.entry.options.get(CONF_SMART_MODE):
            return self.entry.options.get("max_tokens", DEFAULT_MAX_TOKENS_SMART)
        return self.entry.options.get("max_tokens", DEFAULT_MAX_TOKENS)

    @property
    def temperature(self) -> float:
        """Temperatură configurată."""
        if self.entry.options.get(CONF_SMART_MODE):
            return self.entry.options.get("temperature", DEFAULT_TEMPERATURE_SMART)
        return self.entry.options.get("temperature", DEFAULT_TEMPERATURE)

    async def get_memory(self) -> str:
        """Obține memoria (system prompt) curentă."""
        data = await self._memory_store.async_load()
        if data and "memory" in data:
            return data["memory"]
        return DEFAULT_MEMORY

    async def set_memory(self, memory: str) -> None:
        """Setează memoria (system prompt)."""
        await self._memory_store.async_save({"memory": memory})

    async def get_history(self, conversation_id: str | None = None) -> list[dict]:
        """Obține istoricul conversației."""
        data = await self._history_store.async_load()
        if not data:
            return []

        conversations = data.get("conversations", {})
        cid = conversation_id or "default"
        return conversations.get(cid, [])

    async def add_to_history(
        self,
        role: str,
        content: str,
        conversation_id: str | None = None,
    ) -> None:
        """Adaugă mesaj la istoric."""
        data = await self._history_store.async_load() or {"conversations": {}}
        cid = conversation_id or "default"
        if cid not in data["conversations"]:
            data["conversations"][cid] = []

        data["conversations"][cid].append({"role": role, "content": content})
        data["conversations"][cid] = data["conversations"][cid][-50:]
        await self._history_store.async_save(data)

    async def clear_history(self, conversation_id: str | None = None) -> None:
        """Șterge istoricul conversației."""
        data = await self._history_store.async_load() or {"conversations": {}}
        cid = conversation_id or "default"
        data["conversations"][cid] = []
        await self._history_store.async_save(data)

    async def delete_conversation(self, conversation_id: str) -> None:
        """Șterge o conversație completă."""
        data = await self._history_store.async_load() or {"conversations": {}}
        if conversation_id in data["conversations"]:
            del data["conversations"][conversation_id]
            await self._history_store.async_save(data)

    async def get_conversation_list(self) -> list[dict]:
        """Obține lista conversațiilor."""
        data = await self._history_store.async_load()
        if not data:
            return []

        conversations = data.get("conversations", {})
        return [
            {
                "id": cid,
                "title": self._get_conversation_title(messages),
                "message_count": len(messages),
            }
            for cid, messages in conversations.items()
            if messages
        ]

    def _get_conversation_title(self, messages: list[dict]) -> str:
        """Generează titlu pentru conversație din primul mesaj."""
        for msg in messages:
            if msg.get("role") == "user" and msg.get("content"):
                content = msg["content"][:50]
                return content + "..." if len(msg["content"]) > 50 else content
        return "Conversație nouă"

    def _build_context(self) -> str:
        """Construiește contextul curent (dată, oră, timezone)."""
        from homeassistant.util import dt as dt_util
        now = dt_util.now()
        tz = getattr(self.hass.config, "time_zone", "UTC")
        return (
            f"Data și ora curentă: {now.strftime('%d.%m.%Y %H:%M')} "
            f"(timezone: {tz})"
        )

    def _detect_intent(self, message: str) -> str:
        """Detectează intenția userului: command / planner / assistant."""
        text = message.lower().strip()
        # Heuristic simplu și robust pentru comenzi de acțiune.
        command_patterns = [
            r"\b(porneste|pornește|opreste|oprește|aprinde|stinge)\b",
            r"\b(seteaza|setează|ruleaza|rulează|declanseaza|declanșează)\b",
            r"\b(creeaza|crează|adauga|adaugă|sterge|șterge|scrie|modifica|modifică)\b",
            r"\b(activeaza|activează|dezactiveaza|dezactivează|executa|execută)\b",
        ]
        if any(re.search(pattern, text) for pattern in command_patterns):
            return "command"

        planner_patterns = [
            r"\b(cum sa|cum să|plan|strategie|arhitectura|arhitectură)\b",
            r"\b(ce recomanzi|ce varianta|ce variantă|optimizare)\b",
        ]
        if any(re.search(pattern, text) for pattern in planner_patterns):
            return "planner"

        return "assistant"

    def _mode_rules(self, intent: str) -> str:
        """Reguli interne de comportament per intenție."""
        if intent == "command":
            return (
                "Regim COMMAND (strict):\n"
                "- Pentru orice acțiune în HA, folosește tool-uri.\n"
                "- NU spune că ai executat dacă nu există rezultat de tool.\n"
                "- Dacă lipsește entity_id/serviciu, cere clarificare scurtă.\n"
                "- Răspuns final: scurt, factual, bazat pe rezultate before/after."
            )
        if intent == "planner":
            return (
                "Regim PLANNER:\n"
                "- Oferă plan pe pași, opțiuni și trade-offs.\n"
                "- Nu executa acțiuni decât dacă userul cere explicit."
            )
        return (
            "Regim ASSISTANT:\n"
            "- Explică clar și concis.\n"
            "- Execută doar când cererea implică explicit o acțiune."
        )

    def _resolve_model(self) -> tuple[str, str | None]:
        """Returnează model valid și avertisment dacă a fost normalizat."""
        raw_model = (self.model or "").strip()
        default_model = DEFAULT_MODEL_SMART if self.entry.options.get(CONF_SMART_MODE) else DEFAULT_MODEL

        if not raw_model:
            return default_model, "Model gol în opțiuni; folosesc modelul implicit."

        # OpenAI model ids: litere/cifre + -_.:
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]*", raw_model) is None:
            return default_model, (
                f"Model invalid '{raw_model}'. Folosesc '{default_model}'."
            )

        return raw_model, None

    def _resolve_generation_params(self) -> tuple[int, float]:
        """Normalizează max_tokens/temperature în intervale valide."""
        try:
            max_tokens = int(self.max_tokens)
        except (TypeError, ValueError):
            max_tokens = DEFAULT_MAX_TOKENS
        max_tokens = max(64, min(128000, max_tokens))

        try:
            temperature = float(self.temperature)
        except (TypeError, ValueError):
            temperature = DEFAULT_TEMPERATURE
        temperature = max(0.0, min(2.0, temperature))
        return max_tokens, temperature

    async def chat(
        self,
        message: str,
        conversation_id: str | None = None,
    ) -> str:
        """Mod stabil: chat simplu, fără tool orchestration."""
        memory = await self.get_memory()
        history = await self.get_history(conversation_id)
        context = self._build_context()

        messages = [{"role": "system", "content": f"{memory}\n\nContext: {context}"}]
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": message})

        try:
            response = await self.client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=messages,
                max_tokens=1024,
                temperature=0.2,
            )
            reply = (response.choices[0].message.content or "").strip()
            if not reply:
                reply = "Nu am primit conținut de la model."
        except Exception as err:
            _LOGGER.error("Eroare OpenAI API (mod stabil): %s", err)
            reply = f"ERROR: OpenAI request a eșuat în modul stabil. Detaliu: {err}"

        await self.add_to_history("user", message, conversation_id)
        await self.add_to_history("assistant", reply, conversation_id)
        return reply
