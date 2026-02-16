"""Integrare OpenAI Chat cu istoric persistent și memorie editabilă."""

from __future__ import annotations

import json
import logging
import os
import re

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
READ_ONLY_TOOL_NAMES = {
    "get_ha_entities",
    "get_ha_services",
    "read_ha_file",
    "list_ha_files",
    "list_lovelace_dashboards",
    "get_lovelace_views",
}
REFUSAL_HINTS = (
    "nu am acces",
    "nu pot accesa",
    "nu am permisiune",
    "dacă îmi permiți",
    "dacă îmi dai permisiunea",
    "atașezi aici",
)


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
    coordinator = OpenAIChatCoordinator(hass, entry)
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
        entry: ConfigEntry,
    ) -> None:
        """Inițializare coordinator."""
        self.hass = hass
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

    def _read_only_tools(self) -> list[dict]:
        """Returnează doar tools de tip read-only pentru HA."""
        return [
            tool
            for tool in OPENAI_TOOLS
            if tool.get("function", {}).get("name") in READ_ONLY_TOOL_NAMES
        ]

    async def _direct_read_only_fallback(self, message: str) -> str | None:
        """Execută direct un tool read-only pentru cereri clare."""
        text = (message or "").strip()
        lower = text.lower()

        # Citește fișier specific: "citește configuration.yaml"
        file_match = re.search(
            r"(configuration\.yaml|automations\.yaml|scripts\.yaml|secrets\.yaml|.+\.ya?ml)",
            text,
            flags=re.IGNORECASE,
        )
        if ("cite" in lower or "read" in lower) and file_match:
            path = file_match.group(1).strip()
            tool_out = await execute_tool(self.hass, "read_ha_file", {"path": path})
            return f"Rezultat `read_ha_file` pentru `{path}`:\n{tool_out}"

        # Listează fișiere: "listează fișierele din config"
        if "list" in lower and "fi" in lower:
            tool_out = await execute_tool(self.hass, "list_ha_files", {"subdir": ""})
            return f"Rezultat `list_ha_files`:\n{tool_out}"

        # Entități HA (opțional domain)
        if "entit" in lower:
            domain = ""
            for candidate in ("light", "switch", "sensor", "automation", "script"):
                if candidate in lower:
                    domain = candidate
                    break
            tool_out = await execute_tool(
                self.hass,
                "get_ha_entities",
                {"domain": domain} if domain else {},
            )
            return "Rezultat `get_ha_entities`:\n" + tool_out

        # Servicii HA (opțional domain)
        if "servici" in lower:
            domain = ""
            for candidate in ("light", "switch", "automation", "script", "climate"):
                if candidate in lower:
                    domain = candidate
                    break
            tool_out = await execute_tool(
                self.hass,
                "get_ha_services",
                {"domain": domain} if domain else {},
            )
            return "Rezultat `get_ha_services`:\n" + tool_out

        # Lovelace dashboards/views
        if "dashboard" in lower and ("list" in lower or "arată" in lower):
            tool_out = await execute_tool(self.hass, "list_lovelace_dashboards", {})
            return "Rezultat `list_lovelace_dashboards`:\n" + tool_out

        if "view" in lower or "tab" in lower:
            dashboard = "lovelace"
            dashboard_match = re.search(r"dashboard\s+([a-z0-9_-]+)", lower)
            if dashboard_match:
                dashboard = dashboard_match.group(1)
            tool_out = await execute_tool(
                self.hass,
                "get_lovelace_views",
                {"dashboard": dashboard},
            )
            return "Rezultat `get_lovelace_views`:\n" + tool_out

        return None

    async def chat(
        self,
        message: str,
        conversation_id: str | None = None,
    ) -> str:
        """Mod stabil: chat simplu, fără tool orchestration."""
        memory = await self.get_memory()
        history = await self.get_history(conversation_id)
        context = self._build_context()
        model, model_warning = self._resolve_model()
        max_tokens, temperature = self._resolve_generation_params()

        messages = [{"role": "system", "content": f"{memory}\n\nContext: {context}"}]
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": message})

        try:
            http_client = get_async_client(self.hass)
            reply = ""
            read_only_tools = self._read_only_tools()
            for _ in range(4):
                payload = {
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "tools": read_only_tools,
                    "tool_choice": "auto",
                }
                response = await http_client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.entry.data[CONF_API_KEY]}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=30,
                )
                if response.status_code >= 400:
                    try:
                        err_payload = response.json()
                        err_text = err_payload.get("error", {}).get("message", response.text)
                    except Exception:
                        err_text = response.text
                    raise RuntimeError(f"OpenAI HTTP {response.status_code}: {err_text}")

                data = response.json()
                assistant_msg = data.get("choices", [{}])[0].get("message", {})
                content = (assistant_msg.get("content") or "").strip()
                tool_calls = assistant_msg.get("tool_calls") or []

                if tool_calls:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": assistant_msg.get("content") or "",
                            "tool_calls": tool_calls,
                        }
                    )
                    for tool_call in tool_calls:
                        tool_call_id = tool_call.get("id")
                        fn_data = tool_call.get("function", {})
                        tool_name = fn_data.get("name", "")
                        raw_args = fn_data.get("arguments") or "{}"
                        try:
                            tool_args = json.loads(raw_args)
                            if not isinstance(tool_args, dict):
                                tool_args = {}
                        except Exception:
                            tool_args = {}

                        if tool_name not in READ_ONLY_TOOL_NAMES:
                            tool_result = json.dumps(
                                {"error": f"Tool blocat în mod read-only: {tool_name}"},
                                ensure_ascii=False,
                            )
                        else:
                            tool_result = await execute_tool(self.hass, tool_name, tool_args)

                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "content": tool_result,
                            }
                        )
                    continue

                reply = content
                break

            if model_warning:
                reply = f"[Avertisment configurare] {model_warning}\n\n{reply}"
            if reply and any(hint in reply.lower() for hint in REFUSAL_HINTS):
                direct_reply = await self._direct_read_only_fallback(message)
                if direct_reply:
                    reply = direct_reply
            if not reply:
                reply = "Nu am primit conținut de la model."
        except Exception as err:
            _LOGGER.error("Eroare OpenAI API (mod stabil): %s", err)
            reply = f"ERROR: OpenAI request a eșuat în modul stabil. Detaliu: {err}"

        await self.add_to_history("user", message, conversation_id)
        await self.add_to_history("assistant", reply, conversation_id)
        return reply
