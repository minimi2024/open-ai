"""Integrare OpenAI Chat cu istoric persistent și memorie editabilă."""

from __future__ import annotations

import json
import logging
import os

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
            return self.entry.options.get("model", DEFAULT_MODEL_SMART)
        return self.entry.options.get("model", DEFAULT_MODEL)

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

    async def chat(
        self,
        message: str,
        conversation_id: str | None = None,
    ) -> str:
        """Trimite mesaj la OpenAI cu tools și returnează răspunsul."""
        memory = await self.get_memory()
        history = await self.get_history(conversation_id)
        context = self._build_context()

        system_content = f"{memory}\n\n**Context:** {context}"
        messages = [{"role": "system", "content": system_content}]
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": message})

        max_iterations = 10
        tool_errors: list[str] = []
        tool_reports: list[dict] = []
        for _ in range(max_iterations):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=OPENAI_TOOLS,
                    tool_choice="auto",
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
            except Exception as err:
                _LOGGER.error("Eroare OpenAI API: %s", err)
                raise

            choice = response.choices[0]
            message_obj = choice.message

            if message_obj.tool_calls and len(message_obj.tool_calls) > 0:
                assistant_msg = {
                    "role": "assistant",
                    "content": message_obj.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments or "{}",
                            },
                        }
                        for tc in message_obj.tool_calls
                    ],
                }
                messages.append(assistant_msg)
                for tc in message_obj.tool_calls:
                    args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                    result = await execute_tool(
                        self.hass,
                        tc.function.name,
                        args,
                    )
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })
                    # Capturăm erorile tool-urilor ca să nu raportăm fals succesul.
                    try:
                        parsed = json.loads(result)
                        if isinstance(parsed, dict):
                            tool_reports.append(
                                {
                                    "tool": tc.function.name,
                                    "result": parsed,
                                }
                            )
                        if isinstance(parsed, dict) and parsed.get("error"):
                            tool_errors.append(str(parsed.get("error")))
                    except Exception:
                        pass
                continue

            reply = message_obj.content or ""
            break
        else:
            reply = "Am depășit numărul maxim de apeluri. Încearcă din nou."

        if tool_errors:
            unique_errors: list[str] = []
            for err in tool_errors:
                if err not in unique_errors:
                    unique_errors.append(err)
            reply = (
                f"{reply}\n\n"
                "Atenție: unele acțiuni au eșuat:\n"
                + "\n".join(f"- {err}" for err in unique_errors[:5])
            )

        # Răspuns determinist de verificare execuție pentru acțiuni HA.
        service_reports = [
            rep for rep in tool_reports if rep.get("tool") == "call_ha_service"
        ]
        if service_reports:
            changed_total = 0
            unchanged_total = 0
            verify_lines: list[str] = []
            for rep in service_reports:
                res = rep.get("result", {})
                service_name = str(res.get("service", "serviciu"))
                changed = res.get("changed_entities") or []
                unchanged = res.get("unchanged_entities") or []
                changed_total += len(changed)
                unchanged_total += len(unchanged)
                if changed:
                    verify_lines.append(
                        f"- {service_name}: schimbate {', '.join(changed[:5])}"
                    )
                elif unchanged:
                    verify_lines.append(
                        f"- {service_name}: fara schimbare pe {', '.join(unchanged[:5])}"
                    )
                elif res.get("error"):
                    verify_lines.append(f"- {service_name}: eroare {res.get('error')}")

            if changed_total == 0:
                prefix = "Nu pot confirma executia comenzii (nu s-a detectat nicio schimbare de stare)."
            else:
                prefix = f"Executie verificata: {changed_total} entitati si-au schimbat starea."

            verify_block = (
                f"\n\n{prefix}\n"
                + "\n".join(verify_lines[:8])
            )
            reply = f"{reply}{verify_block}"

        await self.add_to_history("user", message, conversation_id)
        await self.add_to_history("assistant", reply, conversation_id)

        return reply
