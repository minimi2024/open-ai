"""Integrare OpenAI Chat cu istoric persistent și memorie editabilă."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import unicodedata

from homeassistant.components.http import StaticPathConfig
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_API_KEY
from homeassistant.core import HomeAssistant
from homeassistant.helpers import storage
from homeassistant.helpers.httpx_client import get_async_client

from .const import (
    CONF_MODEL,
    CONF_SMART_MODE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MAX_TOKENS_SMART,
    DEFAULT_MEMORY,
    DEFAULT_MODEL,
    DEFAULT_MODEL_SMART,
    DEFAULT_TEMPERATURE,
    DEFAULT_TEMPERATURE_SMART,
    DOMAIN,
    MODEL_CHOICES,
    STORAGE_KEY_HISTORY,
    STORAGE_KEY_MEMORY,
    STORAGE_VERSION,
)
from .http import (
    OpenAIChatView,
    OpenAIConversationsView,
    OpenAIHistoryView,
    OpenAIMemoryView,
    OpenAISettingsView,
)
from .tools import OPENAI_TOOLS, execute_tool

_LOGGER = logging.getLogger(__name__)
READ_ONLY_TOOL_NAMES = {
    "get_ha_entities",
    "get_ha_services",
    "read_ha_file",
    "list_ha_files",
    "web_search",
    "fetch_url",
    "list_lovelace_dashboards",
    "get_lovelace_views",
}
WRITE_TOOL_NAMES = {
    "call_ha_service",
    "write_ha_file",
    "add_lovelace_view",
    "create_lovelace_dashboard",
    "delete_lovelace_view",
    "dedupe_lovelace_views",
}
REFUSAL_HINTS = (
    "nu am acces",
    "nu pot accesa",
    "nu am permisiune",
    "nu am acces direct",
    "nu pot scrie direct",
    "nu pot face click",
    "nu exista niciun buton",
    "dacă îmi permiți",
    "dacă îmi dai permisiunea",
    "atașezi aici",
)
STORAGE_KEY_PENDING = "openai_chat_pending_actions"


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
    hass.http.register_view(OpenAISettingsView())

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
        self._pending_store = storage.Store(
            hass, STORAGE_VERSION, f"{DOMAIN}_{entry.entry_id}_{STORAGE_KEY_PENDING}"
        )

    @property
    def model(self) -> str:
        """Model OpenAI configurat."""
        if self.entry.options.get(CONF_SMART_MODE):
            model = self.entry.options.get(CONF_MODEL, DEFAULT_MODEL_SMART)
        else:
            model = self.entry.options.get(CONF_MODEL, DEFAULT_MODEL)
        return str(model).strip()

    def get_chat_settings(self) -> dict:
        """Returnează setările de chat expuse în panel."""
        resolved_model, _ = self._resolve_model()
        return {
            "current_model": resolved_model,
            "available_models": list(MODEL_CHOICES),
            "smart_mode": bool(self.entry.options.get(CONF_SMART_MODE, False)),
        }

    def set_chat_model(self, model: str) -> tuple[bool, str]:
        """Actualizează modelul folosit de chat."""
        selected = str(model or "").strip()
        if selected not in MODEL_CHOICES:
            return False, f"Model invalid: {selected}"
        options = dict(self.entry.options)
        options[CONF_MODEL] = selected
        self.hass.config_entries.async_update_entry(self.entry, options=options)
        return True, selected

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

    async def get_pending_action(self, conversation_id: str | None = None) -> str | None:
        """Obține comanda pending pentru conversație."""
        data = await self._pending_store.async_load()
        if not data:
            return None
        pending = data.get("pending", {})
        cid = conversation_id or "default"
        value = pending.get(cid)
        return value if isinstance(value, str) and value.strip() else None

    async def set_pending_action(
        self,
        action_message: str,
        conversation_id: str | None = None,
    ) -> None:
        """Salvează comanda pending pentru conversație."""
        data = await self._pending_store.async_load() or {"pending": {}}
        cid = conversation_id or "default"
        data.setdefault("pending", {})
        data["pending"][cid] = action_message
        await self._pending_store.async_save(data)

    async def clear_pending_action(self, conversation_id: str | None = None) -> None:
        """Șterge comanda pending pentru conversație."""
        data = await self._pending_store.async_load() or {"pending": {}}
        cid = conversation_id or "default"
        if cid in data.get("pending", {}):
            del data["pending"][cid]
            await self._pending_store.async_save(data)

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

    def _build_openai_payload(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
        tools: list[dict] | None = None,
    ) -> dict:
        """Construiește payload compatibil cu modelul selectat."""
        payload: dict = {"model": model, "messages": messages}
        # GPT-5.x folosește max_completion_tokens, nu max_tokens.
        if model.startswith("gpt-5"):
            payload["max_completion_tokens"] = max_tokens
        else:
            payload["max_tokens"] = max_tokens
            payload["temperature"] = temperature

        if tools is not None:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        return payload

    def _select_history_for_request(
        self,
        history: list[dict],
        is_action_request: bool,
    ) -> list[dict]:
        """Scurtează contextul pentru a reduce consumul de tokeni."""
        if not history:
            return []
        # Comenzile operaționale nu au nevoie de context mare.
        max_items = 12 if is_action_request else 36
        return history[-max_items:]

    def _resolve_runtime_model(self, selected_model: str, is_action_request: bool) -> str:
        """Alege modelul efectiv folosit la runtime."""
        # Pentru comenzi HA preferăm model mai eficient (cost/TPM mai mic).
        if is_action_request and selected_model.startswith("gpt-5"):
            return DEFAULT_MODEL
        return selected_model

    def _reduce_payload_tokens_for_retry(self, payload: dict) -> bool:
        """Reduce token cap-ul pentru retry la rate-limit."""
        for key in ("max_completion_tokens", "max_tokens"):
            if key not in payload:
                continue
            try:
                current = int(payload[key])
            except (TypeError, ValueError):
                continue
            # Coborâm agresiv la 60% pentru a reduce TPM-ul cerut.
            reduced = max(256, int(current * 0.6))
            if reduced >= current:
                return False
            payload[key] = reduced
            return True
        return False

    def _extract_rate_limit_wait_seconds(
        self,
        err_text: str,
        headers: dict | None = None,
    ) -> float:
        """Extrage timpul de așteptare recomandat pentru 429."""
        if headers:
            retry_after = headers.get("Retry-After") or headers.get("retry-after")
            if retry_after:
                try:
                    return max(0.0, float(retry_after))
                except ValueError:
                    pass
        match = re.search(r"try again in\s*([0-9]+(?:\.[0-9]+)?)s", err_text, flags=re.IGNORECASE)
        if match:
            try:
                return max(0.0, float(match.group(1)))
            except ValueError:
                pass
        return 2.0

    async def _post_openai_with_rate_limit_retry(
        self,
        http_client,
        payload: dict,
    ):
        """Trimite request OpenAI, cu retry pe 429 și param fallback pe 400."""
        max_attempts = 3
        response = None
        attempt_payload = dict(payload)
        for attempt in range(1, max_attempts + 1):
            response = await http_client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.entry.data[CONF_API_KEY]}",
                    "Content-Type": "application/json",
                },
                json=attempt_payload,
                timeout=30,
            )
            if response.status_code == 400:
                try:
                    err_payload = response.json()
                    err_text = err_payload.get("error", {}).get("message", response.text)
                except Exception:
                    err_text = response.text

                # Fallback compatibilitate: unele modele (ex: GPT-5.x) cer
                # `max_completion_tokens` în loc de `max_tokens`.
                if (
                    "unsupported parameter" in str(err_text).lower()
                    and "'max_tokens'" in str(err_text).lower()
                    and "max_tokens" in attempt_payload
                ):
                    converted_payload = dict(attempt_payload)
                    converted_payload["max_completion_tokens"] = converted_payload.pop(
                        "max_tokens"
                    )
                    response = await http_client.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.entry.data[CONF_API_KEY]}",
                            "Content-Type": "application/json",
                        },
                        json=converted_payload,
                        timeout=30,
                    )
                    return response

            if response.status_code != 429 or attempt == max_attempts:
                return response

            try:
                err_payload = response.json()
                err_text = err_payload.get("error", {}).get("message", response.text)
            except Exception:
                err_text = response.text
            wait_seconds = self._extract_rate_limit_wait_seconds(
                err_text,
                dict(response.headers) if getattr(response, "headers", None) else None,
            )
            self._reduce_payload_tokens_for_retry(attempt_payload)
            # Mic buffer peste recomandarea API ca să evităm re-rate-limit.
            await asyncio.sleep(min(15.0, wait_seconds + 0.6))
        return response

    def _read_only_tools(self) -> list[dict]:
        """Returnează doar tools de tip read-only pentru HA."""
        return [
            tool
            for tool in OPENAI_TOOLS
            if tool.get("function", {}).get("name") in READ_ONLY_TOOL_NAMES
        ]

    def _allowed_tools(self, allow_write: bool) -> list[dict]:
        """Returnează tool-urile permise pentru mesajul curent."""
        if allow_write:
            return OPENAI_TOOLS
        return self._read_only_tools()

    def _is_action_request(self, message: str) -> bool:
        """Detectează cereri de acțiune (write/control)."""
        text = (message or "").lower()
        patterns = (
            r"\b(porneste|pornește|opreste|oprește|aprinde|stinge)\b",
            r"\b(seteaza|setează|ruleaza|rulează|declanseaza|declanșează)\b",
            r"\b(pune|schimba|schimbă|regleaza|reglează)\b",
            r"\b(creeaza|creează|adauga|adaugă|adage|sterge|șterge|scrie|modifica|modifică)\b",
            r"\b(activeaza|activează|dezactiveaza|dezactivează|executa|execută)\b",
            r"\b(delete|remove|turn on|turn off|write)\b",
            r"\b(temperatura|temperatură|termostat|climate)\b.{0,40}\b(la|pe)\s*\d+([.,]\d+)?\b",
            r"\b(dashboard|tab|view|lovelace|card)\b",
        )
        return any(re.search(pattern, text) for pattern in patterns)

    def _has_write_confirmation(self, message: str) -> bool:
        """Verifică dacă userul a confirmat explicit acțiuni write/control."""
        text = (message or "").lower()
        confirmation_markers = (
            "confirm",
            "confirmat",
            "da, executa",
            "da executa",
            "executa acum",
            "ruleaza acum",
            "poți executa",
            "poti executa",
        )
        return any(marker in text for marker in confirmation_markers)

    def _is_confirmation_only_message(self, message: str) -> bool:
        """Detectează răspunsuri scurte de confirmare (ex: da/ok/confirm)."""
        text = (message or "").strip().lower()
        if not text:
            return False
        return text in {
            "da",
            "ok",
            "confirm",
            "confirmat",
            "executa",
            "execută",
            "continua",
            "continuă",
            "yes",
        }

    def _is_temperature_set_intent(self, message: str) -> bool:
        """Detectează comenzi de setare temperatură pentru climate."""
        text = (message or "").lower()
        has_temperature_target = re.search(r"\b\d+([.,]\d+)?\b", text) is not None
        has_temp_keyword = any(
            kw in text for kw in ("temperatura", "temperatură", "termostat", "climate")
        )
        has_action_keyword = any(
            kw in text for kw in ("pune", "seteaza", "setează", "schimba", "schimbă", "regleaza", "reglează")
        )
        return has_temperature_target and has_temp_keyword and has_action_keyword

    def _extract_target_temperature(self, message: str) -> float | None:
        """Extrage temperatura cerută din mesaj."""
        text = (message or "").lower().replace(",", ".")
        # Evităm orele/alte numere, preferăm pattern "la/pe X grade".
        specific_match = re.search(r"\b(?:la|pe)\s*(\d{1,2}(?:\.\d)?)\s*(?:grade|°c|c)?\b", text)
        if specific_match:
            try:
                return float(specific_match.group(1))
            except ValueError:
                return None
        any_number = re.search(r"\b(\d{1,2}(?:\.\d)?)\b", text)
        if any_number:
            try:
                return float(any_number.group(1))
            except ValueError:
                return None
        return None

    def _resolve_target_climate_entity(self, message: str) -> str | None:
        """Alege cel mai probabil entity_id de tip climate pe baza mesajului."""
        climates = list(self.hass.states.async_all("climate"))
        if not climates:
            return None
        if len(climates) == 1:
            return climates[0].entity_id

        text = (message or "").lower()
        tokens = [t for t in re.findall(r"[a-z0-9_]+", text) if len(t) >= 3]

        best_entity_id = None
        best_score = -1
        for state in climates:
            entity_id = state.entity_id.lower()
            friendly = str(state.attributes.get("friendly_name", "")).lower()
            haystack = f"{entity_id} {friendly}"
            score = sum(1 for token in tokens if token in haystack)
            if score > best_score:
                best_score = score
                best_entity_id = state.entity_id

        if best_entity_id and best_score > 0:
            return best_entity_id

        # Fallback practic pentru cazuri comune.
        for state in climates:
            haystack = f"{state.entity_id.lower()} {str(state.attributes.get('friendly_name', '')).lower()}"
            if "living" in text and "living" in haystack:
                return state.entity_id

        return None

    def _wants_hvac_explanation(self, message: str) -> bool:
        """Detectează dacă userul cere și explicații despre moduri HVAC."""
        text = (message or "").lower()
        return any(
            marker in text
            for marker in (
                "explica",
                "explică",
                "ce inseamna",
                "ce înseamnă",
                "heat_cool",
                "ce variante",
                "ce moduri",
            )
        )

    def _build_hvac_explanation(self, climate_entity_id: str) -> str:
        """Construiește explicație scurtă pentru modurile HVAC disponibile."""
        state = self.hass.states.get(climate_entity_id)
        if state is None:
            return ""

        hvac_mode = str(state.state)
        hvac_modes = state.attributes.get("hvac_modes") or []
        variants = ", ".join(str(m) for m in hvac_modes) if hvac_modes else "necunoscute"
        heat_cool_note = (
            "- `heat_cool`: menține temperatura între limite și comută automat între încălzire/răcire.\n"
            if "heat_cool" in hvac_modes or hvac_mode == "heat_cool"
            else ""
        )
        return (
            "\n\nModuri HVAC:\n"
            f"- Mod curent: `{hvac_mode}`\n"
            f"- Variante disponibile: {variants}\n"
            f"{heat_cool_note}"
            "- `heat`: doar încălzire\n"
            "- `cool`: doar răcire\n"
            "- `off`: oprit"
        )

    def _normalize(self, text: str) -> str:
        """Normalizează textul pentru matching tolerant la diacritice."""
        normalized = unicodedata.normalize("NFKD", text or "")
        return "".join(ch for ch in normalized if not unicodedata.combining(ch)).lower()

    def _slugify_local(self, text: str) -> str:
        """Slug simplu pentru path-uri Lovelace."""
        normalized = self._normalize(text)
        normalized = re.sub(r"[^a-z0-9]+", "-", normalized).strip("-")
        return normalized or "view-1"

    def _is_thermostat_tab_request(self, message: str) -> bool:
        """Detectează cereri de adăugare/actualizare tab cu carduri thermostat."""
        text = self._normalize(message or "")
        has_tab_context = any(k in text for k in ("tab", "view", "dashboard", "lovelace"))
        has_thermostat_context = any(k in text for k in ("thermostat", "temperatura", "temperaturi", "climate"))
        has_action = any(k in text for k in ("adauga", "adage", "creeaza", "pune", "actualizeaza", "update"))
        return has_tab_context and has_thermostat_context and has_action

    def _has_recent_thermostat_context(self, history: list[dict]) -> bool:
        """Detectează context recent de tab thermostat (YAML/Lovelace/climate)."""
        for msg in reversed(history[-10:]):
            content = msg.get("content")
            if not isinstance(content, str):
                continue
            norm = self._normalize(content)
            if (
                "type: thermostat" in norm
                or "temperaturi casa" in norm
                or ("dashboard" in norm and "control central" in norm)
                or ("climate." in norm and "yaml" in norm)
            ):
                return True
        return False

    def _is_short_tab_action(self, message: str) -> bool:
        """Detectează cereri scurte de tip 'adaugă tu în tab'."""
        norm = self._normalize(message or "")
        has_action = any(k in norm for k in ("adauga", "adage", "pune", "fa", "fă", "executa"))
        has_tab = any(k in norm for k in ("tab", "view"))
        return has_action and has_tab

    def _is_retry_like_message(self, message: str) -> bool:
        """Detectează mesaje scurte de retry care se bazează pe context."""
        norm = self._normalize(message or "")
        return any(
            marker in norm
            for marker in (
                "reincearca",
                "incearca din nou",
                "incearca iar",
                "mai incearca",
                "fa din nou",
                "din nou",
            )
        )

    def _is_generic_capability_refusal(self, text: str) -> bool:
        """Detectează răspunsuri generice de tip 'nu pot scrie în HA/UI'."""
        norm = self._normalize(text or "")
        must_have = any(
            phrase in norm
            for phrase in (
                "nu am acces direct",
                "nu pot scrie direct",
                "nu pot face click",
                "nu exista niciun buton",
                "trebuie sa pui tu in ui",
            )
        )
        ha_context = any(
            phrase in norm
            for phrase in ("lovelace", "dashboard", "yaml", "home assistant")
        )
        return must_have and ha_context

    def _extract_dashboard_and_tab(
        self, message: str, history: list[dict]
    ) -> tuple[str, str]:
        """Extrage dashboard și tab din mesaj sau din istoric recent."""
        candidates: list[str] = [message]
        for msg in reversed(history[-12:]):
            content = msg.get("content")
            if isinstance(content, str):
                candidates.append(content)
        blob = "\n".join(candidates)

        dashboard = None
        tab_title = None

        dashboard_match = re.search(
            r"(?:dashboard(?:ul)?)[\s\"'`]+([^\"'`\n]+)",
            blob,
            flags=re.IGNORECASE,
        )
        if dashboard_match:
            dashboard = dashboard_match.group(1).strip()

        tab_match = re.search(
            r"(?:tab(?:ul)?|view(?:ul)?)[\s\"'`]+([^\"'`\n]+)",
            blob,
            flags=re.IGNORECASE,
        )
        if tab_match:
            tab_title = tab_match.group(1).strip()

        if not tab_title:
            tab_title = "Temperaturi casa"
        if not dashboard:
            dashboard = "lovelace"
        return dashboard, tab_title

    async def _resolve_lovelace_config_obj(self, dashboard_hint: str):
        """Rezolvă obiectul de config Lovelace după hint (url_path sau titlu)."""
        lovelace_data = self.hass.data.get("lovelace")
        if not lovelace_data:
            return None, "Lovelace nu este disponibil."

        hint = (dashboard_hint or "lovelace").strip()
        hint_slug = self._slugify_local(hint)
        candidates = [hint, hint.lower(), hint_slug, "lovelace", None]

        for cand in candidates:
            if cand in lovelace_data.dashboards:
                return lovelace_data.dashboards.get(cand), None

        hint_norm = self._normalize(hint)
        for key, cfg_obj in lovelace_data.dashboards.items():
            key_norm = self._normalize(str(key or "lovelace"))
            if hint_norm and hint_norm in key_norm:
                return cfg_obj, None

            try:
                cfg = await cfg_obj.async_load(False)
            except Exception:
                continue
            title_norm = self._normalize(str(cfg.get("title", "")))
            if hint_norm and hint_norm in title_norm:
                return cfg_obj, None

        return None, f"Dashboard negăsit pentru '{dashboard_hint}'."

    async def _upsert_thermostat_tab(
        self,
        message: str,
        history: list[dict],
    ) -> str:
        """Creează/actualizează un tab Lovelace cu carduri thermostat."""
        climates = list(self.hass.states.async_all("climate"))
        if not climates:
            return "ERROR: Nu există entități `climate` disponibile."

        dashboard_hint, tab_title = self._extract_dashboard_and_tab(message, history)
        config_obj, err = await self._resolve_lovelace_config_obj(dashboard_hint)
        if err:
            return f"ERROR: {err}"

        try:
            config = await config_obj.async_load(False)
        except Exception as load_err:
            return f"ERROR: Nu pot încărca dashboard-ul Lovelace: {load_err}"

        views = config.get("views", [])
        if not isinstance(views, list):
            return "ERROR: Config Lovelace invalid (`views` nu este listă)."

        cards = []
        for state in sorted(
            climates,
            key=lambda s: str(s.attributes.get("friendly_name", s.entity_id)).lower(),
        ):
            cards.append(
                {
                    "type": "thermostat",
                    "entity": state.entity_id,
                    "name": str(state.attributes.get("friendly_name", state.entity_id)),
                }
            )

        tab_path = self._slugify_local(tab_title)
        target_index = None
        tab_title_norm = self._normalize(tab_title)
        for idx, view in enumerate(views):
            title_norm = self._normalize(str(view.get("title", "")))
            path_norm = self._normalize(str(view.get("path", "")))
            if title_norm == tab_title_norm or path_norm == tab_path:
                target_index = idx
                break

        if target_index is None:
            views.append(
                {
                    "title": tab_title,
                    "path": tab_path,
                    "icon": "mdi:thermometer",
                    "cards": cards,
                }
            )
            action = "creat"
        else:
            views[target_index]["title"] = tab_title
            views[target_index]["path"] = tab_path
            views[target_index]["icon"] = views[target_index].get("icon", "mdi:thermometer")
            views[target_index]["cards"] = cards
            action = "actualizat"

        config["views"] = views
        try:
            await config_obj.async_save(config)
        except Exception as save_err:
            return f"ERROR: Nu pot salva dashboard-ul Lovelace: {save_err}"

        return (
            f"Tab-ul `{tab_title}` a fost {action} cu succes în dashboard `{dashboard_hint}`. "
            f"Au fost adăugate {len(cards)} carduri thermostat."
        )

    async def _execute_temperature_set(
        self,
        message: str,
    ) -> str:
        """Execută deterministic climate.set_temperature pentru comenzi de termostat."""
        target_temp = self._extract_target_temperature(message)
        if target_temp is None:
            return "ERROR: Nu am putut determina temperatura țintă din mesaj."
        if target_temp < 5 or target_temp > 35:
            return "ERROR: Temperatura cerută pare invalidă (acceptat 5-35°C)."

        climate_entity_id = self._resolve_target_climate_entity(message)
        if not climate_entity_id:
            return (
                "ERROR: Nu am putut determina termostatul țintă. "
                "Specifică entity_id (ex: climate.living)."
            )

        tool_result = await execute_tool(
            self.hass,
            "call_ha_service",
            {
                "domain": "climate",
                "service": "set_temperature",
                "entity_id": climate_entity_id,
                "data": {"temperature": target_temp},
            },
        )
        try:
            parsed = json.loads(tool_result)
        except Exception:
            parsed = {}

        if isinstance(parsed, dict) and parsed.get("error"):
            return f"ERROR: Setarea temperaturii a eșuat: {parsed.get('error')}"

        state = self.hass.states.get(climate_entity_id)
        current_temp = (
            state.attributes.get("temperature")
            if state is not None and "temperature" in state.attributes
            else None
        )
        response = (
            f"Temperatura a fost trimisă către `{climate_entity_id}` la {target_temp:.1f}°C."
        )
        if current_temp is not None:
            response += f" Setarea curentă raportată: {current_temp}°C."
        if self._wants_hvac_explanation(message):
            response += self._build_hvac_explanation(climate_entity_id)
        return response

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
        pending_action = await self.get_pending_action(conversation_id)
        confirmation_only = self._is_confirmation_only_message(message)
        effective_message = message
        is_action_request = self._is_action_request(message)
        allow_write = self._has_write_confirmation(message)
        runtime_model = self._resolve_runtime_model(model, is_action_request)

        # Flux 2-pași: user dă "da", executăm comanda pending salvată anterior.
        if confirmation_only and pending_action:
            effective_message = pending_action
            is_action_request = True
            allow_write = True
            await self.clear_pending_action(conversation_id)

        if is_action_request and not allow_write:
            await self.set_pending_action(message, conversation_id)
            reply = (
                "Confirmă explicit execuția pentru acțiuni write/control. "
                "Scrie simplu 'da' pentru a continua. "
                "Până atunci pot doar citi date (read-only)."
            )
            await self.add_to_history("user", message, conversation_id)
            await self.add_to_history("assistant", reply, conversation_id)
            return reply

        # Flux deterministic pentru taburi de temperaturi (Lovelace).
        thermostat_tab_by_context = (
            self._is_short_tab_action(effective_message)
            and self._has_recent_thermostat_context(history)
        )
        if (self._is_thermostat_tab_request(effective_message) or thermostat_tab_by_context) and allow_write:
            reply = await self._upsert_thermostat_tab(effective_message, history)
            await self.add_to_history("user", message, conversation_id)
            await self.add_to_history("assistant", reply, conversation_id)
            return reply

        # Flux deterministic pentru termostat: nu depinde de tool-calling-ul modelului.
        if self._is_temperature_set_intent(effective_message) and allow_write:
            reply = await self._execute_temperature_set(effective_message)
            await self.add_to_history("user", message, conversation_id)
            await self.add_to_history("assistant", reply, conversation_id)
            return reply

        selected_history = self._select_history_for_request(history, is_action_request)
        messages = [{"role": "system", "content": f"{memory}\n\nContext: {context}"}]
        for msg in selected_history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": effective_message})

        try:
            http_client = get_async_client(self.hass)
            reply = ""
            pattern_fallback_used = False
            allowed_tool_names = {
                tool.get("function", {}).get("name", "")
                for tool in self._allowed_tools(allow_write)
            }
            tools_for_request = self._allowed_tools(allow_write)
            executed_tools = 0
            successful_tools = 0
            failed_tools: list[str] = []
            last_tool_outputs: list[str] = []
            for _ in range(4):
                payload = self._build_openai_payload(
                    model=runtime_model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    tools=tools_for_request,
                )
                response = await self._post_openai_with_rate_limit_retry(http_client, payload)
                if response.status_code >= 400:
                    try:
                        err_payload = response.json()
                        err_text = err_payload.get("error", {}).get("message", response.text)
                    except Exception:
                        err_text = response.text
                    # Fallback hard: uneori tool-calling poate produce erori de tip
                    # "The string did not match the expected pattern.".
                    if "expected pattern" in str(err_text).lower():
                        fallback_payload = self._build_openai_payload(
                            model=runtime_model,
                            messages=messages,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            tools=None,
                        )
                        fallback_resp = await self._post_openai_with_rate_limit_retry(
                            http_client, fallback_payload
                        )
                        if fallback_resp.status_code < 400:
                            fallback_data = fallback_resp.json()
                            reply = (
                                fallback_data.get("choices", [{}])[0]
                                .get("message", {})
                                .get("content", "")
                                .strip()
                            )
                            pattern_fallback_used = True
                            break
                        try:
                            fb_payload = fallback_resp.json()
                            fb_text = fb_payload.get("error", {}).get("message", fallback_resp.text)
                        except Exception:
                            fb_text = fallback_resp.text
                        raise RuntimeError(
                            "OpenAI HTTP "
                            f"{response.status_code}: {err_text}; "
                            f"fallback fără tools a eșuat: HTTP {fallback_resp.status_code}: {fb_text}"
                        )
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
                        executed_tools += 1
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

                        if tool_name not in allowed_tool_names:
                            tool_result = json.dumps(
                                {"error": f"Tool blocat în modul curent: {tool_name}"},
                                ensure_ascii=False,
                            )
                            failed_tools.append(f"{tool_name}: blocat")
                        else:
                            tool_result = await execute_tool(self.hass, tool_name, tool_args)
                            try:
                                parsed = json.loads(tool_result)
                            except Exception:
                                parsed = {}
                            if isinstance(parsed, dict) and parsed.get("error"):
                                failed_tools.append(f"{tool_name}: {parsed.get('error')}")
                            else:
                                successful_tools += 1
                            last_tool_outputs.append(f"{tool_name}: {tool_result}")

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
            if pattern_fallback_used:
                reply = (
                    "[Fallback anti-pattern activat] Am continuat fără tool-calling pentru acest mesaj.\n\n"
                    + reply
                )
            # Guard rail: blocăm refuzurile generice despre lipsa accesului dacă avem
            # intent de acțiune sau context clar de execuție.
            if reply and self._is_generic_capability_refusal(reply):
                thermostat_tab_by_context = (
                    self._is_short_tab_action(effective_message)
                    and self._has_recent_thermostat_context(history)
                )
                if allow_write and (
                    self._is_thermostat_tab_request(effective_message)
                    or thermostat_tab_by_context
                    or self._has_recent_thermostat_context(history)
                ):
                    reply = await self._upsert_thermostat_tab(effective_message, history)
                elif is_action_request and not allow_write:
                    await self.set_pending_action(message, conversation_id)
                    reply = (
                        "Pot executa direct în Home Assistant. Confirmă cu 'da' și continui imediat."
                    )
                elif self._has_recent_thermostat_context(history):
                    # Caz frecvent: user zice doar "reîncearcă", modelul răspunde generic.
                    await self.set_pending_action(effective_message, conversation_id)
                    if allow_write or self._is_retry_like_message(effective_message):
                        retry_reply = await self._upsert_thermostat_tab(
                            effective_message, history
                        )
                        if not retry_reply.startswith("ERROR:"):
                            reply = retry_reply
                        else:
                            reply = (
                                "Pot executa direct modificarea Lovelace. "
                                "Confirmă cu 'da' și continui imediat."
                            )
                    else:
                        reply = (
                            "Pot executa direct modificarea Lovelace. "
                            "Confirmă cu 'da' și continui imediat."
                        )
            if reply and any(hint in reply.lower() for hint in REFUSAL_HINTS):
                direct_reply = await self._direct_read_only_fallback(message)
                if direct_reply:
                    reply = direct_reply
                elif is_action_request:
                    # Evită răspunsuri halucinante despre "nu am permisiune".
                    await self.set_pending_action(message, conversation_id)
                    reply = (
                        "Pot executa comanda în Home Assistant, dar pentru write/control am nevoie de confirmare. "
                        "Scrie 'da' ca să continui."
                    )
            if is_action_request:
                if executed_tools == 0:
                    reply = (
                        "ERROR: Nu am executat nicio acțiune în Home Assistant. "
                        "Reformulează comanda mai specific (entity_id, serviciu, date)."
                    )
                elif successful_tools == 0:
                    details = "; ".join(failed_tools[:3]) or "toate tool-urile au eșuat"
                    reply = f"ERROR: Comanda nu s-a executat. Detalii: {details}"
                elif failed_tools:
                    details = "; ".join(failed_tools[:2])
                    reply = f"{reply}\n\nAvertisment execuție: {details}"
                elif not reply:
                    preview = "\n".join(last_tool_outputs[:2])
                    reply = (
                        "Acțiunea a fost executată cu succes. "
                        f"Detalii tool:\n{preview}"
                    )
            if not reply and self._is_temperature_set_intent(message):
                # Fallback explicit pentru intent-ul de termostat când modelul nu livrează conținut.
                await self.set_pending_action(message, conversation_id)
                reply = (
                    "Am înțeles comanda de setare temperatură. Confirmă cu 'da' și o execut imediat."
                )
            if not reply:
                reply = "Nu am primit conținut de la model."
        except Exception as err:
            _LOGGER.error("Eroare OpenAI API (mod stabil): %s", err)
            reply = f"ERROR: OpenAI request a eșuat în modul stabil. Detaliu: {err}"

        await self.add_to_history("user", message, conversation_id)
        await self.add_to_history("assistant", reply, conversation_id)
        return reply
