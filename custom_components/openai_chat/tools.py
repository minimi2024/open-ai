"""Tools (function calling) pentru OpenAI - acces la Home Assistant."""

from __future__ import annotations

import json
import logging
from typing import Any

from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)

# Definiții tools pentru OpenAI API
OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_ha_entities",
            "description": "Obține lista entităților Home Assistant cu starea lor. Poți filtra după domain (light, switch, sensor, automation, script, etc.) sau căuta după nume.",
            "parameters": {
                "type": "object",
                "properties": {
                    "domain": {
                        "type": "string",
                        "description": "Filtrează după domain: light, switch, sensor, automation, script, climate, cover, media_player, etc. Lăsat gol pentru toate.",
                    },
                    "search": {
                        "type": "string",
                        "description": "Caută entități care conțin acest text în entity_id sau friendly_name.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "call_ha_service",
            "description": "Apelează un serviciu Home Assistant. Ex: light.turn_on, switch.turn_off, automation.trigger, script.run.",
            "parameters": {
                "type": "object",
                "properties": {
                    "domain": {
                        "type": "string",
                        "description": "Domain-ul serviciului: light, switch, automation, script, climate, cover, media_player, etc.",
                    },
                    "service": {
                        "type": "string",
                        "description": "Numele serviciului: turn_on, turn_off, trigger, run, set_temperature, etc.",
                    },
                    "entity_id": {
                        "type": "string",
                        "description": "Entity ID (ex: light.bucatarie) sau listă separată prin virgulă.",
                    },
                    "data": {
                        "type": "object",
                        "description": "Parametri suplimentari pentru serviciu (brightness, rgb_color, temperature, etc.)",
                    },
                },
                "required": ["domain", "service"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_ha_services",
            "description": "Listează serviciile disponibile în Home Assistant, opțional filtrate după domain.",
            "parameters": {
                "type": "object",
                "properties": {
                    "domain": {
                        "type": "string",
                        "description": "Filtrează după domain. Gol pentru toate.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_ha_file",
            "description": "Citește conținutul unui fișier din directorul config Home Assistant. Doar fișiere din config/ sunt permise.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Calea relativă la config (ex: configuration.yaml, automations.yaml, scripts.yaml, secrets.yaml)",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_ha_files",
            "description": "Listează fișierele și folderele din directorul config Home Assistant.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subdir": {
                        "type": "string",
                        "description": "Subdirector (ex: automations, scripts, packages). Gol pentru root config.",
                    },
                },
            },
        },
    },
]


async def execute_tool(
    hass: HomeAssistant,
    tool_name: str,
    arguments: dict[str, Any],
) -> str:
    """Execută un tool și returnează rezultatul ca string."""
    try:
        if tool_name == "get_ha_entities":
            return await _get_ha_entities(hass, arguments)
        if tool_name == "call_ha_service":
            return await _call_ha_service(hass, arguments)
        if tool_name == "get_ha_services":
            return await _get_ha_services(hass, arguments)
        if tool_name == "read_ha_file":
            return await _read_ha_file(hass, arguments)
        if tool_name == "list_ha_files":
            return await _list_ha_files(hass, arguments)
        return json.dumps({"error": f"Tool necunoscut: {tool_name}"})
    except Exception as e:
        _LOGGER.exception("Eroare la executarea tool %s: %s", tool_name, e)
        return json.dumps({"error": str(e)})


async def _get_ha_entities(hass: HomeAssistant, args: dict) -> str:
    """Obține entitățile HA."""
    domain = args.get("domain", "").strip().lower()
    search = args.get("search", "").strip().lower()

    states = hass.states.async_all(domain) if domain else hass.states.async_all()

    result = []
    for state in states:
        entity_id = state.entity_id
        friendly = state.attributes.get("friendly_name", entity_id)
        if search and search not in entity_id.lower() and search not in str(friendly).lower():
            continue
        result.append({
            "entity_id": entity_id,
            "state": state.state,
            "friendly_name": friendly,
            "attributes": {k: v for k, v in state.attributes.items() if k != "friendly_name"},
        })

    return json.dumps(result[:100], ensure_ascii=False, default=str)


async def _call_ha_service(hass: HomeAssistant, args: dict) -> str:
    """Apelează un serviciu HA."""
    domain = args.get("domain", "").strip()
    service = args.get("service", "").strip()
    entity_id = args.get("entity_id")
    data = args.get("data") or {}

    if not domain or not service:
        return json.dumps({"error": "domain și service sunt obligatorii"})

    service_data = dict(data)
    if entity_id:
        if isinstance(entity_id, str):
            entity_id = [e.strip() for e in entity_id.split(",") if e.strip()]
        service_data["entity_id"] = entity_id

    try:
        await hass.services.async_call(domain, service, service_data, blocking=True)
        return json.dumps({
            "success": True,
            "message": f"Serviciu {domain}.{service} executat cu succes",
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


async def _get_ha_services(hass: HomeAssistant, args: dict) -> str:
    """Listează serviciile HA."""
    domain_filter = args.get("domain", "").strip().lower()

    services = hass.services.async_services()
    result = {}
    for dom, svcs in services.items():
        if domain_filter and dom != domain_filter:
            continue
        result[dom] = list(svcs.keys())

    return json.dumps(result, ensure_ascii=False)


async def _read_ha_file(hass: HomeAssistant, args: dict) -> str:
    """Citește un fișier din config."""
    path = args.get("path", "").strip()
    if not path or ".." in path:
        return json.dumps({"error": "Cale invalidă"})

    full_path = hass.config.path(path)
    if not hass.config.is_allowed_path(full_path):
        return json.dumps({"error": f"Fișierul nu este în directorul config permis: {path}"})

    try:
        with open(full_path, encoding="utf-8") as f:
            content = f.read()
        return json.dumps({"content": content, "path": path})
    except FileNotFoundError:
        return json.dumps({"error": f"Fișier negăsit: {path}"})
    except Exception as e:
        return json.dumps({"error": str(e)})


async def _list_ha_files(hass: HomeAssistant, args: dict) -> str:
    """Listează fișierele din config."""
    subdir = args.get("subdir", "").strip()
    if subdir and (".." in subdir or subdir.startswith("/")):
        return json.dumps({"error": "Subdirector invalid"})

    base = hass.config.config_dir
    if subdir:
        base = hass.config.path(subdir)

    if not hass.config.is_allowed_path(base):
        return json.dumps({"error": "Director nu este permis"})

    try:
        import os
        items = []
        for name in sorted(os.listdir(base)):
            full = os.path.join(base, name)
            rel = os.path.join(subdir, name) if subdir else name
            items.append({
                "name": name,
                "path": rel,
                "is_dir": os.path.isdir(full),
            })
        return json.dumps(items, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})
