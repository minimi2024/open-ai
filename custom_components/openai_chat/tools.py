"""Tools (function calling) pentru OpenAI - acces la Home Assistant și internet."""

from __future__ import annotations

import json
import logging
import os
import re
import asyncio
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.helpers.httpx_client import get_async_client

_LOGGER = logging.getLogger(__name__)


def _is_permitted_config_path(hass: HomeAssistant, path: str) -> bool:
    """Returnează True dacă path este în config dir sau în allowlist."""
    try:
        abs_path = os.path.abspath(path)
        config_dir = os.path.abspath(hass.config.config_dir)
        # Permitem întotdeauna directorul de config HA și subdirectoarele lui
        if abs_path == config_dir or abs_path.startswith(f"{config_dir}{os.sep}"):
            return True
    except Exception:
        pass

    return hass.config.is_allowed_path(path)

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
            "name": "write_ha_file",
            "description": "Scrie/înlocuiește conținutul unui fișier din config Home Assistant. Necesită confirm explicit pentru siguranță. Creează backup .bak înainte de overwrite.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Calea relativă la config (ex: configuration.yaml, automations.yaml, scripts.yaml)",
                    },
                    "content": {
                        "type": "string",
                        "description": "Conținutul complet care va fi scris în fișier.",
                    },
                    "confirm": {
                        "type": "boolean",
                        "description": "Trebuie să fie true pentru a permite scrierea.",
                    },
                    "dry_run": {
                        "type": "boolean",
                        "description": "Dacă este true, doar simulează scrierea și returnează ce s-ar schimba.",
                    },
                    "create_dirs": {
                        "type": "boolean",
                        "description": "Dacă este true, creează directoarele lipsă.",
                    },
                },
                "required": ["path", "content", "confirm"],
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
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Caută pe internet folosind DuckDuckGo. Folosește pentru informații actuale, știri, vreme, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Cuvintele de căutat pe internet.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Număr maxim de rezultate (default 5).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Citește conținutul unei pagini web de la un URL. Folosește pentru a obține informații de pe un site specific.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL-ul complet al paginii (ex: https://example.com/article)",
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_lovelace_dashboards",
            "description": "Listează dashboardurile Lovelace din Home Assistant (Overview, Map, etc.).",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_lovelace_views",
            "description": "Obține taburile (views) ale unui dashboard Lovelace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dashboard": {
                        "type": "string",
                        "description": "URL path-ul dashboardului: lovelace (Overview), map, etc. Lăsat gol pentru Overview.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_lovelace_view",
            "description": "Adaugă un tab (view) nou la un dashboard Lovelace. Poți crea taburi noi în dashboard.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dashboard": {
                        "type": "string",
                        "description": "Dashboard-ul: lovelace (Overview), map, etc. Default: lovelace.",
                    },
                    "title": {
                        "type": "string",
                        "description": "Titlul tabului (ex: Living Room, Bedroom).",
                    },
                    "path": {
                        "type": "string",
                        "description": "Path URL pentru tab (ex: living-room). Trebuie să conțină cratimă. Dacă gol, se generează din titlu.",
                    },
                    "icon": {
                        "type": "string",
                        "description": "Iconița Material Design (ex: mdi:sofa, mdi:bed). Default: mdi:view-dashboard.",
                    },
                },
                "required": ["title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_lovelace_dashboard",
            "description": "Creează un dashboard Lovelace nou. Apare în sidebar sub dashboarduri.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Titlul dashboardului (ex: Casa mea, Garaj).",
                    },
                    "url_path": {
                        "type": "string",
                        "description": "URL path (ex: casa-mea). Trebuie să conțină cratimă. Dacă gol, se generează din titlu.",
                    },
                    "icon": {
                        "type": "string",
                        "description": "Iconița Material Design (ex: mdi:home). Default: mdi:view-dashboard.",
                    },
                },
                "required": ["title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_lovelace_view",
            "description": "Șterge un tab (view) dintr-un dashboard Lovelace după path sau title.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dashboard": {
                        "type": "string",
                        "description": "Dashboard-ul: lovelace (Overview), map etc. Default: lovelace.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Path-ul view-ului de șters (preferat).",
                    },
                    "title": {
                        "type": "string",
                        "description": "Titlul view-ului de șters (fallback dacă path lipsește).",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dedupe_lovelace_views",
            "description": "Elimină duplicatele de taburi dintr-un dashboard Lovelace, păstrând primul tab pentru fiecare cheie.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dashboard": {
                        "type": "string",
                        "description": "Dashboard-ul: lovelace (Overview), map etc. Default: lovelace.",
                    },
                    "by": {
                        "type": "string",
                        "description": "Cheia de deduplicare: path, title, sau path_or_title (default).",
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
        if tool_name == "write_ha_file":
            return await _write_ha_file(hass, arguments)
        if tool_name == "list_ha_files":
            return await _list_ha_files(hass, arguments)
        if tool_name == "web_search":
            return await _web_search(hass, arguments)
        if tool_name == "fetch_url":
            return await _fetch_url(hass, arguments)
        if tool_name == "list_lovelace_dashboards":
            return await _list_lovelace_dashboards(hass, arguments)
        if tool_name == "get_lovelace_views":
            return await _get_lovelace_views(hass, arguments)
        if tool_name == "add_lovelace_view":
            return await _add_lovelace_view(hass, arguments)
        if tool_name == "create_lovelace_dashboard":
            return await _create_lovelace_dashboard(hass, arguments)
        if tool_name == "delete_lovelace_view":
            return await _delete_lovelace_view(hass, arguments)
        if tool_name == "dedupe_lovelace_views":
            return await _dedupe_lovelace_views(hass, arguments)
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
    entity_ids: list[str] = []
    if entity_id:
        if isinstance(entity_id, str):
            entity_ids = [e.strip() for e in entity_id.split(",") if e.strip()]
        elif isinstance(entity_id, list):
            entity_ids = [str(e).strip() for e in entity_id if str(e).strip()]
        else:
            entity_ids = [str(entity_id).strip()]

        # Verificare entități înainte de execuție
        missing = [eid for eid in entity_ids if hass.states.get(eid) is None]
        if missing:
            return json.dumps(
                {
                    "error": "Entități inexistente",
                    "missing_entities": missing,
                    "hint": "Verifică exact entity_id în Developer Tools -> States",
                },
                ensure_ascii=False,
            )

        service_data["entity_id"] = entity_ids

    before_states: dict[str, str] = {}
    if entity_ids:
        before_states = {
            eid: (hass.states.get(eid).state if hass.states.get(eid) is not None else "unknown")
            for eid in entity_ids
        }

    try:
        await hass.services.async_call(domain, service, service_data, blocking=True)
        # Lăsăm puțin timp pentru propagarea noii stări în state machine.
        await asyncio.sleep(0.25)

        after_states: dict[str, str] = {}
        changed_entities: list[str] = []
        unchanged_entities: list[str] = []

        if entity_ids:
            after_states = {
                eid: (hass.states.get(eid).state if hass.states.get(eid) is not None else "unknown")
                for eid in entity_ids
            }
            for eid in entity_ids:
                if before_states.get(eid) != after_states.get(eid):
                    changed_entities.append(eid)
                else:
                    unchanged_entities.append(eid)

        result: dict[str, Any] = {
            "success": True,
            "service": f"{domain}.{service}",
            "message": f"Serviciu {domain}.{service} executat",
        }

        if entity_ids:
            result["before_states"] = before_states
            result["after_states"] = after_states
            result["changed_entities"] = changed_entities
            result["unchanged_entities"] = unchanged_entities
            if not changed_entities:
                result["warning"] = (
                    "Serviciul a fost apelat, dar starea entităților nu s-a schimbat."
                )

        return json.dumps(result, ensure_ascii=False)
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
    if not _is_permitted_config_path(hass, full_path):
        return json.dumps({"error": f"Fișierul nu este în directorul config permis: {path}"})

    try:
        with open(full_path, encoding="utf-8") as f:
            content = f.read()
        return json.dumps({"content": content, "path": path})
    except FileNotFoundError:
        return json.dumps({"error": f"Fișier negăsit: {path}"})
    except Exception as e:
        return json.dumps({"error": str(e)})


async def _write_ha_file(hass: HomeAssistant, args: dict) -> str:
    """Scrie un fișier în config (cu protecții)."""
    path = (args.get("path") or "").strip()
    content = args.get("content")
    confirm = bool(args.get("confirm", False))
    dry_run = bool(args.get("dry_run", False))
    create_dirs = bool(args.get("create_dirs", False))

    if not path or ".." in path or path.startswith("/"):
        return json.dumps({"error": "Cale invalidă. Folosește doar cale relativă în /config."})
    if content is None:
        return json.dumps({"error": "Conținutul (content) este obligatoriu."})
    if not confirm:
        return json.dumps(
            {
                "error": "Scriere blocată pentru siguranță. Trimite confirm=true pentru a permite scrierea."
            }
        )

    full_path = hass.config.path(path)
    if not _is_permitted_config_path(hass, full_path):
        return json.dumps({"error": f"Fișierul nu este permis: {path}"})

    target_dir = os.path.dirname(full_path)
    if target_dir and not os.path.isdir(target_dir):
        if create_dirs:
            try:
                os.makedirs(target_dir, exist_ok=True)
            except Exception as e:
                return json.dumps({"error": f"Nu pot crea directoarele: {e}"})
        else:
            return json.dumps(
                {
                    "error": "Directorul nu există. Folosește create_dirs=true dacă vrei să fie creat automat."
                }
            )

    existing_content = None
    file_exists = os.path.isfile(full_path)
    if file_exists:
        try:
            with open(full_path, encoding="utf-8") as f:
                existing_content = f.read()
        except Exception as e:
            return json.dumps({"error": f"Nu pot citi fișierul existent: {e}"})

    changed = existing_content != content
    if dry_run:
        return json.dumps(
            {
                "success": True,
                "dry_run": True,
                "path": path,
                "exists": file_exists,
                "changed": changed,
                "old_length": len(existing_content) if existing_content is not None else 0,
                "new_length": len(content),
            },
            ensure_ascii=False,
        )

    backup_path = None
    if file_exists:
        backup_path = f"{full_path}.bak"
        try:
            with open(backup_path, "w", encoding="utf-8") as f:
                f.write(existing_content if existing_content is not None else "")
        except Exception as e:
            return json.dumps({"error": f"Nu pot crea backup-ul .bak: {e}"})

    try:
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception as e:
        return json.dumps({"error": f"Nu pot scrie fișierul: {e}"})

    return json.dumps(
        {
            "success": True,
            "path": path,
            "changed": changed,
            "backup": f"{path}.bak" if backup_path else None,
            "bytes_written": len(content.encode("utf-8")),
        },
        ensure_ascii=False,
    )


async def _list_ha_files(hass: HomeAssistant, args: dict) -> str:
    """Listează fișierele din config."""
    subdir = args.get("subdir", "").strip()
    if subdir and (".." in subdir or subdir.startswith("/")):
        return json.dumps({"error": "Subdirector invalid"})

    base = hass.config.config_dir
    if subdir:
        base = hass.config.path(subdir)

    if not _is_permitted_config_path(hass, base):
        return json.dumps({"error": "Director nu este permis"})

    try:
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


async def _web_search(hass: HomeAssistant, args: dict) -> str:
    """Caută pe internet cu DuckDuckGo."""
    query = args.get("query", "").strip()
    max_results = min(args.get("max_results", 5) or 5, 10)

    if not query:
        return json.dumps({"error": "Query-ul de căutare este obligatoriu"})

    def _do_search():
        from ddgs import DDGS
        results = list(DDGS().text(query, max_results=max_results, region="wt-wt"))
        return [{"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")} for r in results]

    try:
        results = await hass.async_add_executor_job(_do_search)
        return json.dumps({"results": results, "query": query}, ensure_ascii=False)
    except Exception as e:
        _LOGGER.exception("Eroare la căutare web: %s", e)
        return json.dumps({"error": str(e)})


async def _fetch_url(hass: HomeAssistant, args: dict) -> str:
    """Citește conținutul unei pagini web."""
    url = args.get("url", "").strip()
    if not url:
        return json.dumps({"error": "URL-ul este obligatoriu"})

    if not url.startswith(("http://", "https://")):
        return json.dumps({"error": "URL invalid - trebuie să înceapă cu http:// sau https://"})

    try:
        client = get_async_client(hass)
        response = await client.get(url, timeout=10)
        response.raise_for_status()
        html = response.text

        # Extrage text din HTML (elimină scripturi, style, etc.)
        text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        text = text[:8000]  # Limită pentru context

        return json.dumps({"content": text, "url": url}, ensure_ascii=False)
    except Exception as e:
        _LOGGER.exception("Eroare la fetch URL %s: %s", url, e)
        return json.dumps({"error": str(e)})


def _get_lovelace_data(hass: HomeAssistant):
    """Obține LovelaceData din hass.data."""
    return hass.data.get("lovelace")


async def _list_lovelace_dashboards(hass: HomeAssistant, args: dict) -> str:
    """Listează dashboardurile Lovelace."""
    lovelace_data = _get_lovelace_data(hass)
    if not lovelace_data:
        return json.dumps({"error": "Lovelace nu este disponibil"})

    result = []
    for url_path, config_obj in lovelace_data.dashboards.items():
        cfg = config_obj.config if hasattr(config_obj, "config") else None
        if cfg:
            up = url_path or "lovelace"
            result.append({
                "url_path": up,
                "title": cfg.get("title", up),
                "icon": cfg.get("icon", "mdi:view-dashboard"),
            })
    return json.dumps(result, ensure_ascii=False)


async def _get_lovelace_views(hass: HomeAssistant, args: dict) -> str:
    """Obține view-urile unui dashboard."""
    lovelace_data = _get_lovelace_data(hass)
    if not lovelace_data:
        return json.dumps({"error": "Lovelace nu este disponibil"})

    dashboard = (args.get("dashboard") or "lovelace").strip().lower()
    if dashboard == "lovelace":
        url_path = "lovelace"
    else:
        url_path = dashboard

    config_obj = (
        lovelace_data.dashboards.get(url_path)
        or lovelace_data.dashboards.get("lovelace")
        or lovelace_data.dashboards.get(None)
    )
    if not config_obj:
        return json.dumps({"error": f"Dashboard '{dashboard}' negăsit"})

    try:
        config = await config_obj.async_load(False)
    except Exception as e:
        return json.dumps({"error": str(e)})

    views = config.get("views", [])
    result = [{"title": v.get("title", ""), "path": v.get("path", ""), "icon": v.get("icon", "")} for v in views]
    return json.dumps({"views": result, "dashboard": url_path}, ensure_ascii=False)


async def _add_lovelace_view(hass: HomeAssistant, args: dict) -> str:
    """Adaugă un view la un dashboard."""
    lovelace_data = _get_lovelace_data(hass)
    if not lovelace_data:
        return json.dumps({"error": "Lovelace nu este disponibil"})

    dashboard = (args.get("dashboard") or "lovelace").strip().lower()
    title = args.get("title", "").strip()
    path = args.get("path", "").strip()
    icon = args.get("icon", "mdi:view-dashboard").strip() or "mdi:view-dashboard"

    if not title:
        return json.dumps({"error": "Titlul este obligatoriu"})

    if dashboard == "lovelace":
        url_path = "lovelace"
    else:
        url_path = dashboard

    config_obj = (
        lovelace_data.dashboards.get(url_path)
        or lovelace_data.dashboards.get("lovelace")
        or lovelace_data.dashboards.get(None)
    )
    if not config_obj:
        return json.dumps({"error": f"Dashboard '{dashboard}' negăsit"})

    if not path:
        path = title.lower().replace(" ", "-").replace("_", "-")
        path = "".join(c for c in path if c.isalnum() or c == "-")
        if not path or "-" not in path:
            path = f"{path}-1" if path else "view-1"

    try:
        config = await config_obj.async_load(False)
    except Exception as e:
        return json.dumps({"error": str(e)})

    views = config.get("views", [])
    existing_paths = {v.get("path") for v in views if v.get("path")}
    if path in existing_paths:
        path = f"{path}-{len(views) + 1}"

    new_view = {"title": title, "path": path, "icon": icon, "cards": []}
    views.append(new_view)
    config["views"] = views

    try:
        await config_obj.async_save(config)
        return json.dumps({
            "success": True,
            "message": f"Tab '{title}' adăugat la dashboard",
            "path": path,
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})


async def _create_lovelace_dashboard(hass: HomeAssistant, args: dict) -> str:
    """Creează un dashboard nou (scrie în storage, necesită repornire)."""
    from homeassistant.util import slugify

    title = args.get("title", "").strip()
    url_path = args.get("path", "").strip() or args.get("url_path", "").strip()
    icon = args.get("icon", "mdi:view-dashboard").strip() or "mdi:view-dashboard"

    if not title:
        return json.dumps({"error": "Titlul este obligatoriu"})

    if not url_path:
        url_path = slugify(title, separator="-")
    if "-" not in url_path:
        url_path = f"{url_path}-1"

    from homeassistant.helpers import storage

    dashboards_store = storage.Store(hass, 1, "lovelace_dashboards")
    data = await dashboards_store.async_load() or {"items": []}
    items = data.get("items", [])

    for item in items:
        if item.get("url_path") == url_path:
            return json.dumps({"error": f"Dashboard cu path '{url_path}' există deja"})

    new_item = {
        "id": url_path,
        "url_path": url_path,
        "title": title,
        "icon": icon,
        "show_in_sidebar": True,
        "require_admin": False,
        "mode": "storage",
    }
    items.append(new_item)
    await dashboards_store.async_save({"items": items})

    config_store = storage.Store(hass, 1, f"lovelace.{url_path}")
    await config_store.async_save({"config": {"views": []}})

    return json.dumps({
        "success": True,
        "message": f"Dashboard '{title}' creat. Repornește Home Assistant pentru a-l vedea în sidebar.",
        "url_path": url_path,
    }, ensure_ascii=False)


def _resolve_dashboard_config_obj(hass: HomeAssistant, dashboard: str | None):
    """Returnează config obj pentru dashboard, cu fallback safe."""
    lovelace_data = _get_lovelace_data(hass)
    if not lovelace_data:
        return None, "Lovelace nu este disponibil"
    requested = (dashboard or "lovelace").strip().lower()
    url_path = "lovelace" if requested == "lovelace" else requested
    config_obj = (
        lovelace_data.dashboards.get(url_path)
        or lovelace_data.dashboards.get("lovelace")
        or lovelace_data.dashboards.get(None)
    )
    if not config_obj:
        return None, f"Dashboard '{requested}' negăsit"
    return config_obj, None


async def _delete_lovelace_view(hass: HomeAssistant, args: dict) -> str:
    """Șterge un view din dashboard după path/title."""
    config_obj, err = _resolve_dashboard_config_obj(hass, args.get("dashboard"))
    if err:
        return json.dumps({"error": err}, ensure_ascii=False)

    path = (args.get("path") or "").strip()
    title = (args.get("title") or "").strip()
    if not path and not title:
        return json.dumps(
            {"error": "Trebuie să trimiți path sau title pentru view-ul de șters."},
            ensure_ascii=False,
        )

    try:
        config = await config_obj.async_load(False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)

    views = config.get("views", [])
    if not isinstance(views, list):
        return json.dumps({"error": "Config Lovelace invalid: 'views' nu este listă."})

    kept = []
    removed = []
    for v in views:
        v_path = str(v.get("path", ""))
        v_title = str(v.get("title", ""))
        match = (path and v_path == path) or (not path and title and v_title == title)
        if match:
            removed.append({"title": v_title, "path": v_path})
        else:
            kept.append(v)

    if not removed:
        return json.dumps(
            {"success": False, "message": "Nu am găsit view de șters.", "removed": []},
            ensure_ascii=False,
        )

    config["views"] = kept
    try:
        await config_obj.async_save(config)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)

    return json.dumps(
        {
            "success": True,
            "removed_count": len(removed),
            "removed": removed[:20],
            "remaining_views": len(kept),
        },
        ensure_ascii=False,
    )


async def _dedupe_lovelace_views(hass: HomeAssistant, args: dict) -> str:
    """Elimină duplicatele de view-uri."""
    config_obj, err = _resolve_dashboard_config_obj(hass, args.get("dashboard"))
    if err:
        return json.dumps({"error": err}, ensure_ascii=False)

    by = (args.get("by") or "path_or_title").strip().lower()
    if by not in {"path", "title", "path_or_title"}:
        by = "path_or_title"

    try:
        config = await config_obj.async_load(False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)

    views = config.get("views", [])
    if not isinstance(views, list):
        return json.dumps({"error": "Config Lovelace invalid: 'views' nu este listă."})

    seen: set[str] = set()
    kept = []
    removed = []

    for idx, v in enumerate(views):
        v_path = str(v.get("path", "")).strip()
        v_title = str(v.get("title", "")).strip()
        if by == "path":
            key = f"path:{v_path.lower()}"
        elif by == "title":
            key = f"title:{v_title.lower()}"
        else:
            key = f"path:{v_path.lower()}" if v_path else f"title:{v_title.lower()}"

        if key in seen:
            removed.append(
                {"index": idx, "title": v_title, "path": v_path, "dedupe_key": key}
            )
            continue

        seen.add(key)
        kept.append(v)

    if not removed:
        return json.dumps(
            {
                "success": True,
                "message": "Nu există duplicate de eliminat.",
                "removed_count": 0,
                "remaining_views": len(kept),
            },
            ensure_ascii=False,
        )

    config["views"] = kept
    try:
        await config_obj.async_save(config)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)

    return json.dumps(
        {
            "success": True,
            "removed_count": len(removed),
            "remaining_views": len(kept),
            "removed": removed[:50],
            "dedupe_by": by,
        },
        ensure_ascii=False,
    )
