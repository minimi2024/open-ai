"""Rute HTTP pentru API-ul OpenAI Chat."""

from __future__ import annotations

from http import HTTPStatus

from aiohttp import web
from homeassistant.helpers.http import HomeAssistantView

from .const import DOMAIN


def _get_coordinator(request: web.Request):
    """Obține coordinator-ul din primul config entry disponibil."""
    hass = request.app["hass"]
    data = hass.data.get(DOMAIN, {})
    if not data:
        return None
    return next(iter(data.values()), None)


class OpenAIChatView(HomeAssistantView):
    """View pentru trimitere mesaje chat."""

    url = "/api/openai_chat/chat"
    name = "api:openai_chat:chat"
    requires_auth = True

    async def post(self, request: web.Request) -> web.Response:
        """Procesează mesajul chat."""
        try:
            data = await request.json()
        except Exception:
            return self.json(
                {"error": "JSON invalid"},
                status=HTTPStatus.BAD_REQUEST,
            )

        message = data.get("message", "").strip()
        conversation_id = data.get("conversation_id")

        if not message:
            return self.json(
                {"error": "Mesajul este obligatoriu"},
                status=HTTPStatus.BAD_REQUEST,
            )

        coordinator = _get_coordinator(request)
        if not coordinator:
            return self.json(
                {"error": "Integrarea nu este configurată"},
                status=HTTPStatus.SERVICE_UNAVAILABLE,
            )

        try:
            reply = await coordinator.chat(message, conversation_id)
            return self.json({"reply": reply})
        except Exception as e:
            return self.json(
                {"error": str(e)},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )


class OpenAIHistoryView(HomeAssistantView):
    """View pentru istoricul conversațiilor."""

    url = "/api/openai_chat/history"
    name = "api:openai_chat:history"
    requires_auth = True

    async def get(self, request: web.Request) -> web.Response:
        """Obține istoricul pentru o conversație."""
        coordinator = _get_coordinator(request)
        if not coordinator:
            return self.json(
                {"error": "Integrarea nu este configurată"},
                status=HTTPStatus.SERVICE_UNAVAILABLE,
            )

        conversation_id = request.query.get("conversation_id")
        history = await coordinator.get_history(conversation_id)
        return self.json({"history": history})

    async def delete(self, request: web.Request) -> web.Response:
        """Șterge istoricul."""
        coordinator = _get_coordinator(request)
        if not coordinator:
            return self.json(
                {"error": "Integrarea nu este configurată"},
                status=HTTPStatus.SERVICE_UNAVAILABLE,
            )

        try:
            data = await request.json() if request.content_length else {}
        except Exception:
            data = {}

        conversation_id = data.get("conversation_id")
        await coordinator.clear_history(conversation_id)
        return self.json({"success": True})


class OpenAIMemoryView(HomeAssistantView):
    """View pentru memoria editabilă (system prompt)."""

    url = "/api/openai_chat/memory"
    name = "api:openai_chat:memory"
    requires_auth = True

    async def get(self, request: web.Request) -> web.Response:
        """Obține memoria curentă."""
        coordinator = _get_coordinator(request)
        if not coordinator:
            return self.json(
                {"error": "Integrarea nu este configurată"},
                status=HTTPStatus.SERVICE_UNAVAILABLE,
            )

        memory = await coordinator.get_memory()
        return self.json({"memory": memory})

    async def post(self, request: web.Request) -> web.Response:
        """Actualizează memoria."""
        try:
            data = await request.json()
        except Exception:
            return self.json(
                {"error": "JSON invalid"},
                status=HTTPStatus.BAD_REQUEST,
            )

        memory = data.get("memory", "")
        coordinator = _get_coordinator(request)
        if not coordinator:
            return self.json(
                {"error": "Integrarea nu este configurată"},
                status=HTTPStatus.SERVICE_UNAVAILABLE,
            )

        await coordinator.set_memory(memory)
        return self.json({"success": True, "memory": memory})


class OpenAIConversationsView(HomeAssistantView):
    """View pentru lista de conversații."""

    url = "/api/openai_chat/conversations"
    name = "api:openai_chat:conversations"
    requires_auth = True

    async def get(self, request: web.Request) -> web.Response:
        """Obține lista conversațiilor."""
        coordinator = _get_coordinator(request)
        if not coordinator:
            return self.json(
                {"error": "Integrarea nu este configurată"},
                status=HTTPStatus.SERVICE_UNAVAILABLE,
            )

        conversations = await coordinator.get_conversation_list()
        return self.json({"conversations": conversations})
