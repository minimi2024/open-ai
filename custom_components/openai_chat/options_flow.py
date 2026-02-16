"""Options flow pentru OpenAI Chat."""

from __future__ import annotations

import voluptuous as vol

from homeassistant import config_entries
from homeassistant.helpers import config_validation as cv

from .const import (
    CONF_MAX_TOKENS,
    CONF_MODEL,
    CONF_SMART_MODE,
    CONF_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_MODEL_SMART,
    DEFAULT_TEMPERATURE,
    MODEL_CHOICES,
)


class OpenAIChatOptionsFlow(config_entries.OptionsFlow):
    """Flow pentru opțiuni OpenAI Chat."""

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        """Inițializare."""
        self.config_entry = config_entry

    async def async_step_init(self, user_input=None):
        """Gestionare opțiuni."""
        if user_input is not None:
            return self.async_create_entry(title="", data=user_input)

        options = self.config_entry.options or {}
        try:
            smart_mode = bool(options.get(CONF_SMART_MODE, False))
        except (TypeError, ValueError):
            smart_mode = False
        try:
            max_tokens = int(options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS))
        except (TypeError, ValueError):
            max_tokens = DEFAULT_MAX_TOKENS
        max_tokens = max(100, min(128000, max_tokens))
        try:
            temperature = float(options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE))
        except (TypeError, ValueError):
            temperature = DEFAULT_TEMPERATURE
        temperature = max(0.0, min(2.0, temperature))
        selected_model = options.get(CONF_MODEL)
        if selected_model not in MODEL_CHOICES:
            selected_model = DEFAULT_MODEL_SMART if smart_mode else DEFAULT_MODEL

        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(
                {
                    vol.Optional(CONF_SMART_MODE, default=smart_mode): cv.boolean,
                    vol.Optional(CONF_MODEL, default=selected_model): vol.In(
                        MODEL_CHOICES
                    ),
                    vol.Optional(CONF_MAX_TOKENS, default=max_tokens): vol.All(
                        vol.Coerce(int), vol.Range(min=100, max=128000)
                    ),
                    vol.Optional(CONF_TEMPERATURE, default=temperature): vol.All(
                        vol.Coerce(float), vol.Range(min=0.0, max=2.0)
                    ),
                }
            ),
        )
