"""Options flow pentru OpenAI Chat."""

from __future__ import annotations

import voluptuous as vol

from homeassistant import config_entries

from .const import (
    CONF_MAX_TOKENS,
    CONF_MODEL,
    CONF_SMART_MODE,
    CONF_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
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
        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(
                {
                    vol.Optional(
                        CONF_SMART_MODE,
                        default=options.get(CONF_SMART_MODE, False),
                    ): bool,
                    vol.Optional(
                        CONF_MODEL,
                        default=options.get(CONF_MODEL, DEFAULT_MODEL),
                    ): str,
                    vol.Optional(
                        CONF_MAX_TOKENS,
                        default=options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS),
                    ): vol.All(vol.Coerce(int), vol.Range(min=100, max=128000)),
                    vol.Optional(
                        CONF_TEMPERATURE,
                        default=options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE),
                    ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=2.0)),
                }
            ),
        )
