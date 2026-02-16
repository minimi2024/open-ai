"""Config flow pentru OpenAI Chat."""

from __future__ import annotations

import voluptuous as vol

from homeassistant import config_entries
from homeassistant.const import CONF_API_KEY
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
    DOMAIN,
    MODEL_CHOICES,
)


class OpenAIChatConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Config flow pentru OpenAI Chat."""

    VERSION = 1

    async def async_step_user(self, user_input=None):
        """Configurare prin UI."""
        errors = {}

        if user_input is not None:
            api_key = user_input.get(CONF_API_KEY, "").strip()
            if not api_key:
                errors["base"] = "Cheia API este obligatorie"
            else:
                smart_mode = bool(user_input.get(CONF_SMART_MODE, False))
                selected_model = user_input.get(
                    CONF_MODEL, DEFAULT_MODEL_SMART if smart_mode else DEFAULT_MODEL
                )
                return self.async_create_entry(
                    title="OpenAI Chat",
                    data={CONF_API_KEY: api_key},
                    options={
                        CONF_MODEL: selected_model,
                        CONF_MAX_TOKENS: user_input.get(
                            CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS
                        ),
                        CONF_TEMPERATURE: user_input.get(
                            CONF_TEMPERATURE, DEFAULT_TEMPERATURE
                        ),
                        CONF_SMART_MODE: smart_mode,
                    },
                )

        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_API_KEY): str,
                    vol.Optional(CONF_SMART_MODE, default=False): cv.boolean,
                    vol.Optional(CONF_MODEL, default=DEFAULT_MODEL): vol.In(
                        MODEL_CHOICES
                    ),
                    vol.Optional(
                        CONF_MAX_TOKENS, default=DEFAULT_MAX_TOKENS
                    ): vol.All(vol.Coerce(int), vol.Range(min=100, max=128000)),
                    vol.Optional(
                        CONF_TEMPERATURE, default=DEFAULT_TEMPERATURE
                    ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=2.0)),
                }
            ),
            errors=errors,
        )

    @staticmethod
    def async_get_options_flow(config_entry):
        """Opțiuni pentru integrare."""
        from .options_flow import OpenAIChatOptionsFlow
        return OpenAIChatOptionsFlow(config_entry)
