# OpenAI Chat cu Istoric

Integrare Home Assistant cu chat OpenAI care păstrează **istoricul conversațiilor** și are **memorie editabilă**.

## Instalare prin HACS

1. HACS → Integrări → ⋮ → Repository-uri custom
2. Adaugă: `https://github.com/minimi2024/open-ai`
3. Caută "OpenAI Chat cu Istoric" și instalează
4. Repornește Home Assistant

## Configurare panel

Adaugă în `configuration.yaml`:

```yaml
panel_custom:
  - name: openai-chat-panel
    sidebar_title: AI Chat
    sidebar_icon: mdi:robot
    url_path: openai-chat
    module_url: /api/openai_chat/static/openai_chat_panel.js
```

Repornește din nou, apoi Setări → Integrări → Adaugă → OpenAI Chat cu Istoric.
