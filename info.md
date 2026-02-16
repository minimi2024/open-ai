# OpenAI Chat cu Istoric

Integrare pentru Home Assistant cu chat OpenAI care păstrează **istoricul conversațiilor** și are **memorie editabilă**.

## Caracteristici

- Istoric persistent (conversațiile rămân după repornire)
- Memorie editabilă (system prompt personalizabil)
- Multiple conversații
- Panel în sidebar sub dashboarduri

## Instalare

1. Adaugă acest repository în HACS (Repository-uri custom)
2. Instalează "OpenAI Chat cu Istoric"
3. Repornește Home Assistant
4. Adaugă în `configuration.yaml`:

```yaml
panel_custom:
  - name: openai-chat-panel
    sidebar_title: AI Chat
    sidebar_icon: mdi:robot
    url_path: openai-chat
    module_url: /api/openai_chat/static/openai_chat_panel.js
```

5. Repornește din nou
6. Setări → Integrări → Adaugă → OpenAI Chat cu Istoric
