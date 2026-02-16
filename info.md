# OpenAI Chat cu Istoric

Integrare pentru Home Assistant cu chat OpenAI care păstrează **istoricul conversațiilor**, are **memorie editabilă** și **acces complet la Home Assistant**.

## Caracteristici

- Istoric persistent (conversațiile rămân după repornire)
- Memorie editabilă (system prompt personalizabil)
- Multiple conversații
- Panel în sidebar sub dashboarduri
- **Acces la Home Assistant**: AI-ul poate vedea entități, apela servicii, declanșa automatizări
- **Acces la fișiere**: citește configuration.yaml, automations.yaml, etc. din config

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
