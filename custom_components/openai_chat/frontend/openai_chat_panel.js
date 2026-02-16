/**
 * Panel OpenAI Chat pentru Home Assistant
 * Cu istoric persistent și memorie editabilă
 */

import { LitElement, html, css } from "https://unpkg.com/lit@3.1.0?module";

class OpenAIChatPanel extends LitElement {
  static get properties() {
    return {
      hass: { type: Object },
      narrow: { type: Boolean },
      route: { type: Object },
      panel: { type: Object },
      _messages: { state: true },
      _input: { state: true },
      _loading: { state: true },
      _memory: { state: true },
      _showMemory: { state: true },
      _conversations: { state: true },
      _currentConversation: { state: true },
      _error: { state: true },
    };
  }

  constructor() {
    super();
    this._messages = [];
    this._input = "";
    this._loading = false;
    this._memory = "";
    this._showMemory = false;
    this._conversations = [];
    this._currentConversation = "default";
    this._error = null;
  }

  async connectedCallback() {
    super.connectedCallback();
    await this._loadData();
  }

  async _loadData() {
    await Promise.all([
      this._loadHistory(),
      this._loadMemory(),
      this._loadConversations(),
    ]);
  }

  async _loadHistory() {
    try {
      const url =
        this._currentConversation && this._currentConversation !== "default"
          ? `/api/openai_chat/history?conversation_id=${encodeURIComponent(this._currentConversation)}`
          : "/api/openai_chat/history";
      const resp = await this._fetch(url);
      if (resp.ok) {
        const data = await resp.json();
        this._messages = data.history || [];
      }
    } catch (e) {
      this._error = "Nu s-a putut încărca istoricul";
    }
  }

  async _loadMemory() {
    try {
      const resp = await this._fetch("/api/openai_chat/memory");
      if (resp.ok) {
        const data = await resp.json();
        this._memory = data.memory || "";
      }
    } catch (e) {
      this._error = "Nu s-a putut încărca memoria";
    }
  }

  async _loadConversations() {
    try {
      const resp = await this._fetch("/api/openai_chat/conversations");
      if (resp.ok) {
        const data = await resp.json();
        this._conversations = data.conversations || [];
      }
    } catch (e) {
      this._conversations = [];
    }
  }

  _fetch(url, options = {}) {
    const headers = {
      "Content-Type": "application/json",
      ...options.headers,
    };
    const token = this.hass?.auth?.data?.access_token;
    if (token) {
      headers["Authorization"] = `Bearer ${token}`;
    }
    return fetch(url, {
      ...options,
      credentials: "same-origin",
      headers,
    });
  }

  async _sendMessage() {
    const msg = this._input.trim();
    if (!msg || this._loading) return;

    this._input = "";
    this._error = null;
    this._loading = true;

    this._messages = [...this._messages, { role: "user", content: msg }];

    try {
      const resp = await this._fetch("/api/openai_chat/chat", {
        method: "POST",
        body: JSON.stringify({
          message: msg,
          conversation_id: this._currentConversation === "default" ? null : this._currentConversation,
        }),
      });

      const data = await resp.json();

      if (!resp.ok) {
        throw new Error(data.error || "Eroare la trimitere");
      }

      this._messages = [...this._messages, { role: "assistant", content: data.reply }];
      await this._loadConversations();
    } catch (e) {
      this._error = e.message || "Eroare la comunicarea cu OpenAI";
      this._messages = this._messages.slice(0, -1);
    } finally {
      this._loading = false;
    }
  }

  async _saveMemory() {
    try {
      const resp = await this._fetch("/api/openai_chat/memory", {
        method: "POST",
        body: JSON.stringify({ memory: this._memory }),
      });
      if (resp.ok) {
        this._showMemory = false;
      }
    } catch (e) {
      this._error = "Nu s-a putut salva memoria";
    }
  }

  async _clearHistory() {
    try {
      await this._fetch("/api/openai_chat/history", {
        method: "DELETE",
        body: JSON.stringify({
          conversation_id: this._currentConversation === "default" ? null : this._currentConversation,
        }),
      });
      this._messages = [];
      await this._loadConversations();
    } catch (e) {
      this._error = "Nu s-a putut șterge istoricul";
    }
  }

  _selectConversation(id) {
    this._currentConversation = id || null;
    this._loadHistory();
  }

  _newConversation() {
    this._currentConversation = "conv_" + Date.now();
    this._messages = [];
    this._conversations = [
      {
        id: this._currentConversation,
        title: "Conversație nouă",
        message_count: 0,
      },
      ...this._conversations.filter((c) => c.id !== "default"),
    ];
  }

  _handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      this._sendMessage();
    }
  }

  static get styles() {
    return css`
      :host {
        display: block;
        height: 100%;
        background: var(--ha-card-background, #1c1c1c);
        color: var(--primary-text-color, #e1e1e1);
      }

      .container {
        display: flex;
        flex-direction: column;
        height: 100%;
        max-width: 900px;
        margin: 0 auto;
      }

      .header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 16px 20px;
        border-bottom: 1px solid var(--divider-color, #333);
        flex-shrink: 0;
      }

      .header h1 {
        margin: 0;
        font-size: 1.25rem;
        font-weight: 500;
      }

      .header-actions {
        display: flex;
        gap: 8px;
      }

      .header-actions ha-icon-button {
        --mdc-icon-button-size: 40px;
      }

      .sidebar {
        display: flex;
        flex-direction: column;
        width: 220px;
        border-right: 1px solid var(--divider-color, #333);
        padding: 12px 0;
        flex-shrink: 0;
      }

      .sidebar-title {
        padding: 0 16px 8px;
        font-size: 0.75rem;
        text-transform: uppercase;
        color: var(--secondary-text-color, #888);
      }

      .conversation-item {
        padding: 10px 16px;
        cursor: pointer;
        border-radius: 8px;
        margin: 0 8px;
        font-size: 0.9rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .conversation-item:hover {
        background: rgba(255, 255, 255, 0.05);
      }

      .conversation-item.active {
        background: var(--primary-color, #03a9f4);
        color: white;
      }

      .main {
        display: flex;
        flex: 1;
        overflow: hidden;
      }

      .chat-area {
        flex: 1;
        display: flex;
        flex-direction: column;
        overflow: hidden;
      }

      .messages {
        flex: 1;
        overflow-y: auto;
        padding: 20px;
        display: flex;
        flex-direction: column;
        gap: 16px;
      }

      .message {
        max-width: 85%;
        padding: 12px 16px;
        border-radius: 12px;
        line-height: 1.5;
        white-space: pre-wrap;
        word-break: break-word;
      }

      .message.user {
        align-self: flex-end;
        background: var(--primary-color, #03a9f4);
        color: white;
      }

      .message.assistant {
        align-self: flex-start;
        background: var(--secondary-background-color, #2d2d2d);
        border: 1px solid var(--divider-color, #333);
      }

      .input-area {
        padding: 16px 20px;
        border-top: 1px solid var(--divider-color, #333);
        flex-shrink: 0;
      }

      .input-row {
        display: flex;
        gap: 12px;
        align-items: flex-end;
      }

      textarea {
        flex: 1;
        min-height: 48px;
        max-height: 150px;
        padding: 12px 16px;
        border-radius: 12px;
        border: 1px solid var(--divider-color, #333);
        background: var(--card-background-color, #2d2d2d);
        color: var(--primary-text-color, #e1e1e1);
        font-family: inherit;
        font-size: 1rem;
        resize: none;
      }

      textarea:focus {
        outline: none;
        border-color: var(--primary-color, #03a9f4);
      }

      .error {
        padding: 12px 20px;
        background: rgba(244, 67, 54, 0.2);
        color: #f44336;
        font-size: 0.9rem;
      }

      .memory-modal {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0, 0, 0, 0.7);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 1000;
      }

      .memory-content {
        background: var(--ha-card-background, #1c1c1c);
        border-radius: 12px;
        padding: 24px;
        max-width: 600px;
        width: 90%;
        max-height: 80vh;
        display: flex;
        flex-direction: column;
      }

      .memory-content h2 {
        margin: 0 0 16px;
        font-size: 1.25rem;
      }

      .memory-content textarea {
        flex: 1;
        min-height: 200px;
        margin-bottom: 16px;
      }

      .memory-actions {
        display: flex;
        gap: 12px;
        justify-content: flex-end;
      }

      .narrow .sidebar {
        display: none;
      }
    `;
  }

  render() {
    if (!this.hass) return html``;

    return html`
      <div class="container ${this.narrow ? "narrow" : ""}">
        <div class="header">
          <h1>🤖 AI Chat</h1>
          <div class="header-actions">
            <ha-icon-button
              .title=${"Memorie"}
              @click=${() => (this._showMemory = true)}
            >
              <ha-icon .icon=${"mdi:brain"}></ha-icon>
            </ha-icon-button>
            <ha-icon-button
              .title=${"Conversație nouă"}
              @click=${() => this._newConversation()}
            >
              <ha-icon .icon=${"mdi:plus"}></ha-icon>
            </ha-icon-button>
            <ha-icon-button
              .title=${"Șterge istoric"}
              @click=${this._clearHistory}
            >
              <ha-icon .icon=${"mdi:delete-outline"}></ha-icon>
            </ha-icon-button>
          </div>
        </div>

        <div class="main">
          ${!this.narrow
            ? html`
                <div class="sidebar">
                  <div class="sidebar-title">Conversații</div>
                  <div
                    class="conversation-item ${this._currentConversation === "default" ? "active" : ""}"
                    @click=${() => {
                      this._currentConversation = "default";
                      this._loadHistory();
                    }}
                  >
                    Conversație principală
                  </div>
                  ${this._conversations
                    .filter((c) => c.id !== "default")
                    .map(
                      (c) => html`
                        <div
                          class="conversation-item ${c.id === this._currentConversation ? "active" : ""}"
                          @click=${() => this._selectConversation(c.id)}
                        >
                          ${c.title}
                        </div>
                      `
                    )}
                </div>
              `
            : ""}

          <div class="chat-area">
            ${this._error
              ? html`<div class="error">${this._error}</div>`
              : ""}

            <div class="messages">
              ${this._messages.length === 0 && !this._loading
                ? html`
                    <div
                      style="text-align: center; color: var(--secondary-text-color); padding: 40px;"
                    >
                      Scrie un mesaj pentru a începe conversația. Istoricul este
                      salvat automat.
                    </div>
                  `
                : this._messages.map(
                    (m) => html`
                      <div class="message ${m.role}">${m.content}</div>
                    `
                  )}
              ${this._loading
                ? html`
                    <div class="message assistant">
                      Se procesează...
                    </div>
                  `
                : ""}
            </div>

            <div class="input-area">
              <div class="input-row">
                <textarea
                  .value=${this._input}
                  .disabled=${this._loading}
                  placeholder="Scrie mesajul tău..."
                  rows="1"
                  @input=${(e) => (this._input = e.target.value)}
                  @keydown=${this._handleKeyDown}
                ></textarea>
                <ha-icon-button
                  .title=${"Trimite"}
                  .disabled=${this._loading || !this._input.trim()}
                  @click=${this._sendMessage}
                >
                  <ha-icon .icon=${"mdi:send"}></ha-icon>
                </ha-icon-button>
              </div>
            </div>
          </div>
        </div>
      </div>

      ${this._showMemory
        ? html`
            <div class="memory-modal" @click=${() => (this._showMemory = false)}>
              <div
                class="memory-content"
                @click=${(e) => e.stopPropagation()}
              >
                <h2>Memorie (System Prompt)</h2>
                <p style="font-size: 0.9rem; color: var(--secondary-text-color); margin: 0 0 12px;">
                  Instrucțiunile pe care AI-ul le va urma. Poți personaliza
                  comportamentul asistentului.
                </p>
                <textarea
                  .value=${this._memory}
                  @input=${(e) => (this._memory = e.target.value)}
                  placeholder="Descrie cum vrei să se comporte asistentul..."
                ></textarea>
                <div class="memory-actions">
                  <ha-button @click=${() => (this._showMemory = false)}
                    >Anulare</ha-button
                  >
                  <ha-button @click=${this._saveMemory}>Salvează</ha-button>
                </div>
              </div>
            </div>
          `
        : ""}
    `;
  }
}

customElements.define("openai-chat-panel", OpenAIChatPanel);
