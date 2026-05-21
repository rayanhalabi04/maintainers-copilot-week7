import React, { FormEvent, useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

type WidgetConfig = {
  widget_id: string;
  theme: {
    primaryColor: string;
    position: "bottom-right" | "bottom-left";
  };
  greeting: string;
  enabled_tools: string[];
};

type ToolCall = {
  tool_name: string;
  status: string;
  summary?: string;
  error?: string;
};

type ChatTurn = {
  role: "assistant" | "user";
  text: string;
  toolCalls?: ToolCall[];
};

const DEFAULT_CONFIG: WidgetConfig = {
  widget_id: "demo-widget",
  theme: {
    primaryColor: "#2563eb",
    position: "bottom-right"
  },
  greeting: "Hi, I'm the Maintainer's Copilot.",
  enabled_tools: ["classify", "ner", "summarize", "rag"]
};

function App() {
  const params = useMemo(() => new URLSearchParams(window.location.search), []);
  const widgetId = params.get("widget_id") || "demo-widget";
  const modelServerUrl = (params.get("model_server_url") || "http://localhost:8001").replace(/\/$/, "");

  const [config, setConfig] = useState<WidgetConfig>(DEFAULT_CONFIG);
  const [open, setOpen] = useState(false);
  const [email, setEmail] = useState("user@example.com");
  const [password, setPassword] = useState("");
  const [token, setToken] = useState("");
  const [role, setRole] = useState("");
  const [message, setMessage] = useState("");
  const [issueTitle, setIssueTitle] = useState("");
  const [issueBody, setIssueBody] = useState("");
  const [turns, setTurns] = useState<ChatTurn[]>([
    { role: "assistant", text: DEFAULT_CONFIG.greeting }
  ]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    fetch(`${modelServerUrl}/widget/config/${encodeURIComponent(widgetId)}`)
      .then((response) => response.ok ? response.json() : DEFAULT_CONFIG)
      .then((payload) => {
        setConfig(payload);
        setTurns([{ role: "assistant", text: payload.greeting || DEFAULT_CONFIG.greeting }]);
      })
      .catch(() => {
        setConfig(DEFAULT_CONFIG);
      });
  }, [modelServerUrl, widgetId]);

  useEffect(() => {
    window.parent.postMessage({
      type: "maintainers-copilot:resize",
      width: open ? 380 : 88,
      height: open ? 620 : 88
    }, "*");
  }, [open]);

  async function login(event: FormEvent) {
    event.preventDefault();
    setError("");
    setLoading(true);
    try {
      const payload = await apiPost(modelServerUrl, "/auth/login", {
        email,
        password
      });
      setToken(payload.access_token);
      setRole(payload.role);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Login failed.");
    } finally {
      setLoading(false);
    }
  }

  async function sendMessage(event: FormEvent) {
    event.preventDefault();
    if (!message.trim()) {
      return;
    }
    setError("");
    setLoading(true);
    setTurns((existing) => [...existing, { role: "user", text: message }]);
    try {
      const payload = await apiPost(modelServerUrl, "/chat", {
        message,
        issue_title: issueTitle || null,
        issue_body: issueBody || null,
        use_rag: true,
        top_k: 5
      }, token);
      setTurns((existing) => [
        ...existing,
        {
          role: "assistant",
          text: payload.answer,
          toolCalls: payload.tool_calls || []
        }
      ]);
      setMessage("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Chat request failed.");
    } finally {
      setLoading(false);
    }
  }

  if (!open) {
    return (
      <button
        className="copilot-bubble"
        style={{ background: config.theme.primaryColor }}
        onClick={() => setOpen(true)}
        aria-label="Open Maintainer's Copilot"
      >
        MC
      </button>
    );
  }

  return (
    <section className="copilot-panel">
      <header style={{ background: config.theme.primaryColor }}>
        <div>
          <strong>Maintainer's Copilot</strong>
          <span>{token ? `${email} (${role})` : "Login required"}</span>
        </div>
        <button onClick={() => setOpen(false)} aria-label="Close">×</button>
      </header>

      <main>
        {turns.map((turn, index) => (
          <article className={`turn ${turn.role}`} key={`${turn.role}-${index}`}>
            <p>{turn.text}</p>
            {turn.toolCalls && turn.toolCalls.length > 0 && (
              <details>
                <summary>Tool calls</summary>
                <ul>
                  {turn.toolCalls.map((tool, toolIndex) => (
                    <li key={`${tool.tool_name}-${toolIndex}`}>
                      <strong>{tool.tool_name}</strong>: {tool.status}
                      {tool.summary ? <span> - {tool.summary}</span> : null}
                      {tool.error ? <span> - {tool.error}</span> : null}
                    </li>
                  ))}
                </ul>
              </details>
            )}
          </article>
        ))}
      </main>

      {error && <div className="error">{error}</div>}

      {!token ? (
        <form className="login-form" onSubmit={login}>
          <input value={email} onChange={(event) => setEmail(event.target.value)} placeholder="email" />
          <input value={password} onChange={(event) => setPassword(event.target.value)} placeholder="password" type="password" />
          <button style={{ background: config.theme.primaryColor }} disabled={loading}>
            {loading ? "Logging in..." : "Login"}
          </button>
          <small>Demo: user@example.com / user123 or admin@example.com / admin123</small>
        </form>
      ) : (
        <form className="chat-form" onSubmit={sendMessage}>
          <input value={issueTitle} onChange={(event) => setIssueTitle(event.target.value)} placeholder="Optional issue title" />
          <textarea value={issueBody} onChange={(event) => setIssueBody(event.target.value)} placeholder="Optional issue body" />
          <div>
            <input value={message} onChange={(event) => setMessage(event.target.value)} placeholder="Ask for triage..." />
            <button style={{ background: config.theme.primaryColor }} disabled={loading}>
              {loading ? "..." : "Send"}
            </button>
          </div>
        </form>
      )}
    </section>
  );
}

async function apiPost(baseUrl: string, path: string, body: unknown, token?: string) {
  const response = await fetch(`${baseUrl}${path}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {})
    },
    body: JSON.stringify(body)
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || `Request failed with ${response.status}`);
  }
  return payload;
}

createRoot(document.getElementById("root")!).render(<App />);
