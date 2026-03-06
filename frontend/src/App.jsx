import React, { useEffect, useMemo, useRef, useState } from "react";

const STORAGE_KEY = "rag_chat_history_react_v3";

function loadHistory() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function saveHistory(items) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(items));
}

function uid() {
  return Math.random().toString(16).slice(2) + Date.now().toString(16);
}

function clamp(items, max = 80) {
  return items.length > max ? items.slice(items.length - max) : items;
}

// Parse SSE stream (event + data)
function parseSSEChunk(buffer) {
  const events = [];
  const parts = buffer.split("\n\n");
  const rest = parts.pop() || "";

  for (const p of parts) {
    const lines = p.split("\n");
    let event = "message";
    let dataLines = [];
    for (const line of lines) {
      if (line.startsWith("event:")) event = line.slice(6).trim();
      if (line.startsWith("data:")) dataLines.push(line.slice(5)); // DO NOT trim -> preserves spaces
    }
    const data = dataLines.join("\n");
    events.push({ event, data });
  }

  return { events, rest };
}

function Sidebar({ onNewChat, quickPrompts, onPickPrompt }) {
  return (
    <div className="sidebar">
      <div className="sideTop">
        <div className="sideBrand">
          <div className="sideLogo">R</div>
          <div>
            <div className="sideTitle">Local RAG</div>
            <div className="sideSub">Flask + FAISS + Ollama</div>
          </div>
        </div>

        <button className="sideBtn" onClick={onNewChat}>
          + New chat
        </button>
      </div>

      <div className="sideSection">
        <div className="sideSectionTitle">Quick prompts</div>
        <div className="sideChips">
          {quickPrompts.map((p) => (
            <button key={p.label} className="chip" onClick={() => onPickPrompt(p.q)}>
              {p.label}
            </button>
          ))}
        </div>
      </div>

      <div className="sideSection">
        <div className="sideSectionTitle">Tips</div>
        <div className="sideTip">
          Add docs to <code>data/</code> → run <code>python ingest.py</code>
        </div>
      </div>
    </div>
  );
}

function SourcesPanel({ sources }) {
  return (
    <div className="sourcesPanel">
      <div className="sourcesHeader">
        <div className="sourcesTitle">Sources</div>
        <div className="sourcesMeta">Top retrieved chunks</div>
      </div>

      <div className="sourcesBody">
        {sources?.length ? (
          <div className="sourcesList">
            {sources.map((s, i) => (
              <details key={i} open={i === 0}>
                <summary>
                  <span className="srcLeft">
                    <span className="badge">{i + 1}</span>
                    <span className="srcPath">{s.source}</span>
                  </span>
                  <span className="srcView">View</span>
                </summary>
                <div className="snippet">{s.snippet}</div>
              </details>
            ))}
          </div>
        ) : (
          <div className="sourcesEmpty">Ask a question to see retrieved context here.</div>
        )}
      </div>
    </div>
  );
}

function Message({ role, text }) {
  const isUser = role === "user";
  return (
    <div className={`msgRow ${isUser ? "user" : "assistant"}`}>
      <div className="avatar">{isUser ? "You" : "Butler"}</div>
      <div className="msgBubble">
        <div className="msgText">{text}</div>
      </div>
    </div>
  );
}

function TypingBubble() {
  return (
    <div className="msgRow assistant">
      <div className="avatar">Butler</div>
      <div className="msgBubble">
        <div className="typing">
          <span className="dot" />
          <span className="dot" />
          <span className="dot" />
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const [messages, setMessages] = useState(() => loadHistory());
  const [input, setInput] = useState("");
  const [sources, setSources] = useState([]);
  const [streaming, setStreaming] = useState(false);

  const chatRef = useRef(null);
  const inputRef = useRef(null);
  const abortRef = useRef(null);

  const quickPrompts = useMemo(
    () => [
      { label: "Summarize docs", q: "Summarize the key points in the documents." },
      { label: "What is RAG?", q: "What is RAG and how does it work?" },
      { label: "Topics + sources", q: "List the main topics covered and where they appear." },
      { label: "Study plan", q: "Give me a bullet-point study plan from these docs." },
    ],
    []
  );

  useEffect(() => saveHistory(messages), [messages]);

  useEffect(() => {
    if (chatRef.current) chatRef.current.scrollTop = chatRef.current.scrollHeight;
  }, [messages, streaming]);

  function newChat() {
    if (abortRef.current) abortRef.current.abort();
    abortRef.current = null;
    setStreaming(false);
    setMessages([]);
    setSources([]);
    setInput("");
    localStorage.removeItem(STORAGE_KEY);
    inputRef.current?.focus();
  }

  function stop() {
    if (abortRef.current) abortRef.current.abort();
    abortRef.current = null;
    setStreaming(false);
  }

  async function send(questionMaybe) {
    const q = (questionMaybe ?? input).trim();
    if (!q || streaming) return;

    const userMsg = { id: uid(), role: "user", text: q };
    const assistantId = uid();
    const assistantMsg = { id: assistantId, role: "assistant", text: "" };

    setMessages((prev) => clamp([...prev, userMsg, assistantMsg]));
    setInput("");
    setSources([]);
    setStreaming(true);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const res = await fetch("/api/ask/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q }),
        signal: controller.signal,
      });

      if (!res.ok || !res.body) {
        setMessages((prev) =>
          prev.map((m) => (m.id === assistantId ? { ...m, text: `Request failed (${res.status})` } : m))
        );
        setStreaming(false);
        abortRef.current = null;
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buf = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buf += decoder.decode(value, { stream: true });
        const parsed = parseSSEChunk(buf);
        buf = parsed.rest;

        for (const ev of parsed.events) {
          if (ev.event === "sources") {
            try {
              setSources(JSON.parse(ev.data));
            } catch {}
          } else if (ev.event === "token") {
            // token data is JSON string -> preserves spaces/newlines perfectly
            let token = "";
            try {
              token = JSON.parse(ev.data);
            } catch {
              token = ev.data;
            }
            setMessages((prev) =>
              prev.map((m) => (m.id === assistantId ? { ...m, text: (m.text || "") + token } : m))
            );
          } else if (ev.event === "error") {
            let err = ev.data;
            try { err = JSON.parse(ev.data); } catch {}
            setMessages((prev) =>
              prev.map((m) => (m.id === assistantId ? { ...m, text: `Error: ${err}` } : m))
            );
            setStreaming(false);
            abortRef.current = null;
            return;
          } else if (ev.event === "done") {
            setStreaming(false);
            abortRef.current = null;
            return;
          }
        }
      }

      setStreaming(false);
      abortRef.current = null;
    } catch (e) {
      const msg = controller.signal.aborted ? "Stopped." : `Error: ${String(e)}`;
      setMessages((prev) =>
        prev.map((m) => (m.id === assistantId ? { ...m, text: msg } : m))
      );
      setStreaming(false);
      abortRef.current = null;
    } finally {
      inputRef.current?.focus();
    }
  }

  function onKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  }

  function pickPrompt(q) {
    setInput(q);
    inputRef.current?.focus();
  }

  return (
    <div className="appShell">
      <Sidebar onNewChat={newChat} quickPrompts={quickPrompts} onPickPrompt={pickPrompt} />

      <div className="main">
        <div className="topBar">
          <div className="topTitle">Chat</div>
          <div className="topMeta">Model: llama3.1 · RAG: FAISS</div>
        </div>

        <div className="chat" ref={chatRef}>
          {messages.length === 0 ? (
            <div className="empty">
              <div className="emptyTitle">ChatGPT-style local RAG</div>
              <div className="emptySub">
                Add docs to <code>data/</code>, run <code>python ingest.py</code>, then ask.
              </div>
            </div>
          ) : (
            messages.map((m) => <Message key={m.id} role={m.role} text={m.text} />)
          )}

          {/* (4) processing animation */}
          {streaming ? <TypingBubble /> : null}
        </div>

        <div className="composer">
          <div className="composerInner">
            <textarea
              ref={inputRef}
              className="composerInput"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={onKeyDown}
              placeholder="Message… (Enter to send, Shift+Enter for newline)"
              rows={1}
            />

            {/* (3) icon button like ChatGPT */}
            {streaming ? (
              <button className="iconBtn stop" onClick={stop} title="Stop generating">
                ■
              </button>
            ) : (
              <button className="iconBtn send" onClick={() => send()} title="Send">
                <svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true">
                  <path
                    fill="currentColor"
                    d="M2 21l21-9L2 3v7l15 2-15 2v7z"
                  />
                </svg>
              </button>
            )}
          </div>
          <div className="composerHint">Answers are grounded in retrieved context · streaming via Ollama</div>
        </div>
      </div>

      <SourcesPanel sources={sources} />
    </div>
  );
}