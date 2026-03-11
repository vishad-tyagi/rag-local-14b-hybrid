import React, { useEffect, useMemo, useRef, useState } from "react";

const STORAGE_KEY = "rag_chat_history_react_v4";

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
      if (line.startsWith("data:")) dataLines.push(line.slice(5));
    }
    const data = dataLines.join("\n");
    events.push({ event, data });
  }

  return { events, rest };
}

function Sidebar({ onNewChat, quickPrompts, onPickPrompt, mode, setMode }) {
  return (
    <div className="sidebar">
      <div className="sideTop">
        <div className="sideBrand">
          <div className="sideLogo">R</div>
          <div>
            <div className="sideTitle">Local RAG</div>
            <div className="sideSub">Flask + FAISS + MLX</div>
          </div>
        </div>

        <button className="sideBtn" onClick={onNewChat}>
          + New chat
        </button>
      </div>

      <div className="sideSection">
        <div className="sideSectionTitle">Mode</div>
        <div className="modeSwitch">
          <button
            className={`modeBtn ${mode === "rag" ? "active" : ""}`}
            onClick={() => setMode("rag")}
          >
            RAG
          </button>
          <button
            className={`modeBtn ${mode === "sql" ? "active" : ""}`}
            onClick={() => setMode("sql")}
          >
            SQL
          </button>
        </div>
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
          RAG mode uses <code>data/</code>. SQL mode uses <code>data.db</code>.
        </div>
      </div>
    </div>
  );
}

function SourcesPanel({ sources, mode }) {
  return (
    <div className="sourcesPanel">
      <div className="sourcesHeader">
        <div className="sourcesTitle">Context</div>
        <div className="sourcesMeta">
          {mode === "rag" ? "Retrieved chunks" : "SQL mode doesn't use document chunks"}
        </div>
      </div>

      <div className="sourcesBody">
        {mode !== "rag" ? (
          <div className="sourcesEmpty">Switch back to RAG mode to see retrieved sources here.</div>
        ) : sources?.length ? (
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
          <div className="sourcesEmpty">Ask a question in RAG mode to see retrieved context here.</div>
        )}
      </div>
    </div>
  );
}

function ResultTable({ columns, rows }) {
  if (!columns?.length) return null;

  return (
    <div className="sqlTableWrap">
      <table className="sqlTable">
        <thead>
          <tr>
            {columns.map((col) => (
              <th key={col}>{col}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, idx) => (
            <tr key={idx}>
              {columns.map((col) => (
                <td key={col}>{String(row[col] ?? "")}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Message({ message }) {
  const isUser = message.role === "user";

  return (
    <div className={`msgRow ${isUser ? "user" : "assistant"}`}>
      <div className="avatar">{isUser ? "You" : "Butler"}</div>
      <div className="msgBubble">
        <div className="msgText">{message.text}</div>

        {!isUser && message.mode === "sql" ? (
          <div className="sqlBlock">
            <details open>
              <summary>Generated SQL</summary>
              <pre className="sqlCode">{message.sql || "No SQL generated."}</pre>
            </details>

            <details open>
              <summary>Raw output</summary>
              <ResultTable columns={message.columns || []} rows={message.rows || []} />
            </details>
          </div>
        ) : null}
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
  const [mode, setMode] = useState("rag");

  const chatRef = useRef(null);
  const inputRef = useRef(null);
  const abortRef = useRef(null);

  const quickPrompts = useMemo(
    () =>
      mode === "rag"
        ? [
            { label: "Summarize docs", q: "Summarize the key points in the documents." },
            { label: "What is RAG?", q: "What is RAG and how does it work?" },
            { label: "Topics + sources", q: "List the main topics covered and where they appear." },
          ]
        : [
            { label: "Inventory value", q: "What is the total inventory value for each product?" },
            { label: "Low stock", q: "Which products have stock quantity less than 20?" },
            { label: "Average price", q: "What is the average price by category?" },
          ],
    [mode]
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

    const userMsg = { id: uid(), role: "user", text: q, mode };
    setMessages((prev) => clamp([...prev, userMsg]));
    setInput("");
    setStreaming(true);

    if (mode === "sql") {
      setSources([]);
      try {
        const res = await fetch("/api/sql/query", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question: q }),
        });

        const data = await res.json();

        const assistantMsg = {
          id: uid(),
          role: "assistant",
          mode: "sql",
          text: data.summary || "No summary returned.",
          sql: data.sql || "",
          columns: data.columns || [],
          rows: data.rows || [],
        };

        setMessages((prev) => clamp([...prev, assistantMsg]));
      } catch (e) {
        setMessages((prev) =>
          clamp([...prev, { id: uid(), role: "assistant", mode: "sql", text: `Error: ${String(e)}` }])
        );
      } finally {
        setStreaming(false);
        inputRef.current?.focus();
      }
      return;
    }

    // RAG mode (streaming)
    const assistantId = uid();
    const assistantMsg = { id: assistantId, role: "assistant", mode: "rag", text: "" };
    setMessages((prev) => clamp([...prev, assistantMsg]));
    setSources([]);

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
      <Sidebar
        onNewChat={newChat}
        quickPrompts={quickPrompts}
        onPickPrompt={pickPrompt}
        mode={mode}
        setMode={setMode}
      />

      <div className="main">
        <div className="topBar">
          <div className="topTitle">{mode === "rag" ? "RAG Chat" : "SQL Chat"}</div>
          <div className="topMeta">
            {mode === "rag" ? "Document Q&A" : "Text-to-SQL on data.db"}
          </div>
        </div>

        <div className="chat" ref={chatRef}>
          {messages.length === 0 ? (
            <div className="empty">
              <div className="emptyTitle">
                {mode === "rag" ? "Local RAG mode" : "Text-to-SQL mode"}
              </div>
              <div className="emptySub">
                {mode === "rag"
                  ? <>Ask questions about your files in <code>data/</code>.</>
                  : <>Ask questions about your SQLite database in <code>data.db</code>.</>}
              </div>
            </div>
          ) : (
            messages.map((m) => <Message key={m.id} message={m} />)
          )}

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
              placeholder={
                mode === "rag"
                  ? "Ask about your documents…"
                  : "Ask about your SQLite data…"
              }
              rows={1}
            />

            {streaming ? (
              <button className="iconBtn stop" onClick={stop} title="Stop generating">
                ■
              </button>
            ) : (
              <button className="iconBtn send" onClick={() => send()} title="Send">
                <svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true">
                  <path fill="currentColor" d="M2 21l21-9L2 3v7l15 2-15 2v7z" />
                </svg>
              </button>
            )}
          </div>
          <div className="composerHint">
            {mode === "rag"
              ? "RAG mode uses FAISS + local MLX model."
              : "SQL mode returns generated SQL, raw output, and a natural-language summary."}
          </div>
        </div>
      </div>

      <SourcesPanel sources={sources} mode={mode} />
    </div>
  );
}