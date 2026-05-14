import React, { useEffect, useMemo, useRef, useState } from "react";

const STORAGE_KEY = "rag_chat_history_react_v5";

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
          <div className="sideLogo">B</div>
          <div>
            <div className="sideTitle">Bank Butler</div>
            <div className="sideSub">RAG + SQL + Graph + AUTO</div>
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
          {/* --- NEW: Graph Mode Button --- */}
          <button
            className={`modeBtn ${mode === "graph" ? "active" : ""}`}
            onClick={() => setMode("graph")}
          >
            Graph
          </button>
          <button
            className={`modeBtn ${mode === "auto" ? "active" : ""}`}
            onClick={() => setMode("auto")}
          >
            AUTO
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
          {/* --- UPDATED: Added Neo4j tip --- */}
          RAG mode uses <code>data/</code>. SQL mode uses <code>data.db</code>. Graph mode uses <code>Neo4j</code>. AUTO routes for you.
        </div>
      </div>
    </div>
  );
}

function SourcesPanel({ sources, mode }) {
  const canShowSources = mode === "rag" || mode === "auto";

  return (
    <div className="sourcesPanel">
      <div className="sourcesHeader">
        <div className="sourcesTitle">Context</div>
        <div className="sourcesMeta">
          {/* --- UPDATED: Header text for Graph mode --- */}
          {mode === "rag"
            ? "Retrieved chunks"
            : mode === "sql"
            ? "SQL mode doesn't use document chunks"
            : mode === "graph"
            ? "Graph mode uses Neo4j relationships"
            : "AUTO shows chunks when RAG is selected"}
        </div>
      </div>

      <div className="sourcesBody">
        {!canShowSources ? (
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
          <div className="sourcesEmpty">Ask a RAG or AUTO question that needs documents to see retrieved context here.</div>
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
  const hasSql = !isUser && (message.mode === "sql" || (message.mode === "auto" && message.sql));
  const hasGraph = !isUser && (message.mode === "graph" || (message.mode === "auto" && message.cypher));
  const hasAutoRoute = !isUser && message.mode === "auto";
  const selectedModes = message.selected_modes?.length ? message.selected_modes.join(" + ").toUpperCase() : "Unknown";

  return (
    <div className={`msgRow ${isUser ? "user" : "assistant"}`}>
      <div className="avatar">{isUser ? "You" : "Butler"}</div>
      <div className="msgBubble">
        <div className="msgText">{message.text}</div>

        {hasAutoRoute ? (
          <div className="sqlBlock">
            <details open>
              <summary>AUTO route</summary>
              <pre className="sqlCode">
                {`Selected modes: ${selectedModes}
Router: ${message.route?.router || "unknown"}
Confidence: ${typeof message.route?.confidence === "number" ? message.route.confidence.toFixed(2) : "n/a"}
Reason: ${message.route?.reason || "No route reason returned."}`}
              </pre>
            </details>

            {message.partial_errors?.length ? (
              <details>
                <summary>Partial pipeline errors</summary>
                <pre className="sqlCode">{JSON.stringify(message.partial_errors, null, 2)}</pre>
              </details>
            ) : null}
          </div>
        ) : null}

        {hasSql ? (
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

        {/* --- NEW: Graph Mode Message Rendering --- */}
        {hasGraph ? (
          <div className="sqlBlock">
            <details open>
              <summary>Generated Cypher</summary>
              <pre className="sqlCode">{message.cypher || "No Cypher generated."}</pre>
            </details>

            <details>
              <summary>Graph Results (JSON)</summary>
              <pre className="sqlCode">
                {message.results && message.results.length > 0 
                  ? JSON.stringify(message.results, null, 2) 
                  : "No graph results found."}
              </pre>
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

  const quickPrompts = useMemo(() => {
    // --- UPDATED: Added Graph mode quick prompts ---
    if (mode === "rag") {
      return [
        { label: "Urgent AML policy", q: "What does Meridian Trust Bank consider an urgent AML alert?" },
        { label: "EDD guidance", q: "When should enhanced due diligence be performed?" },
        { label: "Branch context", q: "How should branch context be used during risk review?" },
      ];
    } else if (mode === "sql") {
      return [
        { label: "Urgent alerts", q: "Which customers have open AML alerts with scores above 80?" },
        { label: "Crypto total", q: "What is the total amount transferred to Northstar Crypto Exchange?" },
        { label: "Largest loan", q: "Which customer has the highest outstanding loan balance?" },
      ];
    } else if (mode === "graph") {
      return [
        { label: "Asha path", q: "How is Asha Rao connected to Northstar Crypto Exchange?" },
        { label: "Urgent review", q: "Which alerts require Urgent AML Review?" },
        { label: "Downtown links", q: "Which customers are connected to Downtown Branch?" },
      ];
    } else {
      return [
        { label: "Asha escalation", q: "Does Asha Rao need escalation? Use her alert score, account relationships, and the bank policy." },
        { label: "Risk + merchants", q: "Which high-risk customers have open alerts, and what merchants are their accounts connected to?" },
        { label: "Compare reviews", q: "Compare Asha Rao and Carlos Diaz from a risk review perspective using available records, relationships, and policy guidance." },
      ];
    }
  }, [mode]);

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

    if (mode === "auto") {
      setSources([]);
      try {
        const res = await fetch("/api/auto/query", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question: q }),
        });

        const data = await res.json();
        setSources(data.sources || []);

        const assistantMsg = {
          id: uid(),
          role: "assistant",
          mode: "auto",
          text: data.answer || "No answer returned.",
          selected_modes: data.selected_modes || [],
          route: data.route || {},
          sources: data.sources || [],
          sql: data.sql || "",
          columns: data.columns || [],
          rows: data.rows || [],
          cypher: data.cypher || "",
          results: data.results || [],
          partial_errors: data.partial_errors || [],
        };

        setMessages((prev) => clamp([...prev, assistantMsg]));
      } catch (e) {
        setMessages((prev) =>
          clamp([...prev, { id: uid(), role: "assistant", mode: "auto", text: `Error: ${String(e)}` }])
        );
      } finally {
        setStreaming(false);
        inputRef.current?.focus();
      }
      return;
    }

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

    // --- NEW: Graph Mode Submission Logic ---
    if (mode === "graph") {
      setSources([]);
      try {
        const res = await fetch("/api/graph/query", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question: q }),
        });

        const data = await res.json();

        const assistantMsg = {
          id: uid(),
          role: "assistant",
          mode: "graph",
          text: data.summary || "No summary returned.",
          cypher: data.cypher || "",
          results: data.results || [],
        };

        setMessages((prev) => clamp([...prev, assistantMsg]));
      } catch (e) {
        setMessages((prev) =>
          clamp([...prev, { id: uid(), role: "assistant", mode: "graph", text: `Error: ${String(e)}` }])
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

  const modeTitle = {
    rag: "RAG Chat",
    sql: "SQL Chat",
    graph: "Graph Chat",
    auto: "AUTO Chat",
  }[mode];
  const modeMeta = {
    rag: "Document policy Q&A",
    sql: "Text-to-SQL on bank records",
    graph: "Text-to-Cypher on banking relationships",
    auto: "Smart routing across policy, records, and relationships",
  }[mode];
  const emptyTitle = {
    rag: "Policy document mode",
    sql: "Banking SQL mode",
    graph: "Banking relationship graph mode",
    auto: "AUTO routing mode",
  }[mode];
  const placeholder = {
    rag: "Ask about bank policy documents...",
    sql: "Ask about structured bank records...",
    graph: "Ask about banking relationships...",
    auto: "Ask anything; AUTO will choose the right pipeline...",
  }[mode];
  const composerHint = {
    rag: "RAG mode answers from policy text in data/.",
    sql: "SQL mode returns generated SQL, raw output, and a natural-language summary.",
    graph: "Graph mode returns generated Cypher queries, raw JSON relationships, and a summary.",
    auto: "AUTO mode routes to RAG, SQL, Graph, or a true hybrid only when needed.",
  }[mode];

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
          <div className="topTitle">
            {/* --- UPDATED: Dynamic Title --- */}
            {modeTitle}
          </div>
          <div className="topMeta">
            {/* --- UPDATED: Dynamic Subtitle --- */}
            {modeMeta}
          </div>
        </div>

        <div className="chat" ref={chatRef}>
          {messages.length === 0 ? (
            <div className="empty">
              <div className="emptyTitle">
                {/* --- UPDATED: Dynamic Empty State Title --- */}
                {emptyTitle}
              </div>
              <div className="emptySub">
                {/* --- UPDATED: Dynamic Empty State Text --- */}
                {mode === "rag"
                  ? <>Ask questions about banking policy files in <code>data/</code>.</>
                  : mode === "sql" 
                  ? <>Ask questions about banking tables in <code>data.db</code>.</>
                  : mode === "graph"
                  ? <>Discover customer, account, merchant, alert, and policy relationships in <code>Neo4j</code>.</>
                  : <>Let AUTO choose policy, records, relationships, or a focused hybrid.</>}
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
                // --- UPDATED: Dynamic Placeholder ---
                placeholder
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
            {/* --- UPDATED: Dynamic Hint Text --- */}
            {composerHint}
          </div>
        </div>
      </div>

      <SourcesPanel sources={sources} mode={mode} />
    </div>
  );
}

































// import React, { useEffect, useMemo, useRef, useState } from "react";

// const STORAGE_KEY = "rag_chat_history_react_v4";

// function loadHistory() {
//   try {
//     const raw = localStorage.getItem(STORAGE_KEY);
//     return raw ? JSON.parse(raw) : [];
//   } catch {
//     return [];
//   }
// }

// function saveHistory(items) {
//   localStorage.setItem(STORAGE_KEY, JSON.stringify(items));
// }

// function uid() {
//   return Math.random().toString(16).slice(2) + Date.now().toString(16);
// }

// function clamp(items, max = 80) {
//   return items.length > max ? items.slice(items.length - max) : items;
// }

// function parseSSEChunk(buffer) {
//   const events = [];
//   const parts = buffer.split("\n\n");
//   const rest = parts.pop() || "";

//   for (const p of parts) {
//     const lines = p.split("\n");
//     let event = "message";
//     let dataLines = [];
//     for (const line of lines) {
//       if (line.startsWith("event:")) event = line.slice(6).trim();
//       if (line.startsWith("data:")) dataLines.push(line.slice(5));
//     }
//     const data = dataLines.join("\n");
//     events.push({ event, data });
//   }

//   return { events, rest };
// }

// function Sidebar({ onNewChat, quickPrompts, onPickPrompt, mode, setMode }) {
//   return (
//     <div className="sidebar">
//       <div className="sideTop">
//         <div className="sideBrand">
//           <div className="sideLogo">R</div>
//           <div>
//             <div className="sideTitle">Local RAG</div>
//             <div className="sideSub">Flask + FAISS + MLX</div>
//           </div>
//         </div>

//         <button className="sideBtn" onClick={onNewChat}>
//           + New chat
//         </button>
//       </div>

//       <div className="sideSection">
//         <div className="sideSectionTitle">Mode</div>
//         <div className="modeSwitch">
//           <button
//             className={`modeBtn ${mode === "rag" ? "active" : ""}`}
//             onClick={() => setMode("rag")}
//           >
//             RAG
//           </button>
//           <button
//             className={`modeBtn ${mode === "sql" ? "active" : ""}`}
//             onClick={() => setMode("sql")}
//           >
//             SQL
//           </button>
//         </div>
//       </div>

//       <div className="sideSection">
//         <div className="sideSectionTitle">Quick prompts</div>
//         <div className="sideChips">
//           {quickPrompts.map((p) => (
//             <button key={p.label} className="chip" onClick={() => onPickPrompt(p.q)}>
//               {p.label}
//             </button>
//           ))}
//         </div>
//       </div>

//       <div className="sideSection">
//         <div className="sideSectionTitle">Tips</div>
//         <div className="sideTip">
//           RAG mode uses <code>data/</code>. SQL mode uses <code>data.db</code>.
//         </div>
//       </div>
//     </div>
//   );
// }

// function SourcesPanel({ sources, mode }) {
//   return (
//     <div className="sourcesPanel">
//       <div className="sourcesHeader">
//         <div className="sourcesTitle">Context</div>
//         <div className="sourcesMeta">
//           {mode === "rag" ? "Retrieved chunks" : "SQL mode doesn't use document chunks"}
//         </div>
//       </div>

//       <div className="sourcesBody">
//         {mode !== "rag" ? (
//           <div className="sourcesEmpty">Switch back to RAG mode to see retrieved sources here.</div>
//         ) : sources?.length ? (
//           <div className="sourcesList">
//             {sources.map((s, i) => (
//               <details key={i} open={i === 0}>
//                 <summary>
//                   <span className="srcLeft">
//                     <span className="badge">{i + 1}</span>
//                     <span className="srcPath">{s.source}</span>
//                   </span>
//                   <span className="srcView">View</span>
//                 </summary>
//                 <div className="snippet">{s.snippet}</div>
//               </details>
//             ))}
//           </div>
//         ) : (
//           <div className="sourcesEmpty">Ask a question in RAG mode to see retrieved context here.</div>
//         )}
//       </div>
//     </div>
//   );
// }

// function ResultTable({ columns, rows }) {
//   if (!columns?.length) return null;

//   return (
//     <div className="sqlTableWrap">
//       <table className="sqlTable">
//         <thead>
//           <tr>
//             {columns.map((col) => (
//               <th key={col}>{col}</th>
//             ))}
//           </tr>
//         </thead>
//         <tbody>
//           {rows.map((row, idx) => (
//             <tr key={idx}>
//               {columns.map((col) => (
//                 <td key={col}>{String(row[col] ?? "")}</td>
//               ))}
//             </tr>
//           ))}
//         </tbody>
//       </table>
//     </div>
//   );
// }

// function Message({ message }) {
//   const isUser = message.role === "user";

//   return (
//     <div className={`msgRow ${isUser ? "user" : "assistant"}`}>
//       <div className="avatar">{isUser ? "You" : "Butler"}</div>
//       <div className="msgBubble">
//         <div className="msgText">{message.text}</div>

//         {!isUser && message.mode === "sql" ? (
//           <div className="sqlBlock">
//             <details open>
//               <summary>Generated SQL</summary>
//               <pre className="sqlCode">{message.sql || "No SQL generated."}</pre>
//             </details>

//             <details open>
//               <summary>Raw output</summary>
//               <ResultTable columns={message.columns || []} rows={message.rows || []} />
//             </details>
//           </div>
//         ) : null}
//       </div>
//     </div>
//   );
// }

// function TypingBubble() {
//   return (
//     <div className="msgRow assistant">
//       <div className="avatar">Butler</div>
//       <div className="msgBubble">
//         <div className="typing">
//           <span className="dot" />
//           <span className="dot" />
//           <span className="dot" />
//         </div>
//       </div>
//     </div>
//   );
// }

// export default function App() {
//   const [messages, setMessages] = useState(() => loadHistory());
//   const [input, setInput] = useState("");
//   const [sources, setSources] = useState([]);
//   const [streaming, setStreaming] = useState(false);
//   const [mode, setMode] = useState("rag");

//   const chatRef = useRef(null);
//   const inputRef = useRef(null);
//   const abortRef = useRef(null);

//   const quickPrompts = useMemo(
//     () =>
//       mode === "rag"
//         ? [
//             { label: "Summarize docs", q: "Summarize the key points in the documents." },
//             { label: "What is RAG?", q: "What is RAG and how does it work?" },
//             { label: "Topics + sources", q: "List the main topics covered and where they appear." },
//           ]
//         : [
//             { label: "Inventory value", q: "What is the total inventory value for each product?" },
//             { label: "Low stock", q: "Which products have stock quantity less than 20?" },
//             { label: "Average price", q: "What is the average price by category?" },
//           ],
//     [mode]
//   );

//   useEffect(() => saveHistory(messages), [messages]);

//   useEffect(() => {
//     if (chatRef.current) chatRef.current.scrollTop = chatRef.current.scrollHeight;
//   }, [messages, streaming]);

//   function newChat() {
//     if (abortRef.current) abortRef.current.abort();
//     abortRef.current = null;
//     setStreaming(false);
//     setMessages([]);
//     setSources([]);
//     setInput("");
//     localStorage.removeItem(STORAGE_KEY);
//     inputRef.current?.focus();
//   }

//   function stop() {
//     if (abortRef.current) abortRef.current.abort();
//     abortRef.current = null;
//     setStreaming(false);
//   }

//   async function send(questionMaybe) {
//     const q = (questionMaybe ?? input).trim();
//     if (!q || streaming) return;

//     const userMsg = { id: uid(), role: "user", text: q, mode };
//     setMessages((prev) => clamp([...prev, userMsg]));
//     setInput("");
//     setStreaming(true);

//     if (mode === "sql") {
//       setSources([]);
//       try {
//         const res = await fetch("/api/sql/query", {
//           method: "POST",
//           headers: { "Content-Type": "application/json" },
//           body: JSON.stringify({ question: q }),
//         });

//         const data = await res.json();

//         const assistantMsg = {
//           id: uid(),
//           role: "assistant",
//           mode: "sql",
//           text: data.summary || "No summary returned.",
//           sql: data.sql || "",
//           columns: data.columns || [],
//           rows: data.rows || [],
//         };

//         setMessages((prev) => clamp([...prev, assistantMsg]));
//       } catch (e) {
//         setMessages((prev) =>
//           clamp([...prev, { id: uid(), role: "assistant", mode: "sql", text: `Error: ${String(e)}` }])
//         );
//       } finally {
//         setStreaming(false);
//         inputRef.current?.focus();
//       }
//       return;
//     }

//     // RAG mode (streaming)
//     const assistantId = uid();
//     const assistantMsg = { id: assistantId, role: "assistant", mode: "rag", text: "" };
//     setMessages((prev) => clamp([...prev, assistantMsg]));
//     setSources([]);

//     const controller = new AbortController();
//     abortRef.current = controller;

//     try {
//       const res = await fetch("/api/ask/stream", {
//         method: "POST",
//         headers: { "Content-Type": "application/json" },
//         body: JSON.stringify({ question: q }),
//         signal: controller.signal,
//       });

//       if (!res.ok || !res.body) {
//         setMessages((prev) =>
//           prev.map((m) => (m.id === assistantId ? { ...m, text: `Request failed (${res.status})` } : m))
//         );
//         setStreaming(false);
//         abortRef.current = null;
//         return;
//       }

//       const reader = res.body.getReader();
//       const decoder = new TextDecoder("utf-8");
//       let buf = "";

//       while (true) {
//         const { value, done } = await reader.read();
//         if (done) break;

//         buf += decoder.decode(value, { stream: true });
//         const parsed = parseSSEChunk(buf);
//         buf = parsed.rest;

//         for (const ev of parsed.events) {
//           if (ev.event === "sources") {
//             try {
//               setSources(JSON.parse(ev.data));
//             } catch {}
//           } else if (ev.event === "token") {
//             let token = "";
//             try {
//               token = JSON.parse(ev.data);
//             } catch {
//               token = ev.data;
//             }
//             setMessages((prev) =>
//               prev.map((m) => (m.id === assistantId ? { ...m, text: (m.text || "") + token } : m))
//             );
//           } else if (ev.event === "error") {
//             let err = ev.data;
//             try { err = JSON.parse(ev.data); } catch {}
//             setMessages((prev) =>
//               prev.map((m) => (m.id === assistantId ? { ...m, text: `Error: ${err}` } : m))
//             );
//             setStreaming(false);
//             abortRef.current = null;
//             return;
//           } else if (ev.event === "done") {
//             setStreaming(false);
//             abortRef.current = null;
//             return;
//           }
//         }
//       }

//       setStreaming(false);
//       abortRef.current = null;
//     } catch (e) {
//       const msg = controller.signal.aborted ? "Stopped." : `Error: ${String(e)}`;
//       setMessages((prev) =>
//         prev.map((m) => (m.id === assistantId ? { ...m, text: msg } : m))
//       );
//       setStreaming(false);
//       abortRef.current = null;
//     } finally {
//       inputRef.current?.focus();
//     }
//   }

//   function onKeyDown(e) {
//     if (e.key === "Enter" && !e.shiftKey) {
//       e.preventDefault();
//       send();
//     }
//   }

//   function pickPrompt(q) {
//     setInput(q);
//     inputRef.current?.focus();
//   }

//   return (
//     <div className="appShell">
//       <Sidebar
//         onNewChat={newChat}
//         quickPrompts={quickPrompts}
//         onPickPrompt={pickPrompt}
//         mode={mode}
//         setMode={setMode}
//       />

//       <div className="main">
//         <div className="topBar">
//           <div className="topTitle">{mode === "rag" ? "RAG Chat" : "SQL Chat"}</div>
//           <div className="topMeta">
//             {mode === "rag" ? "Document Q&A" : "Text-to-SQL on data.db"}
//           </div>
//         </div>

//         <div className="chat" ref={chatRef}>
//           {messages.length === 0 ? (
//             <div className="empty">
//               <div className="emptyTitle">
//                 {mode === "rag" ? "Local RAG mode" : "Text-to-SQL mode"}
//               </div>
//               <div className="emptySub">
//                 {mode === "rag"
//                   ? <>Ask questions about your files in <code>data/</code>.</>
//                   : <>Ask questions about your SQLite database in <code>data.db</code>.</>}
//               </div>
//             </div>
//           ) : (
//             messages.map((m) => <Message key={m.id} message={m} />)
//           )}

//           {streaming ? <TypingBubble /> : null}
//         </div>

//         <div className="composer">
//           <div className="composerInner">
//             <textarea
//               ref={inputRef}
//               className="composerInput"
//               value={input}
//               onChange={(e) => setInput(e.target.value)}
//               onKeyDown={onKeyDown}
//               placeholder={
//                 mode === "rag"
//                   ? "Ask about your documents…"
//                   : "Ask about your SQLite data…"
//               }
//               rows={1}
//             />

//             {streaming ? (
//               <button className="iconBtn stop" onClick={stop} title="Stop generating">
//                 ■
//               </button>
//             ) : (
//               <button className="iconBtn send" onClick={() => send()} title="Send">
//                 <svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true">
//                   <path fill="currentColor" d="M2 21l21-9L2 3v7l15 2-15 2v7z" />
//                 </svg>
//               </button>
//             )}
//           </div>
//           <div className="composerHint">
//             {mode === "rag"
//               ? "RAG mode uses FAISS + local MLX model."
//               : "SQL mode returns generated SQL, raw output, and a natural-language summary."}
//           </div>
//         </div>
//       </div>

//       <SourcesPanel sources={sources} mode={mode} />
//     </div>
//   );
// }
