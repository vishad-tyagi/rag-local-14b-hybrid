import React, { useEffect, useMemo, useRef, useState } from "react";

const LEGACY_STORAGE_KEY = "rag_chat_history_react_v5";
const STORAGE_KEY = "rag_chat_sessions_react_v1";
const DEFAULT_MODE = "rag";
const EMPTY_ARRAY = [];

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

  for (const part of parts) {
    const lines = part.split("\n");
    let event = "message";
    const dataLines = [];

    for (const line of lines) {
      if (line.startsWith("event:")) event = line.slice(6).trim();
      if (line.startsWith("data:")) dataLines.push(line.slice(5));
    }

    events.push({ event, data: dataLines.join("\n") });
  }

  return { events, rest };
}

function parseJsonOr(value, fallback) {
  try {
    return JSON.parse(value);
  } catch {
    return fallback;
  }
}

function summarizeTitle(text) {
  const normalized = text.replace(/\s+/g, " ").trim();
  if (!normalized) return "New chat";
  return normalized.length > 40 ? `${normalized.slice(0, 39).trimEnd()}...` : normalized;
}

function getChatPreview(chat) {
  const lastMessage = chat.messages[chat.messages.length - 1];
  if (!lastMessage?.text) return "No messages yet";
  const normalized = lastMessage.text.replace(/\s+/g, " ").trim();
  return normalized.length > 64 ? `${normalized.slice(0, 63).trimEnd()}...` : normalized;
}

function createChat(overrides = {}) {
  const now = Date.now();
  return {
    id: uid(),
    title: "New chat",
    customTitle: false,
    mode: DEFAULT_MODE,
    messages: [],
    sources: [],
    isPinned: false,
    pinnedAt: null,
    createdAt: now,
    updatedAt: now,
    ...overrides,
  };
}

function normalizeChat(chat) {
  return createChat({
    ...chat,
    messages: Array.isArray(chat?.messages) ? chat.messages : [],
    sources: Array.isArray(chat?.sources) ? chat.sources : [],
    mode: chat?.mode || DEFAULT_MODE,
    title: chat?.title || "New chat",
    customTitle: Boolean(chat?.customTitle),
    isPinned: Boolean(chat?.isPinned),
    pinnedAt: chat?.pinnedAt ?? null,
    createdAt: chat?.createdAt ?? Date.now(),
    updatedAt: chat?.updatedAt ?? Date.now(),
  });
}

function loadChatState() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      const chats = Array.isArray(parsed?.chats) ? parsed.chats.map(normalizeChat) : [];
      if (chats.length > 0) {
        const activeChatId = chats.some((chat) => chat.id === parsed?.activeChatId)
          ? parsed.activeChatId
          : chats[0].id;
        return { chats, activeChatId };
      }
    }
  } catch {
    // Ignore invalid saved chat state and fall back to migration/defaults.
  }

  try {
    const legacyRaw = localStorage.getItem(LEGACY_STORAGE_KEY);
    const legacyMessages = legacyRaw ? JSON.parse(legacyRaw) : [];
    if (Array.isArray(legacyMessages) && legacyMessages.length > 0) {
      const firstUserMessage = legacyMessages.find((message) => message.role === "user");
      const migratedChat = createChat({
        title: summarizeTitle(firstUserMessage?.text || "Imported chat"),
        messages: legacyMessages,
        updatedAt: Date.now(),
      });
      return { chats: [migratedChat], activeChatId: migratedChat.id };
    }
  } catch {
    // Ignore malformed legacy history and start fresh.
  }

  const initialChat = createChat();
  return { chats: [initialChat], activeChatId: initialChat.id };
}

function saveChatState(chats, activeChatId) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify({ chats, activeChatId }));
}

function sortPinnedChats(chats) {
  return [...chats].sort(
    (a, b) => (b.pinnedAt ?? 0) - (a.pinnedAt ?? 0) || b.updatedAt - a.updatedAt
  );
}

function sortRecentChats(chats) {
  return [...chats].sort((a, b) => b.updatedAt - a.updatedAt);
}

function ChatSidebar({
  activeChatId,
  chats,
  onNewChat,
  onOpenChat,
  onOpenChatMenu,
}) {
  const pinnedChats = useMemo(
    () => sortPinnedChats(chats.filter((chat) => chat.isPinned)),
    [chats]
  );
  const recentChats = useMemo(
    () => sortRecentChats(chats.filter((chat) => !chat.isPinned)),
    [chats]
  );

  function onMenuButtonClick(event, chatId) {
    const rect = event.currentTarget.getBoundingClientRect();
    onOpenChatMenu(chatId, rect.right - 176, rect.bottom + 6);
  }

  function renderChatRow(chat) {
    return (
      <div
        key={chat.id}
        className={`chatEntry ${chat.id === activeChatId ? "active" : ""}`}
        onClick={() => onOpenChat(chat.id)}
        onContextMenu={(event) => {
          event.preventDefault();
          onOpenChatMenu(chat.id, event.clientX, event.clientY);
        }}
        onKeyDown={(event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            onOpenChat(chat.id);
          }
        }}
        role="button"
        tabIndex={0}
      >
        <div className="chatEntryMain">
          <div className="chatEntryTop">
            <span className="chatEntryTitle">{chat.title}</span>
            {chat.isPinned ? <span className="chatPinTag">Pinned</span> : null}
          </div>
          <div className="chatEntryPreview">{getChatPreview(chat)}</div>
        </div>

        <div className="chatEntryActions">
          <button
            className="chatMoreBtn"
            onClick={(event) => {
              event.stopPropagation();
              onMenuButtonClick(event, chat.id);
            }}
            title="Chat actions"
            type="button"
          >
            ...
          </button>
        </div>
      </div>
    );
  }

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

      <div className="sideSection chatsSection">
        <div className="sideSectionTitle">Chats</div>

        <div className="chatSection">
          {pinnedChats.length ? (
            <div className="chatGroup">
              <div className="chatSectionLabel">Pinned</div>
              <div className="chatList">{pinnedChats.map(renderChatRow)}</div>
            </div>
          ) : null}

          <div className="chatGroup">
            <div className="chatSectionLabel">Recent</div>
            <div className="chatList">
              {recentChats.length ? (
                recentChats.map(renderChatRow)
              ) : (
                <div className="chatListEmpty">
                  No recent chats yet. Start one and it will appear here.
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function ChatContextMenu({ chat, menu, onClose, onDeleteChat, onRenameChat, onTogglePin }) {
  if (!chat || !menu) return null;

  return (
    <div
      className="chatContextMenu"
      style={{ left: `${menu.x}px`, top: `${menu.y}px` }}
      onClick={(event) => event.stopPropagation()}
    >
      <button
        className="chatContextMenuItem"
        onClick={() => {
          onRenameChat(chat.id);
          onClose();
        }}
        type="button"
      >
        Rename
      </button>
      <button
        className="chatContextMenuItem"
        onClick={() => {
          onDeleteChat(chat.id);
          onClose();
        }}
        type="button"
      >
        Delete
      </button>
      <button
        className="chatContextMenuItem"
        onClick={() => {
          onTogglePin(chat.id);
          onClose();
        }}
        type="button"
      >
        {chat.isPinned ? "Unpin" : "Pin"}
      </button>
    </div>
  );
}

function RightPanel({ mode, onModeChange, onPickPrompt, quickPrompts }) {
  return (
    <div className="rightPanel">
      <div className="rightPanelTop">
        <div className="panelSection">
          <div className="sideSectionTitle">Mode</div>
          <div className="modeSwitch">
            <button
              className={`modeBtn ${mode === "rag" ? "active" : ""}`}
              onClick={() => onModeChange("rag")}
            >
              RAG
            </button>
            <button
              className={`modeBtn ${mode === "sql" ? "active" : ""}`}
              onClick={() => onModeChange("sql")}
            >
              SQL
            </button>
            <button
              className={`modeBtn ${mode === "graph" ? "active" : ""}`}
              onClick={() => onModeChange("graph")}
            >
              Graph
            </button>
            <button
              className={`modeBtn ${mode === "auto" ? "active" : ""}`}
              onClick={() => onModeChange("auto")}
            >
              AUTO
            </button>
          </div>
        </div>

        <div className="panelSection">
          <div className="sideSectionTitle">Quick prompts</div>
          <div className="sideChips">
            {quickPrompts.map((prompt) => (
              <button
                key={prompt.label}
                className="chip"
                onClick={() => onPickPrompt(prompt.q)}
              >
                {prompt.label}
              </button>
            ))}
          </div>
        </div>
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
            {columns.map((column) => (
              <th key={column}>{column}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={index}>
              {columns.map((column) => (
                <td key={column}>{String(row[column] ?? "")}</td>
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
  const hasSql =
    !isUser && (message.mode === "sql" || (message.mode === "auto" && message.sql));
  const hasGraph =
    !isUser && (message.mode === "graph" || (message.mode === "auto" && message.cypher));
  const hasAutoRoute = !isUser && message.mode === "auto";
  const selectedModes = message.selected_modes?.length
    ? message.selected_modes.join(" + ").toUpperCase()
    : "Unknown";

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
                <pre className="sqlCode">
                  {JSON.stringify(message.partial_errors, null, 2)}
                </pre>
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

        {hasGraph ? (
          <div className="sqlBlock">
            <details open>
              <summary>Generated Cypher</summary>
              <pre className="sqlCode">{message.cypher || "No Cypher generated."}</pre>
            </details>

            <details>
              <summary>Graph Results (JSON)</summary>
              <pre className="sqlCode">
                {message.results?.length
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
  const initialState = useMemo(() => loadChatState(), []);
  const [chats, setChats] = useState(initialState.chats);
  const [activeChatId, setActiveChatId] = useState(initialState.activeChatId);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [streamingChatId, setStreamingChatId] = useState(null);
  const [chatMenu, setChatMenu] = useState(null);

  const chatRef = useRef(null);
  const inputRef = useRef(null);
  const abortRef = useRef(null);

  const activeChat = chats.find((chat) => chat.id === activeChatId) || chats[0];
  const mode = activeChat?.mode || DEFAULT_MODE;
  const messages = activeChat?.messages ?? EMPTY_ARRAY;
  const isCurrentChatStreaming = streaming && streamingChatId === activeChat?.id;
  const menuChat = chats.find((chat) => chat.id === chatMenu?.chatId) || null;

  const quickPrompts = useMemo(() => {
    if (mode === "rag") {
      return [
        {
          label: "Urgent AML policy",
          q: "What does Meridian Trust Bank consider an urgent AML alert?",
        },
        {
          label: "EDD guidance",
          q: "When should enhanced due diligence be performed?",
        },
        {
          label: "Branch context",
          q: "How should branch context be used during risk review?",
        },
      ];
    }

    if (mode === "sql") {
      return [
        {
          label: "Urgent alerts",
          q: "Which customers have open AML alerts with scores above 80?",
        },
        {
          label: "Crypto total",
          q: "What is the total amount transferred to Northstar Crypto Exchange?",
        },
        {
          label: "Largest loan",
          q: "Which customer has the highest outstanding loan balance?",
        },
      ];
    }

    if (mode === "graph") {
      return [
        { label: "Asha path", q: "How is Asha Rao connected to Northstar Crypto Exchange?" },
        { label: "Urgent review", q: "Which alerts require Urgent AML Review?" },
        { label: "Downtown links", q: "Which customers are connected to Downtown Branch?" },
      ];
    }

    return [
      {
        label: "Asha escalation",
        q: "Does Asha Rao need escalation? Use her alert score, account relationships, and the bank policy.",
      },
      {
        label: "Risk + merchants",
        q: "Which high-risk customers have open alerts, and what merchants are their accounts connected to?",
      },
      {
        label: "Compare reviews",
        q: "Compare Asha Rao and Carlos Diaz from a risk review perspective using available records, relationships, and policy guidance.",
      },
    ];
  }, [mode]);

  useEffect(() => {
    saveChatState(chats, activeChatId);
  }, [activeChatId, chats]);

  useEffect(() => {
    if (!activeChat && chats.length > 0) {
      setActiveChatId(chats[0].id);
    }
  }, [activeChat, chats]);

  useEffect(() => {
    if (chatRef.current) {
      chatRef.current.scrollTop = chatRef.current.scrollHeight;
    }
  }, [messages, isCurrentChatStreaming]);

  useEffect(() => {
    inputRef.current?.focus();
  }, [activeChatId]);

  useEffect(() => {
    if (!chatMenu) return undefined;

    function closeMenu() {
      setChatMenu(null);
    }

    function closeOnEscape(event) {
      if (event.key === "Escape") closeMenu();
    }

    window.addEventListener("click", closeMenu);
    window.addEventListener("resize", closeMenu);
    window.addEventListener("keydown", closeOnEscape);

    return () => {
      window.removeEventListener("click", closeMenu);
      window.removeEventListener("resize", closeMenu);
      window.removeEventListener("keydown", closeOnEscape);
    };
  }, [chatMenu]);

  function updateChat(chatId, updater) {
    setChats((previousChats) =>
      previousChats.map((chat) => (chat.id === chatId ? updater(chat) : chat))
    );
  }

  function stop() {
    if (abortRef.current) abortRef.current.abort();
    abortRef.current = null;
    setStreaming(false);
    setStreamingChatId(null);
  }

  function openChat(chatId) {
    if (chatId === activeChatId) return;
    if (streaming) stop();
    setChatMenu(null);
    setActiveChatId(chatId);
    setInput("");
  }

  function createNewChat() {
    if (streaming) stop();
    setChatMenu(null);

    if (activeChat && activeChat.messages.length === 0 && activeChat.title === "New chat") {
      inputRef.current?.focus();
      return;
    }

    const nextChat = createChat({ mode });
    setChats((previousChats) => [nextChat, ...previousChats]);
    setActiveChatId(nextChat.id);
    setInput("");
  }

  function renameChat(chatId) {
    const chat = chats.find((item) => item.id === chatId);
    if (!chat) return;

    const nextTitle = window.prompt("Rename chat", chat.title);
    if (nextTitle === null) return;

    const trimmedTitle = nextTitle.trim();
    if (!trimmedTitle) return;

    updateChat(chatId, (currentChat) => ({
      ...currentChat,
      title: trimmedTitle,
      customTitle: true,
      updatedAt: Date.now(),
    }));
  }

  function togglePin(chatId) {
    updateChat(chatId, (chat) => ({
      ...chat,
      isPinned: !chat.isPinned,
      pinnedAt: chat.isPinned ? null : Date.now(),
      updatedAt: Date.now(),
    }));
  }

  function deleteChat(chatId) {
    const chat = chats.find((item) => item.id === chatId);
    if (!chat) return;
    if (!window.confirm(`Delete "${chat.title}"?`)) return;

    if (streamingChatId === chatId) stop();
    setChatMenu(null);

    const remainingChats = chats.filter((item) => item.id !== chatId);

    if (remainingChats.length === 0) {
      const fallbackChat = createChat({ mode });
      setChats([fallbackChat]);
      setActiveChatId(fallbackChat.id);
      setInput("");
      return;
    }

    setChats(remainingChats);

    if (activeChatId === chatId) {
      const nextActiveChat = sortPinnedChats(remainingChats.filter((item) => item.isPinned))[0]
        || sortRecentChats(remainingChats.filter((item) => !item.isPinned))[0]
        || remainingChats[0];
      setActiveChatId(nextActiveChat.id);
      setInput("");
    }
  }

  function openChatMenu(chatId, x, y) {
    const menuWidth = 176;
    const menuHeight = 132;
    const clampedX = Math.max(12, Math.min(x, window.innerWidth - menuWidth - 12));
    const clampedY = Math.max(12, Math.min(y, window.innerHeight - menuHeight - 12));
    setChatMenu({ chatId, x: clampedX, y: clampedY });
  }

  function setActiveMode(nextMode) {
    if (!activeChat) return;
    updateChat(activeChat.id, (chat) => ({ ...chat, mode: nextMode }));
  }

  function pickPrompt(question) {
    setInput(question);
    inputRef.current?.focus();
  }

  async function send(questionMaybe) {
    const question = (questionMaybe ?? input).trim();
    if (!question || streaming || !activeChat) return;

    const chatId = activeChat.id;
    const chatMode = activeChat.mode;
    const userMessage = { id: uid(), role: "user", text: question, mode: chatMode };

    updateChat(chatId, (chat) => {
      const nextMessages = clamp([...chat.messages, userMessage]);
      const shouldAutoTitle =
        !chat.customTitle && !chat.messages.some((message) => message.role === "user");

      return {
        ...chat,
        messages: nextMessages,
        title: shouldAutoTitle ? summarizeTitle(question) : chat.title,
        updatedAt: Date.now(),
      };
    });

    setInput("");
    setStreaming(true);
    setStreamingChatId(chatId);

    if (chatMode === "auto") {
      updateChat(chatId, (chat) => ({ ...chat, sources: [], updatedAt: Date.now() }));

      try {
        const response = await fetch("/api/auto/query", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question }),
        });

        const data = await response.json();

        updateChat(chatId, (chat) => ({
          ...chat,
          sources: data.sources || [],
          messages: clamp([
            ...chat.messages,
            {
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
            },
          ]),
          updatedAt: Date.now(),
        }));
      } catch (error) {
        updateChat(chatId, (chat) => ({
          ...chat,
          messages: clamp([
            ...chat.messages,
            { id: uid(), role: "assistant", mode: "auto", text: `Error: ${String(error)}` },
          ]),
          updatedAt: Date.now(),
        }));
      } finally {
        setStreaming(false);
        setStreamingChatId(null);
        inputRef.current?.focus();
      }
      return;
    }

    if (chatMode === "sql") {
      updateChat(chatId, (chat) => ({ ...chat, sources: [], updatedAt: Date.now() }));

      try {
        const response = await fetch("/api/sql/query", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question }),
        });

        const data = await response.json();

        updateChat(chatId, (chat) => ({
          ...chat,
          messages: clamp([
            ...chat.messages,
            {
              id: uid(),
              role: "assistant",
              mode: "sql",
              text: data.summary || "No summary returned.",
              sql: data.sql || "",
              columns: data.columns || [],
              rows: data.rows || [],
            },
          ]),
          updatedAt: Date.now(),
        }));
      } catch (error) {
        updateChat(chatId, (chat) => ({
          ...chat,
          messages: clamp([
            ...chat.messages,
            { id: uid(), role: "assistant", mode: "sql", text: `Error: ${String(error)}` },
          ]),
          updatedAt: Date.now(),
        }));
      } finally {
        setStreaming(false);
        setStreamingChatId(null);
        inputRef.current?.focus();
      }
      return;
    }

    if (chatMode === "graph") {
      updateChat(chatId, (chat) => ({ ...chat, sources: [], updatedAt: Date.now() }));

      try {
        const response = await fetch("/api/graph/query", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question }),
        });

        const data = await response.json();

        updateChat(chatId, (chat) => ({
          ...chat,
          messages: clamp([
            ...chat.messages,
            {
              id: uid(),
              role: "assistant",
              mode: "graph",
              text: data.summary || "No summary returned.",
              cypher: data.cypher || "",
              results: data.results || [],
            },
          ]),
          updatedAt: Date.now(),
        }));
      } catch (error) {
        updateChat(chatId, (chat) => ({
          ...chat,
          messages: clamp([
            ...chat.messages,
            { id: uid(), role: "assistant", mode: "graph", text: `Error: ${String(error)}` },
          ]),
          updatedAt: Date.now(),
        }));
      } finally {
        setStreaming(false);
        setStreamingChatId(null);
        inputRef.current?.focus();
      }
      return;
    }

    const assistantId = uid();

    updateChat(chatId, (chat) => ({
      ...chat,
      sources: [],
      messages: clamp([
        ...chat.messages,
        { id: assistantId, role: "assistant", mode: "rag", text: "" },
      ]),
      updatedAt: Date.now(),
    }));

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const response = await fetch("/api/ask/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
        signal: controller.signal,
      });

      if (!response.ok || !response.body) {
        updateChat(chatId, (chat) => ({
          ...chat,
          messages: chat.messages.map((message) =>
            message.id === assistantId
              ? { ...message, text: `Request failed (${response.status})` }
              : message
          ),
          updatedAt: Date.now(),
        }));
        setStreaming(false);
        setStreamingChatId(null);
        abortRef.current = null;
        return;
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const parsed = parseSSEChunk(buffer);
        buffer = parsed.rest;

        for (const event of parsed.events) {
          if (event.event === "sources") {
            const nextSources = parseJsonOr(event.data, EMPTY_ARRAY);
            updateChat(chatId, (chat) => ({
              ...chat,
              sources: Array.isArray(nextSources) ? nextSources : [],
              updatedAt: Date.now(),
            }));
            continue;
          }

          if (event.event === "token") {
            let token = "";

            try {
              token = JSON.parse(event.data);
            } catch {
              token = event.data;
            }

            updateChat(chatId, (chat) => ({
              ...chat,
              messages: chat.messages.map((message) =>
                message.id === assistantId
                  ? { ...message, text: `${message.text || ""}${token}` }
                  : message
              ),
              updatedAt: Date.now(),
            }));
            continue;
          }

          if (event.event === "error") {
            const errorMessage = parseJsonOr(event.data, event.data);

            updateChat(chatId, (chat) => ({
              ...chat,
              messages: chat.messages.map((message) =>
                message.id === assistantId
                  ? { ...message, text: `Error: ${errorMessage}` }
                  : message
              ),
              updatedAt: Date.now(),
            }));
            setStreaming(false);
            setStreamingChatId(null);
            abortRef.current = null;
            return;
          }

          if (event.event === "done") {
            setStreaming(false);
            setStreamingChatId(null);
            abortRef.current = null;
            return;
          }
        }
      }

      setStreaming(false);
      setStreamingChatId(null);
      abortRef.current = null;
    } catch (error) {
      const message = controller.signal.aborted ? "Stopped." : `Error: ${String(error)}`;

      updateChat(chatId, (chat) => ({
        ...chat,
        messages: chat.messages.map((item) =>
          item.id === assistantId ? { ...item, text: message } : item
        ),
        updatedAt: Date.now(),
      }));

      setStreaming(false);
      setStreamingChatId(null);
      abortRef.current = null;
    } finally {
      inputRef.current?.focus();
    }
  }

  function onKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      send();
    }
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
    auto: "AUTO mode routes to RAG, SQL, Graph, or a focused hybrid only when needed.",
  }[mode];

  return (
    <>
      <div className="appShell">
        <ChatSidebar
          activeChatId={activeChat?.id}
          chats={chats}
          onNewChat={createNewChat}
          onOpenChat={openChat}
          onOpenChatMenu={openChatMenu}
        />

        <div className="main">
          <div className="topBar">
            <div className="topTitle">{modeTitle}</div>
            <div className="topMeta">{modeMeta}</div>
          </div>

          <div className="chat" ref={chatRef}>
            {messages.length === 0 ? (
              <div className="empty">
                <div className="emptyTitle">{emptyTitle}</div>
                <div className="emptySub">
                  {mode === "rag" ? (
                    <>
                      Ask questions about banking policy files in <code>data/</code>.
                    </>
                  ) : mode === "sql" ? (
                    <>
                      Ask questions about banking tables in <code>data.db</code>.
                    </>
                  ) : mode === "graph" ? (
                    <>
                      Discover customer, account, merchant, alert, and policy
                      relationships in <code>Neo4j</code>.
                    </>
                  ) : (
                    <>
                      Let AUTO choose policy, records, relationships, or a focused
                      hybrid.
                    </>
                  )}
                </div>
              </div>
            ) : (
              messages.map((message) => <Message key={message.id} message={message} />)
            )}

            {isCurrentChatStreaming ? <TypingBubble /> : null}
          </div>

          <div className="composer">
            <div className="composerInner">
              <textarea
                ref={inputRef}
                className="composerInput"
                value={input}
                onChange={(event) => setInput(event.target.value)}
                onKeyDown={onKeyDown}
                placeholder={placeholder}
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
            <div className="composerHint">{composerHint}</div>
          </div>
        </div>

        <RightPanel
          mode={mode}
          onModeChange={setActiveMode}
          onPickPrompt={pickPrompt}
          quickPrompts={quickPrompts}
        />
      </div>

      <ChatContextMenu
        chat={menuChat}
        menu={chatMenu}
        onClose={() => setChatMenu(null)}
        onDeleteChat={deleteChat}
        onRenameChat={renameChat}
        onTogglePin={togglePin}
      />
    </>
  );
}
