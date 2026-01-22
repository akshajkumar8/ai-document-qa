import { useEffect, useMemo, useRef, useState } from "react";
import { uploadAndIndex, askQuestion, deleteDocument } from "./api";

export default function App() {
  const [doc, setDoc] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isUploading, setIsUploading] = useState(false);
  const [isAsking, setIsAsking] = useState(false);
  const [error, setError] = useState("");
  const [showEvidence, setShowEvidence] = useState(false);

  const fileInputRef = useRef(null);
  const chatBottomRef = useRef(null);

  // NEW: reference to the scrollable chat container
  const chatBodyRef = useRef(null);

  // NEW: only auto-scroll when user is near the bottom
  const [stickToBottom, setStickToBottom] = useState(true);
  const [isDragOver, setIsDragOver] = useState(false);

  useEffect(() => {
    const el = chatBodyRef.current;
    if (!el) return;

    const onScroll = () => {
      // within 80px of the bottom counts as "at bottom"
      const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
      setStickToBottom(nearBottom);
    };

    el.addEventListener("scroll", onScroll, { passive: true });
    onScroll(); // initialize
    return () => el.removeEventListener("scroll", onScroll);
  }, []);

  useEffect(() => {
    if (!stickToBottom) return;
    chatBottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages.length, showEvidence, stickToBottom]);

  const canAsk = !!doc?.doc_id && !isUploading && !isAsking;

  function resetAll() {
    setDoc(null);
    setMessages([]);
    setInput("");
    setError("");
    setShowEvidence(false);
    if (fileInputRef.current) fileInputRef.current.value = "";
  }

  async function handleRemoveDoc() {
    if (!doc) return;

    setError("");
    setShowEvidence(false);

    try {
      await deleteDocument(doc.doc_id);
    } catch (err) {
      setError(err?.message || "Failed to remove document.");
      return;
    }

    resetAll();
  }

  async function handlePickFile(e) {
    const file = e.target.files?.[0];
    if (!file) return;

    setError("");
    setIsUploading(true);
    setShowEvidence(false);

    try {
      const res = await uploadAndIndex(file);
      setDoc(res);
      setMessages([]);
    } catch (err) {
      setError(err?.message || "Upload failed. Make sure the backend is running.");
    } finally {
      setIsUploading(false);
    }
  }

  async function handleAsk(e) {
    e.preventDefault();
    if (!canAsk) return;

    const q = input.trim();
    if (!q) return;

    setError("");
    setIsAsking(true);
    setShowEvidence(false);

    setMessages((prev) => [...prev, { role: "user", text: q }]);
    setInput("");

    try {
      const res = await askQuestion({
        doc_id: doc.doc_id,
        question: q,
        top_k: 5,
      });

      setMessages((prev) => [
        ...prev,
        {
          role: "ai",
          text: res.answer,
          sources: res.sources || [],
          evidence: res.evidence || [],
        },
      ]);
    } catch (err) {
      setError(err?.message || "Request failed. Make sure the backend is running.");
    } finally {
      setIsAsking(false);
    }
  }

  const lastAi = useMemo(() => {
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === "ai") return messages[i];
    }
    return null;
  }, [messages]);

  return (
    <div className="appShell">
      <div className="panelWrap">
        {/* LEFT */}
        <div
          className={`leftPanel ${!doc ? "leftPanelInteractive" : ""} ${
            isDragOver ? "leftPanelDragOver" : ""
          }`}
          onClick={() => {
            if (!doc && !isUploading) {
              fileInputRef.current?.click();
            }
          }}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => {
            if (!doc && (e.key === "Enter" || e.key === " ")) {
              e.preventDefault();
              if (!isUploading) {
                fileInputRef.current?.click();
              }
            }
          }}
          onDragEnter={(e) => {
            e.preventDefault();
            if (!doc) setIsDragOver(true);
          }}
          onDragOver={(e) => {
            e.preventDefault();
            if (!doc && !isDragOver) setIsDragOver(true);
          }}
          onDragLeave={(e) => {
            e.preventDefault();
            if (isDragOver) setIsDragOver(false);
          }}
          onDrop={async (e) => {
            e.preventDefault();
            if (isDragOver) setIsDragOver(false);

            if (doc || isUploading) return;

            const file = e.dataTransfer?.files?.[0];
            if (!file) return;

            // Mirror handlePickFile behavior
            setError("");
            setIsUploading(true);
            setShowEvidence(false);

            try {
              const res = await uploadAndIndex(file);
              setDoc(res);
              setMessages([]);
            } catch (err) {
              setError(err?.message || "Upload failed. Make sure the backend is running.");
            } finally {
              setIsUploading(false);
            }
          }}
        >
          {!doc ? (
            <div className="uploadCard">
              <div className="uploadTitle">Upload a PDF</div>
              <div className="uploadSub">
                Drag &amp; drop anywhere on this side <br />
                or click to browse
              </div>

              <div className="uploadControls">
                <button
                  type="button"
                  className="btn uploadBtn"
                  onClick={(e) => {
                    e.stopPropagation();
                    if (!isUploading) {
                      fileInputRef.current?.click();
                    }
                  }}
                  disabled={isUploading}
                >
                  {isUploading ? "Uploading..." : "Choose file"}
                </button>

                {isUploading && (
                  <span className="muted inlineUploadStatus">Uploading &amp; indexing…</span>
                )}
              </div>

              <input
                ref={fileInputRef}
                type="file"
                accept="application/pdf"
                onChange={handlePickFile}
                disabled={isUploading}
                className="fileInputNative"
              />

              {error && <div className="error">{error}</div>}
            </div>
          ) : (
            <div className="docCard">
              <div className="docLabel">DOCUMENT</div>
              <div className="docName">{doc.original_filename}</div>

              <div className="docStats">
                <div>
                  <div className="statLabel">Status</div>
                  <div className="statValue">Ready</div>
                </div>
                <div>
                  <div className="statLabel">Pages</div>
                  <div className="statValue">{doc.pages}</div>
                </div>
                <div>
                  <div className="statLabel">Chunks</div>
                  <div className="statValue">{doc.chunks}</div>
                </div>
              </div>

              <div className="docButtons">
                <button
                  className="btn"
                  onClick={() => fileInputRef.current?.click()}
                  disabled={isUploading}
                >
                  Replace PDF
                </button>

                <button className="btn" onClick={() => setMessages([])} disabled={isAsking}>
                  Clear chat
                </button>

                <button className="btn" onClick={handleRemoveDoc} disabled={isUploading || isAsking}>
                  Remove document
                </button>
              </div>

              <div className="docTip">
                Tip: Ask questions on the right. “Show evidence” reveals the excerpt used.
              </div>

              <input
                ref={fileInputRef}
                type="file"
                accept="application/pdf"
                onChange={handlePickFile}
                style={{ display: "none" }}
              />

              {error && <div className="error">{error}</div>}
            </div>
          )}
        </div>

        {/* RIGHT */}
        <div className="rightPanel">
          <div className="chatHeader">Chat</div>

          {/* NEW: attach ref to the scroll container */}
          <div className="chatBody" ref={chatBodyRef}>
            {!doc ? (
              <div className="emptyState">Upload a PDF on the left to start chatting.</div>
            ) : (
              <div className="chatList">
                {messages.map((m, idx) =>
                  m.role === "user" ? (
                    <div key={idx} className="msgUser">
                      <div className="msgRole">You</div>
                      <div className="msgText">{m.text}</div>
                    </div>
                  ) : (
                    <div key={idx} className="msgAiWrap">
                      <div className="msgAi">
                        <div className="msgRole">AI</div>
                        <div className="msgText">{m.text}</div>
                      </div>

                      {(m.sources || []).length > 0 && (
                        <div className="sourcesRow">
                          <div className="sourcesLabel">Sources:</div>
                          <div className="sourcesBadges">
                            {m.sources.map((p) => (
                              <span key={p} className="badge">
                                Page {p}
                              </span>
                            ))}
                          </div>
                        </div>
                      )}

                      {lastAi === m && (m.evidence || []).length > 0 && (
                        <>
                          <button
                            className="btn evidenceBtn"
                            onClick={() => setShowEvidence((v) => !v)}
                          >
                            {showEvidence ? "Hide evidence" : "Show evidence"}
                          </button>

                          {showEvidence && (
                            <div className="evidenceCard">
                              <div className="evidenceTitle">Evidence</div>
                              {(m.evidence || []).slice(0, 2).map((ev, i) => (
                                <div key={i} className="evidenceRow">
                                  <div className="evidencePagePill">Page {ev.page}</div>
                                  <div className="evidenceSnippet">{ev.excerpt}</div>
                                </div>
                              ))}
                            </div>
                          )}
                        </>
                      )}
                    </div>
                  )
                )}

                <div ref={chatBottomRef} />
              </div>
            )}

            {error && <div className="error">{error}</div>}
          </div>

          <form className="chatComposer" onSubmit={handleAsk}>
            <input
              className="chatInput"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder={doc ? "Ask something about the document..." : "Upload a PDF to start..."}
              disabled={!doc || isAsking}
            />
            <button className="sendBtn" type="submit" disabled={!doc || isAsking}>
              {isAsking ? "..." : "Send"}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
