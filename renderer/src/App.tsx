import { useCallback, useEffect, useRef, useState } from "react";
import { api, cmd, isElectron } from "./api";

type TabId =
  | "transcript"
  | "record"
  | "summary"
  | "ask"
  | "prompts"
  | "models"
  | "settings";

const TABS: { id: TabId; label: string }[] = [
  { id: "transcript", label: "Transcript" },
  { id: "record", label: "Record" },
  { id: "summary", label: "AI Summary" },
  { id: "ask", label: "Ask AI" },
  { id: "prompts", label: "AI Prompts" },
  { id: "models", label: "Models" },
  { id: "settings", label: "Settings" },
];

export default function App() {
  const [ready, setReady] = useState(false);
  const [tab, setTab] = useState<TabId>("transcript");
  const [recording, setRecording] = useState(false);
  const [level, setLevel] = useState(0);
  const [status, setStatus] = useState("Loading…");
  const [error, setError] = useState<string | null>(null);
  const [version, setVersion] = useState("");

  const [meetings, setMeetings] = useState<Meeting[]>([]);
  const [currentId, setCurrentId] = useState<string | null>(null);
  const [transcript, setTranscript] = useState("");
  const [manualNotes, setManualNotes] = useState("");
  const [summary, setSummary] = useState("");
  const [chatMessages, setChatMessages] = useState<{ role: string; content: string }[]>([]);
  const [meetingName, setMeetingName] = useState("New Meeting");
  const [meetingDate, setMeetingDate] = useState("");

  const [settings, setSettings] = useState<Settings | null>(null);
  const [prompts, setPrompts] = useState<Prompt[]>([]);
  const [selectedPromptId, setSelectedPromptId] = useState("");
  const [modelInstalled, setModelInstalled] = useState(false);
  const [modelDownloading, setModelDownloading] = useState(false);
  const [apiKeyMasked, setApiKeyMasked] = useState("");
  const [apiKeyInput, setApiKeyInput] = useState("");
  const [devices, setDevices] = useState<{
    inputs: { index: number; name: string }[];
    loopbacks: { index: number; name: string }[];
  }>({ inputs: [], loopbacks: [] });

  const [askInput, setAskInput] = useState("");
  const [askLoading, setAskLoading] = useState(false);
  const [summaryLoading, setSummaryLoading] = useState(false);
  const [exportPrependDate, setExportPrependDate] = useState(true);

  const saveTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const scheduleSave = useCallback(() => {
    if (!currentId) return;
    if (saveTimer.current) clearTimeout(saveTimer.current);
    saveTimer.current = setTimeout(async () => {
      const m: Meeting = {
        id: currentId,
        meeting_name: meetingName,
        meeting_date: meetingDate || undefined,
        manual_notes: manualNotes,
        transcript,
        ai_summary: summary,
        ai_chat_messages: chatMessages,
        updated_at: new Date().toISOString(),
      };
      await cmd("save_meeting", { meeting: m });
    }, 1000);
  }, [currentId, meetingName, meetingDate, manualNotes, transcript, summary, chatMessages]);

  useEffect(() => {
    scheduleSave();
  }, [manualNotes, transcript, summary, chatMessages, meetingName, meetingDate, scheduleSave]);

  const loadMeeting = useCallback(async (id: string) => {
    const m = await cmd<Meeting>("load_meeting", { id });
    setCurrentId(m.id);
    setMeetingName(m.meeting_name || "New Meeting");
    setMeetingDate(m.meeting_date || "");
    setTranscript(m.transcript || "");
    setManualNotes(m.manual_notes || "");
    setSummary(m.ai_summary || "");
    setChatMessages(m.ai_chat_messages || []);
  }, []);

  const refreshMeetings = useCallback(async () => {
    const list = await cmd<Meeting[]>("list_meetings");
    const meetings = Array.isArray(list) ? list : [];
    setMeetings(meetings);
    if (meetings.length && !currentId) {
      await loadMeeting(meetings[0].id);
    }
  }, [currentId, loadMeeting]);

  const refreshAll = useCallback(async () => {
    const withTimeout = <T,>(p: Promise<T>, ms = 60000): Promise<T> =>
      Promise.race([
        p,
        new Promise<T>((_, reject) =>
          setTimeout(() => reject(new Error("Timed out connecting to audio engine")), ms)
        ),
      ]);
    const [ver, s, p, dev, model] = await Promise.all([
      withTimeout(cmd<{ version: string }>("get_version")),
      withTimeout(cmd<Settings>("get_settings")),
      withTimeout(cmd<Prompt[]>("list_prompts")),
      withTimeout(
        cmd<{ inputs: { index: number; name: string }[]; loopbacks: { index: number; name: string }[] }>("list_devices")
      ),
      withTimeout(cmd<{ installed: boolean }>("model_status")),
    ]);
    setVersion(ver.version);
    setSettings(s);
    const promptList = Array.isArray(p) ? p : [];
    setPrompts(promptList);
    if (promptList.length) setSelectedPromptId(promptList[0].id);
    setDevices(dev);
    setModelInstalled(!!model.installed);
    setExportPrependDate(!!s.export_prepend_meeting_date);
    const key = await cmd<{ has_key: boolean; masked: string }>("get_api_key");
    setApiKeyMasked(key.masked);
    await refreshMeetings();
    setReady(true);
    setStatus("Ready to record & transcribe");
  }, [refreshMeetings]);

  useEffect(() => {
    if (!isElectron()) {
      setError(
        "This page only works inside the Electron app. Look for the Rapid Scribe desktop window opened by npm run dev — do not use this browser tab."
      );
      return;
    }
    refreshAll().catch((e) => setError(String(e)));
    const unsubs = [
      api.on("sidecar:transcript_line", (d) => {
        const data = d as { text?: string };
        if (data.text) {
          setTranscript((t) => t + data.text);
          setTab((current) => (current === "record" ? "transcript" : current));
        }
      }),
      api.on("sidecar:audio_level", (d) => {
        const data = d as { level?: number };
        setLevel(data.level ?? 0);
      }),
      api.on("sidecar:status", (d) => {
        const data = d as { message?: string };
        if (data.message === "recording") setRecording(true);
        if (data.message === "stopped" || data.message === "stopping") {
          if (data.message === "stopped") setRecording(false);
          setStatus(data.message === "stopping" ? "Stopping…" : "Stopped");
        }
      }),
      api.on("sidecar:capture_error", (d) => {
        const data = d as { message?: string };
        setError(data.message || "Capture error");
        setRecording(false);
      }),
      api.on("sidecar:exit", (d) => {
        const data = d as { code?: number; log?: string };
        setError(`Audio engine exited (${data.code}). ${data.log || ""}`);
        setRecording(false);
      }),
    ];
    return () => unsubs.forEach((u) => u());
  }, [refreshAll]);

  const startRecording = async () => {
    setError(null);
    if (!modelInstalled) {
      setTab("models");
      setError("Install the transcription model first (Models tab).");
      return;
    }
    const ok = await cmd<{ recording?: boolean }>("start_recording");
    if (ok) setRecording(true);
  };

  const stopRecording = async () => {
    await cmd("stop_recording");
    setRecording(false);
    if (settings?.auto_generate_summary_when_stopping) {
      await generateSummary();
    }
  };

  const generateSummary = async () => {
    const prompt = prompts.find((p) => p.id === selectedPromptId);
    if (!prompt) {
      setError("Select a prompt");
      return;
    }
    setSummaryLoading(true);
    setError(null);
    try {
      const result = await cmd<{ text: string }>("generate_summary", {
        prompt_id: prompt.id,
        transcript,
        manual_notes: manualNotes,
      });
      setSummary(result.text);
    } catch (e) {
      setError(String(e));
    } finally {
      setSummaryLoading(false);
    }
  };

  const sendAsk = async () => {
    if (!askInput.trim()) return;
    setAskLoading(true);
    const userMsg = { role: "user", content: askInput.trim() };
    const nextChat = [...chatMessages, userMsg];
    setChatMessages(nextChat);
    setAskInput("");
    try {
      const result = await cmd<{ text: string }>("ask_ai", {
        manual_notes: manualNotes,
        transcript,
        ai_summary: summary,
        chat_messages: chatMessages,
        message: userMsg.content,
      });
      setChatMessages([...nextChat, { role: "assistant", content: result.text }]);
    } catch (e) {
      setError(String(e));
    } finally {
      setAskLoading(false);
    }
  };

  const exportMarkdown = async () => {
    if (!summary && !transcript && !manualNotes) {
      setError("Nothing to export");
      return;
    }
    let namePart = meetingName.replace(/[^\w .-]/g, "-").trim() || "export";
    const datePart = exportPrependDate && meetingDate ? `${meetingDate} ` : "";
    const parts: string[] = [];
    if (summary) parts.push("# AI Summary\n\n" + summary);
    if (manualNotes) parts.push("# Manual Notes\n\n" + manualNotes);
    if (transcript) parts.push("# Full Transcript\n\n" + transcript);
    await api.saveMarkdown(`${datePart}${namePart}.md`, parts.join("\n\n"));
  };

  const downloadModel = async () => {
    setModelDownloading(true);
    const unsub = api.on("sidecar:model_progress", (d) => {
      const data = d as { status?: string; error?: string };
      if (data.status === "done") {
        setModelInstalled(true);
        setModelDownloading(false);
      }
      if (data.status === "error") {
        setError(data.error || "Download failed");
        setModelDownloading(false);
      }
    });
    try {
      await cmd("download_model", {});
      setModelInstalled(true);
    } catch (e) {
      setError(String(e));
    } finally {
      setModelDownloading(false);
      unsub();
    }
  };

  if (!ready) {
    return (
      <div className="splash">
        <h1>Rapid Scribe</h1>
        {!isElectron() ? (
          <p className="muted" style={{ maxWidth: 420, textAlign: "center", lineHeight: 1.5 }}>
            Open the <strong>Electron desktop window</strong> (not this browser tab).
            It launches automatically when you run <code>npm run dev</code>.
          </p>
        ) : (
          <p className="muted">Loading audio engine…</p>
        )}
        {error && <p className="error-banner" style={{ maxWidth: 480 }}>{error}</p>}
      </div>
    );
  }

  return (
    <div className="app-shell">
      {error && (
        <p className="error-banner">
          {error}
          <button type="button" className="secondary" style={{ marginLeft: 8 }} onClick={() => setError(null)}>
            Dismiss
          </button>
        </p>
      )}
      <header className="top-bar">
        <h1>Rapid Scribe</h1>
        <span className="muted">{status}</span>
        <div className="level-meter" title="Input level">
          <span style={{ width: `${Math.round(level * 100)}%` }} />
        </div>
        {recording ? (
          <button type="button" className="danger" onClick={stopRecording}>
            Stop
          </button>
        ) : (
          <button type="button" onClick={startRecording} disabled={!modelInstalled}>
            Record
          </button>
        )}
        <span className="muted">v{version}</span>
      </header>

      <div className="main-layout">
        <aside className="sidebar">
          <button
            type="button"
            className="secondary"
            style={{ width: "100%", marginBottom: 8 }}
            onClick={async () => {
              const m = await cmd<Meeting>("create_meeting", { meeting_name: "New Meeting" });
              await refreshMeetings();
              await loadMeeting(m.id);
            }}
          >
            + New meeting
          </button>
          {meetings.map((m) => (
            <button
              key={m.id}
              type="button"
              className={`meeting-item ${m.id === currentId ? "active" : ""}`}
              onClick={() => loadMeeting(m.id)}
            >
              {m.meeting_name || "Untitled"}
            </button>
          ))}
        </aside>

        <div className="content">
          <nav className="tabs">
            {TABS.map((t) => (
              <button
                key={t.id}
                type="button"
                className={tab === t.id ? "active" : ""}
                onClick={() => setTab(t.id)}
              >
                {t.label}
              </button>
            ))}
          </nav>

          <div className="tab-panel">
            {tab === "transcript" && (
              <>
                <div className="row">
                  <input
                    value={meetingName}
                    onChange={(e) => setMeetingName(e.target.value)}
                    placeholder="Meeting name"
                    style={{ flex: 1, minWidth: 160 }}
                  />
                  <input
                    type="date"
                    value={meetingDate}
                    onChange={(e) => setMeetingDate(e.target.value)}
                  />
                </div>
                <div className="card">
                  <h2>Manual Notes</h2>
                  <textarea value={manualNotes} onChange={(e) => setManualNotes(e.target.value)} rows={5} />
                </div>
                <div className="card">
                  <h2>Live Transcript</h2>
                  <div className="transcript-box">{transcript || <span className="muted">Press Record to transcribe…</span>}</div>
                  <div className="row" style={{ marginTop: 12 }}>
                    <label>
                      <input
                        type="checkbox"
                        checked={exportPrependDate}
                        onChange={(e) => setExportPrependDate(e.target.checked)}
                      />{" "}
                      Prepend meeting date to export file name
                    </label>
                    <button type="button" className="secondary" onClick={exportMarkdown}>
                      Export Markdown
                    </button>
                    <button
                      type="button"
                      className="secondary"
                      onClick={() => {
                        setTranscript("");
                        setManualNotes("");
                      }}
                    >
                      Clear
                    </button>
                  </div>
                </div>
              </>
            )}

            {tab === "record" && (
              <>
                <div className="card">
                  <h2>Recording</h2>
                  <p className="muted">
                    Mode: <strong>{settings?.audio_mode || "meeting"}</strong> — configure devices in Settings.
                    Live text appears on the <strong>Transcript</strong> tab (first lines may take several seconds in
                    meeting/VAD mode).
                  </p>
                  <div className="row">
                    {recording ? (
                      <button type="button" className="danger" onClick={stopRecording}>
                        Stop recording
                      </button>
                    ) : (
                      <button type="button" onClick={startRecording} disabled={!modelInstalled}>
                        Start recording
                      </button>
                    )}
                  </div>
                  <p>Level: {Math.round(level * 100)}%</p>
                </div>
                <div className="card">
                  <h2>Live Transcript</h2>
                  <div className="transcript-box">
                    {transcript || (
                      <span className="muted">
                        {recording ? "Listening… waiting for speech chunk." : "Press Record to transcribe…"}
                      </span>
                    )}
                  </div>
                </div>
              </>
            )}

            {tab === "summary" && (
              <div className="card">
                <h2>AI Summary</h2>
                <div className="row">
                  <select value={selectedPromptId} onChange={(e) => setSelectedPromptId(e.target.value)}>
                    {prompts.map((p) => (
                      <option key={p.id} value={p.id}>
                        {p.name}
                      </option>
                    ))}
                  </select>
                  <button type="button" onClick={generateSummary} disabled={summaryLoading}>
                    {summaryLoading ? "Generating…" : "Generate"}
                  </button>
                </div>
                <textarea value={summary} onChange={(e) => setSummary(e.target.value)} rows={16} />
              </div>
            )}

            {tab === "ask" && (
              <div className="card">
                <h2>Ask AI</h2>
                <div className="chat-messages">
                  {chatMessages.map((m, i) => (
                    <div key={i} className={`chat-msg ${m.role}`}>
                      <strong>{m.role}</strong>
                      <pre style={{ margin: "4px 0 0", whiteSpace: "pre-wrap", fontFamily: "inherit" }}>{m.content}</pre>
                    </div>
                  ))}
                </div>
                <div className="row">
                  <input
                    style={{ flex: 1 }}
                    value={askInput}
                    onChange={(e) => setAskInput(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && sendAsk()}
                    placeholder="Ask about this meeting…"
                  />
                  <button type="button" onClick={sendAsk} disabled={askLoading}>
                    Send
                  </button>
                  <button type="button" className="secondary" onClick={() => setChatMessages([])}>
                    Clear chat
                  </button>
                </div>
              </div>
            )}

            {tab === "prompts" && (
              <PromptsTab prompts={prompts} onRefresh={async () => setPrompts(await cmd("list_prompts"))} />
            )}

            {tab === "models" && (
              <div className="card">
                <h2>Transcription model</h2>
                {modelInstalled ? (
                  <p className="muted">Model is installed and ready.</p>
                ) : (
                  <>
                    <p className="muted">Download the Parakeet ONNX model (~650 MB) before your first recording.</p>
                    <button type="button" onClick={downloadModel} disabled={modelDownloading}>
                      {modelDownloading ? "Downloading…" : "Download & install"}
                    </button>
                  </>
                )}
              </div>
            )}

            {tab === "settings" && settings && (
              <SettingsTab
                settings={settings}
                devices={devices}
                apiKeyMasked={apiKeyMasked}
                apiKeyInput={apiKeyInput}
                onApiKeyInput={setApiKeyInput}
                onSave={async (patch) => {
                  const next = await cmd<Settings>("set_settings", { settings: { ...settings, ...patch } });
                  setSettings(next);
                }}
                onSaveApiKey={async () => {
                  await cmd("set_api_key", { api_key: apiKeyInput });
                  const k = await cmd<{ masked: string }>("get_api_key");
                  setApiKeyMasked(k.masked);
                  setApiKeyInput("");
                }}
                onCheckUpdate={async () => {
                  const u = await cmd<{ update_available?: boolean; download_url?: string }>("check_for_update");
                  if (u.update_available && u.download_url) {
                    await api.openExternal(u.download_url);
                  } else {
                    alert("You are on the latest version.");
                  }
                }}
              />
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function PromptsTab({
  prompts,
  onRefresh,
}: {
  prompts: Prompt[];
  onRefresh: () => Promise<void>;
}) {
  const [editId, setEditId] = useState<string | null>(null);
  const [name, setName] = useState("");
  const [text, setText] = useState("");

  const startEdit = (p?: Prompt) => {
    if (p) {
      setEditId(p.id);
      setName(p.name);
      setText(p.prompt);
    } else {
      setEditId("new");
      setName("");
      setText("Summarize this meeting:\n\n{{transcript}}\n\nNotes:\n{{manual_notes}}");
    }
  };

  const save = async () => {
    await cmd("save_prompt", {
      id: editId === "new" ? undefined : editId,
      name,
      prompt: text,
    });
    setEditId(null);
    await onRefresh();
  };

  return (
    <div className="card">
      <h2>AI Prompts</h2>
      <ul style={{ listStyle: "none", padding: 0 }}>
        {prompts.map((p) => (
          <li key={p.id} className="row">
            <span style={{ flex: 1 }}>{p.name}</span>
            <button type="button" className="secondary" onClick={() => startEdit(p)}>
              Edit
            </button>
            <button
              type="button"
              className="danger"
              onClick={async () => {
                if (confirm(`Delete "${p.name}"?`)) {
                  await cmd("delete_prompt", { id: p.id });
                  await onRefresh();
                }
              }}
            >
              Delete
            </button>
          </li>
        ))}
      </ul>
      <button type="button" onClick={() => startEdit()} style={{ marginBottom: 12 }}>
        Add prompt
      </button>
      {editId && (
        <div>
          <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Name" style={{ width: "100%", marginBottom: 8 }} />
          <textarea value={text} onChange={(e) => setText(e.target.value)} rows={10} />
          <div className="row">
            <button type="button" onClick={save}>
              Save
            </button>
            <button type="button" className="secondary" onClick={() => setEditId(null)}>
              Cancel
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

function SettingsTab({
  settings,
  devices,
  apiKeyMasked,
  apiKeyInput,
  onApiKeyInput,
  onSave,
  onSaveApiKey,
  onCheckUpdate,
}: {
  settings: Settings;
  devices: { inputs: { index: number; name: string }[]; loopbacks: { index: number; name: string }[] };
  apiKeyMasked: string;
  apiKeyInput: string;
  onApiKeyInput: (v: string) => void;
  onSave: (patch: Partial<Settings>) => Promise<void>;
  onSaveApiKey: () => Promise<void>;
  onCheckUpdate: () => Promise<void>;
}) {
  return (
    <>
      <div className="card">
        <h2>Audio</h2>
        <div className="row">
          <label>Mode</label>
          <select
            value={settings.audio_mode}
            onChange={(e) => onSave({ audio_mode: e.target.value })}
          >
            <option value="meeting">Meeting (mic + loopback)</option>
            <option value="loopback">Loopback only</option>
            <option value="default">Default input</option>
          </select>
        </div>
        <div className="row">
          <label>Mic device</label>
          <select
            value={settings.meeting_mic_device ?? ""}
            onChange={(e) =>
              onSave({ meeting_mic_device: e.target.value === "" ? null : Number(e.target.value) })
            }
          >
            <option value="">System default</option>
            {devices.inputs
              .filter((d) => d.name)
              .map((d) => (
                <option key={d.index} value={d.index}>
                  {d.name}
                </option>
              ))}
          </select>
        </div>
        <div className="row">
          <label>Loopback</label>
          <select
            value={settings.loopback_device_index ?? ""}
            onChange={(e) =>
              onSave({ loopback_device_index: e.target.value === "" ? null : Number(e.target.value) })
            }
          >
            <option value="">Default speakers</option>
            {devices.loopbacks.map((d) => (
              <option key={d.index} value={d.index}>
                {d.name}
              </option>
            ))}
          </select>
        </div>
        <label>
          <input
            type="checkbox"
            checked={!!settings.auto_generate_summary_when_stopping}
            onChange={(e) => onSave({ auto_generate_summary_when_stopping: e.target.checked })}
          />{" "}
          Auto-generate summary when stopping recording
        </label>
      </div>
      <div className="card">
        <h2>OpenAI API key</h2>
        {apiKeyMasked && <p className="muted">Current: {apiKeyMasked}</p>}
        <input
          type="password"
          placeholder="sk-…"
          value={apiKeyInput}
          onChange={(e) => onApiKeyInput(e.target.value)}
          style={{ width: "100%", marginBottom: 8 }}
        />
        <button type="button" onClick={onSaveApiKey}>
          Save API key
        </button>
      </div>
      <div className="card">
        <h2>Updates</h2>
        <button type="button" className="secondary" onClick={onCheckUpdate}>
          Check for updates
        </button>
        <p className="muted">Packaged builds also check via electron-updater on startup.</p>
      </div>
    </>
  );
}
