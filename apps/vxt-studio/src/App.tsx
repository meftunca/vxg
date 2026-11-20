import * as React from "react";
import { open } from "@tauri-apps/plugin-dialog";
import type { Message } from "ai";
import {
  CloudDownload,
  Cpu,
  Globe,
  Loader2,
  MessageSquare,
  PanelLeftClose,
  Plus,
  PowerOff,
  Trash2,
} from "lucide-react";
import { ApiHubWorkspace } from "@/components/ApiHubWorkspace";
import { ChatWorkspace } from "@/components/ChatWorkspace";
import { ForgeWorkspace } from "@/components/ForgeWorkspace";
import {
  fetchStudioState,
  isTauriRuntime,
  listenForgeProgress,
  loadModel,
  resetChatCache,
  unloadModel,
  type ForgeProgressPayload,
  type StudioState,
} from "@/lib/vxt-bridge";

interface ChatThread {
  id: string;
  title: string;
  updatedAt: number;
  messages: Message[];
}

const STORAGE_KEY = "vxt_studio_threads_v1";

function createThread(): ChatThread {
  return {
    id: `chat-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
    title: "Yeni Sohbet",
    updatedAt: Date.now(),
    messages: [],
  };
}

function loadThreads(): ChatThread[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [createThread()];
    const parsed = JSON.parse(raw) as ChatThread[];
    if (Array.isArray(parsed) && parsed.length > 0) return parsed;
  } catch {
    // Ignore corrupt localStorage
  }
  return [createThread()];
}

function deriveTitle(messages: Message[]): string {
  const firstUser = messages.find((m) => m.role === "user" && m.content.trim().length > 0);
  if (!firstUser) return "Yeni Sohbet";
  const single = firstUser.content.replace(/\s+/g, " ").trim();
  return single.length > 36 ? `${single.slice(0, 36)}…` : single;
}

function shortBackendLabel(backend?: string): string {
  if (!backend) return "GPU Hazır";
  if (backend.includes("CUDA") && backend.includes("Resident")) return "CUDA Resident";
  if (backend.includes("CUDA")) return "NVIDIA CUDA";
  if (backend.includes("VXT-Turbo")) return "Metal VXT-Turbo";
  if (backend.includes("CB4-Fast")) return "Metal CB4-Fast";
  if (backend.includes("Metal")) return "Apple Metal";
  return backend.split("(")[0].trim() || "CPU";
}

export default function App() {
  const [view, setView] = React.useState<"chat" | "hub" | "api">("chat");
  const [sidebarCollapsed, setSidebarCollapsed] = React.useState(false);
  const [studio, setStudio] = React.useState<StudioState | null>(null);
  const [loadingModel, setLoadingModel] = React.useState(false);
  const [loadErr, setLoadErr] = React.useState<string | null>(null);

  // Persistent multi-thread conversations
  const [threads, setThreads] = React.useState<ChatThread[]>(loadThreads);
  const [activeThreadId, setActiveThreadId] = React.useState<string>(() => loadThreads()[0].id);

  // Background Forge / HF Download state (persists across tabs)
  const [forgeBusy, setForgeBusy] = React.useState(false);
  const [forgeJobName, setForgeJobName] = React.useState<string | null>(null);
  const [forgeProgress, setForgeProgress] = React.useState<ForgeProgressPayload | null>(null);

  React.useEffect(() => {
    try {
      // Avoid storing huge base64 images in localStorage
      const light = threads.slice(0, 25).map((t) => ({
        ...t,
        messages: t.messages.map((m) => ({
          id: m.id,
          role: m.role,
          content: m.content,
        })),
      }));
      localStorage.setItem(STORAGE_KEY, JSON.stringify(light));
    } catch {
      // Ignore quota errors
    }
  }, [threads]);

  React.useEffect(() => {
    let unlisten: (() => void) | undefined;
    void listenForgeProgress((payload) => {
      setForgeProgress(payload);
    }).then((fn) => {
      unlisten = fn;
    });
    return () => {
      if (unlisten) unlisten();
    };
  }, []);

  const refresh = React.useCallback(async () => {
    try {
      const st = await fetchStudioState();
      setStudio(st);
    } catch {
      // Embedded server still booting
    }
  }, []);

  React.useEffect(() => {
    void refresh();
    const timer = setInterval(() => {
      void refresh();
    }, 2000);
    return () => clearInterval(timer);
  }, [refresh]);

  const handleNewChat = React.useCallback(async () => {
    const fresh = createThread();
    setThreads((prev) => [fresh, ...prev]);
    setActiveThreadId(fresh.id);
    setView("chat");
    try {
      await resetChatCache(studio?.base_url);
      await refresh();
    } catch {
      // Ignore
    }
  }, [studio?.base_url, refresh]);

  // Cmd+N shortcut for New Chat
  React.useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "n") {
        e.preventDefault();
        void handleNewChat();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [handleNewChat]);

  const handleMessagesChange = React.useCallback((tid: string, msgs: Message[]) => {
    setThreads((prev) =>
      prev.map((t) => {
        if (t.id !== tid) return t;
        if (t.messages.length === msgs.length && t.messages[msgs.length - 1]?.content === msgs[msgs.length - 1]?.content) {
          return t;
        }
        return {
          ...t,
          title: deriveTitle(msgs),
          updatedAt: Date.now(),
          messages: msgs,
        };
      }),
    );
  }, []);

  const handleDeleteThread = (tid: string) => {
    setThreads((prev) => {
      const next = prev.filter((t) => t.id !== tid);
      if (next.length === 0) {
        const fresh = createThread();
        setActiveThreadId(fresh.id);
        return [fresh];
      }
      if (activeThreadId === tid) {
        setActiveThreadId(next[0].id);
      }
      return next;
    });
  };

  const handleLoadModelPath = async (path: string) => {
    setLoadErr(null);
    setLoadingModel(true);
    try {
      const st = await loadModel(path, true);
      setStudio(st);
      setView("chat");
    } catch (e) {
      setLoadErr(e instanceof Error ? e.message : String(e));
    } finally {
      setLoadingModel(false);
    }
  };

  const handlePickModelDialog = async () => {
    if (!isTauriRuntime()) return;
    const picked = await open({
      multiple: false,
      filters: [{ name: "VXT or GGUF LLM", extensions: ["vxt", "gguf"] }],
    });
    if (typeof picked === "string") {
      await handleLoadModelPath(picked);
    }
  };

  const handleUnloadModel = async () => {
    setLoadErr(null);
    setLoadingModel(true);
    try {
      const st = await unloadModel();
      setStudio(st);
    } catch (e) {
      setLoadErr(e instanceof Error ? e.message : String(e));
    } finally {
      setLoadingModel(false);
    }
  };

  const activeThread = threads.find((t) => t.id === activeThreadId) ?? threads[0];
  const vxtCount = (studio?.discovered_files ?? []).filter((f) => f.kind === "VXT").length;

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-[#19191c] text-zinc-100">
      {/* ── Collapsible Left Sidebar (Claude Desktop / Codex Style) ─────────── */}
      {!sidebarCollapsed && (
        <aside className="flex h-full w-[260px] shrink-0 flex-col border-r border-white/[0.06] bg-[#131316]">
          {/* Brand Row */}
          <div className="flex h-[54px] shrink-0 items-center justify-between px-3.5">
            <div className="flex items-center gap-2.5">
              <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-emerald-500/15 text-xs font-bold text-emerald-400 border border-emerald-500/25">
                VX
              </div>
              <span className="text-sm font-semibold tracking-tight text-zinc-100">
                VXT Studio
              </span>
            </div>
            <button
              type="button"
              onClick={() => setSidebarCollapsed(true)}
              title="Kenar çubuğunu gizle"
              className="flex h-7 w-7 items-center justify-center rounded-lg text-zinc-500 transition hover:bg-white/[0.06] hover:text-zinc-200 cursor-pointer"
            >
              <PanelLeftClose className="h-4 w-4" />
            </button>
          </div>

          {/* Primary Action + Nav */}
          <div className="px-2.5 space-y-1">
            <button
              type="button"
              onClick={() => void handleNewChat()}
              className="flex w-full items-center justify-between rounded-xl border border-white/[0.08] bg-white/[0.04] px-3 py-2 text-xs font-medium text-zinc-100 transition hover:bg-white/[0.08] cursor-pointer"
            >
              <span className="flex items-center gap-2">
                <Plus className="h-3.5 w-3.5 text-emerald-400" />
                Yeni Sohbet
              </span>
              <span className="font-mono text-[10px] text-zinc-500">⌘N</span>
            </button>

            <div className="pt-2 space-y-0.5">
              <button
                type="button"
                onClick={() => setView("chat")}
                className={`flex w-full items-center gap-2.5 rounded-xl px-3 py-2 text-xs transition cursor-pointer ${
                  view === "chat"
                    ? "bg-white/[0.08] font-medium text-zinc-100"
                    : "text-zinc-400 hover:bg-white/[0.04] hover:text-zinc-200"
                }`}
              >
                <MessageSquare className="h-3.5 w-3.5" />
                <span>Sohbet</span>
              </button>

              <button
                type="button"
                onClick={() => setView("hub")}
                className={`flex w-full items-center justify-between rounded-xl px-3 py-2 text-xs transition cursor-pointer ${
                  view === "hub"
                    ? "bg-white/[0.08] font-medium text-zinc-100"
                    : "text-zinc-400 hover:bg-white/[0.04] hover:text-zinc-200"
                }`}
              >
                <span className="flex items-center gap-2.5">
                  <CloudDownload className="h-3.5 w-3.5" />
                  <span>Modeller &amp; Hub</span>
                </span>
                <span className="rounded-md bg-white/[0.07] px-1.5 py-0.5 font-mono text-[10px] text-zinc-400">
                  {vxtCount}
                </span>
              </button>

              <button
                type="button"
                onClick={() => setView("api")}
                className={`flex w-full items-center gap-2.5 rounded-xl px-3 py-2 text-xs transition cursor-pointer ${
                  view === "api"
                    ? "bg-white/[0.08] font-medium text-zinc-100"
                    : "text-zinc-400 hover:bg-white/[0.04] hover:text-zinc-200"
                }`}
              >
                <Globe className="h-3.5 w-3.5" />
                <span>API &amp; Telemetri</span>
              </button>
            </div>
          </div>

          {/* Recent Chat Threads */}
          <div className="mt-4 flex-1 overflow-y-auto px-2.5">
            <div className="px-2 pb-1.5 text-[10px] font-medium uppercase tracking-wider text-zinc-500">
              Son Sohbetler
            </div>
            <div className="space-y-0.5">
              {threads.map((t) => {
                const isCurrent = view === "chat" && t.id === activeThread.id;
                return (
                  <div
                    key={t.id}
                    className={`group flex items-center justify-between gap-1 rounded-xl px-2.5 py-1.5 text-xs transition ${
                      isCurrent
                        ? "bg-white/[0.07] text-zinc-100"
                        : "text-zinc-400 hover:bg-white/[0.04] hover:text-zinc-200"
                    }`}
                  >
                    <button
                      type="button"
                      onClick={() => {
                        setActiveThreadId(t.id);
                        setView("chat");
                      }}
                      className="min-w-0 flex-1 truncate text-left cursor-pointer"
                    >
                      {t.title}
                    </button>
                    {threads.length > 1 && (
                      <button
                        type="button"
                        onClick={() => handleDeleteThread(t.id)}
                        title="Sohbeti sil"
                        className="opacity-0 group-hover:opacity-100 flex h-5 w-5 shrink-0 items-center justify-center rounded-md text-zinc-500 hover:bg-red-500/15 hover:text-red-400 transition cursor-pointer"
                      >
                        <Trash2 className="h-3 w-3" />
                      </button>
                    )}
                  </div>
                );
              })}
            </div>
          </div>

          {/* Background Download / Pack Progress Pill */}
          {forgeBusy && (
            <div className="px-2.5 pb-2">
              <button
                type="button"
                onClick={() => setView("hub")}
                className="w-full rounded-xl border border-emerald-500/30 bg-emerald-500/10 p-2.5 text-left transition hover:bg-emerald-500/15 cursor-pointer"
              >
                <div className="flex items-center justify-between text-[11px] font-medium text-emerald-300">
                  <span className="flex items-center gap-1.5 truncate">
                    <Loader2 className="h-3 w-3 shrink-0 animate-spin" />
                    <span className="truncate">{forgeJobName || "Model hazırlanıyor"}</span>
                  </span>
                  <span className="font-mono ml-1">
                    {(forgeProgress?.percent ?? 0).toFixed(0)}%
                  </span>
                </div>
                <div className="mt-1.5 h-1 w-full overflow-hidden rounded-full bg-white/[0.1]">
                  <div
                    className="h-full rounded-full bg-emerald-400 transition-all"
                    style={{ width: `${Math.max(4, forgeProgress?.percent ?? 4)}%` }}
                  />
                </div>
              </button>
            </div>
          )}

          {/* Bottom GPU Status Dock */}
          <div className="border-t border-white/[0.06] p-2.5">
            <div className="flex items-center justify-between gap-2 rounded-xl bg-white/[0.03] px-3 py-2">
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-1.5 text-[11px] font-medium text-zinc-200">
                  <span
                    className={`h-1.5 w-1.5 rounded-full shrink-0 ${
                      studio?.model
                        ? "bg-emerald-400"
                        : studio?.busy
                          ? "bg-amber-400 animate-pulse"
                          : "bg-zinc-600"
                    }`}
                  />
                  <span className="truncate">
                    {studio?.model
                      ? studio.model.name
                      : studio?.busy
                        ? "GPU'ya yükleniyor..."
                        : "Model yüklü değil"}
                  </span>
                </div>
                <div className="mt-0.5 flex items-center gap-1.5 text-[10px] font-mono text-zinc-500">
                  <Cpu className="h-2.5 w-2.5" />
                  <span className="truncate">
                    {studio?.model
                      ? `${shortBackendLabel(studio.model.backend)} · ${studio.model.compressed_mib.toFixed(0)} MiB`
                      : "Metal / CUDA hazır"}
                  </span>
                </div>
              </div>
              {studio?.model && (
                <button
                  type="button"
                  disabled={loadingModel || studio.busy}
                  onClick={() => void handleUnloadModel()}
                  title="Modeli GPU belleğinden kaldır"
                  className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg text-zinc-500 transition hover:bg-red-500/15 hover:text-red-400 cursor-pointer"
                >
                  <PowerOff className="h-3.5 w-3.5" />
                </button>
              )}
            </div>
          </div>
        </aside>
      )}

      {/* ── Main Workspace Canvas ──────────────────────────────────────────── */}
      <main className="relative flex min-w-0 flex-1 flex-col overflow-hidden">
        {loadErr && (
          <div className="flex items-center justify-between border-b border-red-500/30 bg-red-500/10 px-4 py-2 text-xs text-red-300">
            <span>{loadErr}</span>
            <button
              type="button"
              onClick={() => setLoadErr(null)}
              className="text-red-200 hover:underline cursor-pointer"
            >
              Kapat
            </button>
          </div>
        )}

        <div className={view === "chat" ? "flex-1 min-h-0" : "hidden"}>
          <ChatWorkspace
            key={activeThread.id}
            threadId={activeThread.id}
            initialMessages={activeThread.messages}
            onMessagesChange={handleMessagesChange}
            studio={studio}
            loadingModel={loadingModel}
            sidebarCollapsed={sidebarCollapsed}
            onToggleSidebar={() => setSidebarCollapsed(false)}
            onLoadModelPath={handleLoadModelPath}
            onUnloadModel={handleUnloadModel}
            onPickModelDialog={handlePickModelDialog}
            onRefresh={refresh}
            onSwitchToHub={() => setView("hub")}
          />
        </div>

        <div className={view === "hub" ? "flex-1 min-h-0" : "hidden"}>
          <ForgeWorkspace
            discovered={studio?.discovered_files ?? []}
            activeModelPath={studio?.model?.path}
            sidebarCollapsed={sidebarCollapsed}
            onToggleSidebar={() => setSidebarCollapsed(false)}
            busy={forgeBusy}
            activeJobRepo={forgeJobName}
            progress={forgeProgress}
            onStartJob={(name) => {
              setForgeBusy(true);
              setForgeJobName(name);
              setForgeProgress(null);
            }}
            onFinishJob={() => {
              setForgeBusy(false);
            }}
            onLoadIntoChat={handleLoadModelPath}
            onRefresh={refresh}
          />
        </div>

        <div className={view === "api" ? "flex-1 min-h-0" : "hidden"}>
          <ApiHubWorkspace
            studio={studio}
            sidebarCollapsed={sidebarCollapsed}
            onToggleSidebar={() => setSidebarCollapsed(false)}
          />
        </div>
      </main>
    </div>
  );
}
