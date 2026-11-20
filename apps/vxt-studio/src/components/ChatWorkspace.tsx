import * as React from "react";
import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import { useChat } from "@ai-sdk/react";
import { convertToCoreMessages, streamText, type Message } from "ai";
import {
  ArrowUp,
  Brain,
  Check,
  ChevronDown,
  ChevronRight,
  CloudDownload,
  Copy,
  Cpu,
  Eye,
  FolderOpen,
  ImagePlus,
  Loader2,
  PanelLeft,
  PowerOff,
  RefreshCw,
  RotateCcw,
  SlidersHorizontal,
  Sparkles,
  Square,
  Zap,
} from "lucide-react";
import { MarkdownRenderer } from "@/components/MarkdownRenderer";
import {
  MessageImages,
  readImageFiles,
  VisionPreviewStrip,
  type VisionAttachment,
} from "@/components/VisionAttachments";
import { Input, Slider, Switch, Textarea } from "@/components/ui/primitives";
import {
  abortGeneration,
  resetChatCache,
  type DiscoveredFile,
  type StudioState,
} from "@/lib/vxt-bridge";

const STARTER_PROMPTS = [
  {
    title: "Sıkıştırılmış GPU Mimarisi",
    subtitle: "E8-HQ & bellek bant genişliği",
    text: "Sıkıştırılmış E8-HQ (.vxt) ağırlıkları üzerinde doğrudan GPU GEMV çalıştırmak bellek bant genişliği darboğazını nasıl çözer? 3 maddede özetle.",
  },
  {
    title: "Rust Arena Allocator",
    subtitle: "64-byte hizalı bellek yöneticisi",
    text: "Write a clean, idiomatic Rust bump-arena allocator struct with 64-byte alignment in ~30 lines of code, with a short explanation.",
  },
  {
    title: "RTX 5090 Teorik Hız",
    subtitle: "Bant genişliği & tok/s hesabı",
    text: "Bir model 18.4 GB BF16 yerine 7.0 GB VXT-Turbo-HQ (Q6) olarak saklanıyor ve RTX 5090'da 1.79 TB/s bant genişliğiyle okunuyor. Teorik maksimum decode hızı kaç tok/s olur? Formülüyle göster.",
  },
  {
    title: "MoE + Vision Analizi",
    subtitle: "Multimodal mimari karşılaştırması",
    text: "Why do Mixture-of-Experts (MoE) vision-language models benefit disproportionately from E8 lattice quantization and streaming disk packing? Answer with a concise comparison table.",
  },
];

function shortBackendLabel(backend?: string): string {
  if (!backend) return "GPU";
  if (backend.includes("CUDA") && backend.includes("Resident")) return "CUDA Resident";
  if (backend.includes("CUDA")) return "NVIDIA CUDA";
  if (backend.includes("VXT-Turbo")) return "Metal VXT-Turbo";
  if (backend.includes("CB4-Fast")) return "Metal CB4-Fast";
  if (backend.includes("Metal")) return "Apple Metal";
  return backend.split("(")[0].trim() || "CPU";
}

function splitThinking(text: string): { think: string | null; answer: string; open: boolean } {
  const trimmed = text.trimStart();
  if (!trimmed.startsWith("<think>")) {
    return { think: null, answer: text, open: false };
  }
  const endIdx = trimmed.indexOf("</think>");
  if (endIdx === -1) {
    return {
      think: trimmed.slice("<think>".length).trim(),
      answer: "",
      open: true,
    };
  }
  return {
    think: trimmed.slice("<think>".length, endIdx).trim(),
    answer: trimmed.slice(endIdx + "</think>".length).trimStart(),
    open: false,
  };
}

function AssistantMessage({
  content,
  isLast,
  isStreaming,
  metricsText,
  onRegenerate,
}: {
  content: string;
  isLast: boolean;
  isStreaming: boolean;
  metricsText: string | null;
  onRegenerate: () => void;
}) {
  const { think, answer, open } = splitThinking(content);
  const [thinkOpen, setThinkOpen] = React.useState(false);
  const [copied, setCopied] = React.useState(false);
  const showThink = thinkOpen || open;
  const visibleAnswer = answer || (open ? "" : content);

  const handleCopy = () => {
    void navigator.clipboard.writeText(visibleAnswer || content);
    setCopied(true);
    setTimeout(() => setCopied(false), 1600);
  };

  return (
    <div className="group space-y-2.5">
      {think !== null && (
        <div className="rounded-xl border border-white/[0.07] bg-white/[0.02] px-3.5 py-2.5">
          <button
            type="button"
            onClick={() => setThinkOpen((v) => !v)}
            className="flex w-full items-center gap-2 text-xs font-medium text-zinc-400 transition hover:text-zinc-200 cursor-pointer"
          >
            {showThink ? (
              <ChevronDown className="h-3.5 w-3.5 text-zinc-500" />
            ) : (
              <ChevronRight className="h-3.5 w-3.5 text-zinc-500" />
            )}
            <Brain className={`h-3.5 w-3.5 ${open ? "text-emerald-400 animate-pulse" : "text-zinc-400"}`} />
            <span>{open ? "Düşünüyor..." : "Düşünme Süreci"}</span>
          </button>
          {showThink && think.length > 0 && (
            <div className="mt-2.5 border-l border-white/[0.08] pl-3 text-xs leading-relaxed text-zinc-400 whitespace-pre-wrap font-mono">
              {think}
            </div>
          )}
        </div>
      )}

      {visibleAnswer ? (
        <MarkdownRenderer content={visibleAnswer} />
      ) : isStreaming && isLast ? (
        <div className="flex items-center gap-2 py-1 text-sm text-zinc-500">
          <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
          <span>Yanıt üretiliyor...</span>
        </div>
      ) : null}

      {!(isStreaming && isLast) && (visibleAnswer || content) && (
        <div className="flex items-center gap-1 pt-1 opacity-0 transition-opacity duration-150 group-hover:opacity-100 focus-within:opacity-100">
          <button
            type="button"
            onClick={handleCopy}
            className="flex items-center gap-1.5 rounded-lg px-2 py-1 text-[11px] text-zinc-400 transition hover:bg-white/[0.06] hover:text-zinc-200 cursor-pointer"
          >
            {copied ? (
              <>
                <Check className="h-3.5 w-3.5 text-emerald-400" />
                <span className="text-emerald-400">Kopyalandı</span>
              </>
            ) : (
              <>
                <Copy className="h-3.5 w-3.5" />
                <span>Kopyala</span>
              </>
            )}
          </button>
          {isLast && (
            <button
              type="button"
              onClick={onRegenerate}
              className="flex items-center gap-1.5 rounded-lg px-2 py-1 text-[11px] text-zinc-400 transition hover:bg-white/[0.06] hover:text-zinc-200 cursor-pointer"
            >
              <RefreshCw className="h-3.5 w-3.5" />
              <span>Yeniden Üret</span>
            </button>
          )}
          {isLast && metricsText && (
            <span className="ml-2 font-mono text-[11px] text-zinc-500">{metricsText}</span>
          )}
        </div>
      )}
    </div>
  );
}

export interface ChatWorkspaceProps {
  threadId: string;
  initialMessages: Message[];
  onMessagesChange: (threadId: string, messages: Message[]) => void;
  studio: StudioState | null;
  loadingModel: boolean;
  sidebarCollapsed: boolean;
  onToggleSidebar: () => void;
  onLoadModelPath: (path: string) => Promise<void>;
  onSetDraftModel: (path: string | null) => Promise<void>;
  onUnloadModel: () => Promise<void>;
  onPickModelDialog: () => Promise<void>;
  onRefresh: () => Promise<void>;
  onSwitchToHub: () => void;
}

export function ChatWorkspace({
  threadId,
  initialMessages,
  onMessagesChange,
  studio,
  loadingModel,
  sidebarCollapsed,
  onToggleSidebar,
  onLoadModelPath,
  onSetDraftModel,
  onUnloadModel,
  onPickModelDialog,
  onRefresh,
  onSwitchToHub,
}: ChatWorkspaceProps) {
  const [cb4Greedy, setCb4Greedy] = React.useState(true);
  const [temperature, setTemperature] = React.useState(0.6);
  const [maxTokens, setMaxTokens] = React.useState(1536);
  const [enableThinking, setEnableThinking] = React.useState(false);
  const [jsonMode, setJsonMode] = React.useState(false);
  const [stopSeqs, setStopSeqs] = React.useState("");
  const [systemPrompt, setSystemPrompt] = React.useState("");
  const [attachments, setAttachments] = React.useState<VisionAttachment[]>([]);
  const [dragOver, setDragOver] = React.useState(false);
  const [modelMenuOpen, setModelMenuOpen] = React.useState(false);
  const [settingsOpen, setSettingsOpen] = React.useState(false);

  const modelMenuRef = React.useRef<HTMLDivElement>(null);
  const settingsRef = React.useRef<HTMLDivElement>(null);
  const textareaRef = React.useRef<HTMLTextAreaElement>(null);
  const fileInputRef = React.useRef<HTMLInputElement>(null);

  React.useEffect(() => {
    const onDocClick = (e: MouseEvent) => {
      if (modelMenuRef.current && !modelMenuRef.current.contains(e.target as Node)) {
        setModelMenuOpen(false);
      }
      if (settingsRef.current && !settingsRef.current.contains(e.target as Node)) {
        setSettingsOpen(false);
      }
    };
    document.addEventListener("mousedown", onDocClick);
    return () => document.removeEventListener("mousedown", onDocClick);
  }, []);

  const baseUrl = studio?.base_url || "http://127.0.0.1:11435/v1";
  const modelId = studio?.model?.name || "vxt-local";
  const useFastGreedy = cb4Greedy && !enableThinking;
  const effectiveTemp = useFastGreedy ? 0.0 : temperature;

  const cfgRef = React.useRef({
    baseUrl,
    modelId,
    useFastGreedy,
    effectiveTemp,
    maxTokens,
    enableThinking,
    jsonMode,
    stopSeqs,
    systemPrompt,
  });
  cfgRef.current = {
    baseUrl,
    modelId,
    useFastGreedy,
    effectiveTemp,
    maxTokens,
    enableThinking,
    jsonMode,
    stopSeqs,
    systemPrompt,
  };

  const localAiSdkFetch = React.useCallback(
    async (_input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
      const cfg = cfgRef.current;
      const bodyObj = init?.body ? JSON.parse(String(init.body)) : {};
      const uiMsgs: Array<{ role?: string; experimental_attachments?: VisionAttachment[] }> =
        Array.isArray(bodyObj.messages) ? bodyObj.messages : [];
      const vxtProvider = createOpenAICompatible({
        name: "vxt-local",
        baseURL: cfg.baseUrl,
        apiKey: "vxt-local",
        fetch: async (url, reqInit) => {
          if (reqInit?.body && typeof reqInit.body === "string") {
            const payload = JSON.parse(reqInit.body);
            payload.enable_thinking = cfg.enableThinking;
            if (cfg.jsonMode) {
              payload.response_format = { type: "json_object" };
            }
            const stops = cfg.stopSeqs
              .split(",")
              .map((s) => s.trim())
              .filter(Boolean);
            if (stops.length > 0) {
              payload.stop = stops;
            }
            if (cfg.useFastGreedy) {
              payload.presence_penalty = 0.0;
            } else {
              delete payload.presence_penalty;
            }
            if (Array.isArray(payload.messages)) {
              const offset = payload.messages[0]?.role === "system" ? 1 : 0;
              uiMsgs.forEach((um, idx) => {
                const atts = (um.experimental_attachments ?? []).filter((a) =>
                  a.url?.startsWith("data:image/"),
                );
                const target = payload.messages[idx + offset];
                if (target && atts.length > 0) {
                  const hasImgPart =
                    Array.isArray(target.content) &&
                    target.content.some((p: { type?: string }) => p.type === "image_url");
                  if (!hasImgPart) {
                    target.images = atts.map((a) => a.url);
                  }
                }
              });
            }
            reqInit = { ...reqInit, body: JSON.stringify(payload) };
          }
          return globalThis.fetch(url, reqInit);
        },
      });

      const result = streamText({
        model: vxtProvider(cfg.modelId),
        system: cfg.systemPrompt.trim() || undefined,
        messages: convertToCoreMessages(bodyObj.messages ?? []),
        temperature: cfg.effectiveTemp,
        maxTokens: cfg.maxTokens,
        abortSignal: init?.signal ?? undefined,
      });

      return result.toDataStreamResponse();
    },
    [],
  );

  const {
    messages,
    input,
    setInput,
    handleInputChange,
    append,
    reload,
    stop,
    setMessages,
    status,
    error,
  } = useChat({
    id: threadId,
    initialMessages,
    api: "/api/vxt-local-chat",
    fetch: localAiSdkFetch,
    onFinish: () => {
      void onRefresh();
    },
  });

  // Persist thread messages upward whenever messages settle or change
  React.useEffect(() => {
    onMessagesChange(threadId, messages);
  }, [threadId, messages, onMessagesChange]);

  const isStreaming = status === "submitted" || status === "streaming";
  const scrollRef = React.useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 180;
    if (nearBottom || isStreaming) {
      el.scrollTop = el.scrollHeight;
    }
  }, [messages, status, isStreaming]);

  // Auto-grow composer textarea
  React.useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 180)}px`;
  }, [input]);

  const handleStop = async () => {
    stop();
    await abortGeneration(baseUrl);
    await onRefresh();
  };

  const handleResetKv = async () => {
    stop();
    setMessages([]);
    setAttachments([]);
    await resetChatCache(baseUrl);
    await onRefresh();
  };

  const loaded = studio?.model;
  const metrics = studio?.last_metrics;
  const visionSupported = Boolean(loaded?.vision);
  const runnableModels: DiscoveredFile[] = (studio?.discovered_files ?? []).filter(
    (f) => f.runnable_hint && f.kind !== "HF-DIR",
  );

  const addImageFiles = async (files: FileList | File[]) => {
    if (!visionSupported) return;
    const next = await readImageFiles(files);
    if (next.length > 0) {
      setAttachments((prev) => [...prev, ...next].slice(0, 4));
    }
  };

  const sendMessage = () => {
    if (!loaded || isStreaming) return;
    const text = input.trim() || (attachments.length > 0 ? "Bu görseli detaylıca analiz et." : "");
    if (!text) return;
    const currentAtts = attachments;
    setAttachments([]);
    setInput("");
    void append(
      { role: "user", content: text },
      currentAtts.length > 0 ? { experimental_attachments: currentAtts } : undefined,
    );
  };

  const metricsSummary = metrics
    ? `${metrics.decode_tps.toFixed(1)} tok/s · ${metrics.generated_tokens} tok · ${metrics.stream_gbs.toFixed(0)} GB/s`
    : null;

  return (
    <div
      className="relative flex h-full flex-col bg-[#19191c]"
      onDragOver={(e) => {
        if (visionSupported && e.dataTransfer?.types?.includes("Files")) {
          e.preventDefault();
          setDragOver(true);
        }
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragOver(false);
        if (visionSupported && e.dataTransfer?.files?.length) {
          void addImageFiles(e.dataTransfer.files);
        }
      }}
    >
      {dragOver && (
        <div className="pointer-events-none absolute inset-3 z-40 flex items-center justify-center rounded-2xl border-2 border-dashed border-emerald-400/60 bg-emerald-500/10 backdrop-blur-sm">
          <div className="rounded-xl bg-[#141416]/90 px-5 py-3 text-sm font-medium text-emerald-300 shadow-xl">
            Görseli bırakın (Vision 3D M-RoPE)
          </div>
        </div>
      )}

      {/* ── Minimal Top Bar (Claude / Codex style) ─────────────────────────── */}
      <header className="flex h-[54px] shrink-0 items-center justify-between border-b border-white/[0.06] px-4">
        <div className="flex items-center gap-2">
          {sidebarCollapsed && (
            <button
              type="button"
              onClick={onToggleSidebar}
              title="Kenar çubuğunu aç"
              className="flex h-8 w-8 items-center justify-center rounded-lg text-zinc-400 transition hover:bg-white/[0.06] hover:text-zinc-100 cursor-pointer"
            >
              <PanelLeft className="h-4 w-4" />
            </button>
          )}

          {/* Interactive Model Switcher Pill */}
          <div className="relative" ref={modelMenuRef}>
            <button
              type="button"
              onClick={() => setModelMenuOpen((v) => !v)}
              className="flex items-center gap-2 rounded-xl px-2.5 py-1.5 text-sm font-medium text-zinc-100 transition hover:bg-white/[0.06] cursor-pointer"
            >
              {loadingModel || (studio?.busy && !loaded) ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin text-emerald-400" />
              ) : (
                <span
                  className={`h-2 w-2 rounded-full ${
                    loaded ? "bg-emerald-400" : "bg-zinc-600"
                  }`}
                />
              )}
              <span className="max-w-[260px] truncate">
                {loaded ? loaded.name : studio?.busy ? "Model GPU'ya yükleniyor..." : "Model Seç"}
              </span>
              {loaded && (
                <span className="hidden sm:inline-flex items-center gap-1 rounded-md bg-white/[0.06] px-1.5 py-0.5 text-[10px] font-mono text-zinc-400">
                  {shortBackendLabel(loaded.backend)}
                </span>
              )}
              {studio?.draft_model && (
                <span className="inline-flex items-center gap-1 rounded-md bg-emerald-500/15 px-1.5 py-0.5 text-[10px] font-mono text-emerald-300">
                  <Zap className="h-2.5 w-2.5" />
                  Spec
                </span>
              )}
              {visionSupported && (
                <span className="inline-flex items-center gap-1 rounded-md bg-sky-500/15 px-1.5 py-0.5 text-[10px] font-medium text-sky-300">
                  <Eye className="h-2.5 w-2.5" />
                  VLM
                </span>
              )}
              <ChevronDown className="h-3.5 w-3.5 text-zinc-500" />
            </button>

            {modelMenuOpen && (
              <div className="absolute left-0 top-full z-50 mt-1.5 w-[360px] overflow-hidden rounded-xl border border-white/[0.1] bg-[#1e1e22] shadow-2xl">
                <div className="border-b border-white/[0.07] px-3 py-2">
                  <div className="text-[11px] font-medium uppercase tracking-wider text-zinc-500">
                    Yerel Modeller (.vxt / .gguf)
                  </div>
                </div>
                <div className="max-h-[220px] overflow-y-auto p-1.5 space-y-0.5">
                  {runnableModels.length === 0 ? (
                    <div className="px-3 py-4 text-center text-xs text-zinc-500">
                      Henüz yerel .vxt modeli bulunamadı.
                    </div>
                  ) : (
                    runnableModels.map((m) => {
                      const active = loaded?.path === m.path;
                      return (
                        <button
                          key={m.path}
                          type="button"
                          disabled={loadingModel}
                          onClick={() => {
                            setModelMenuOpen(false);
                            void onLoadModelPath(m.path);
                          }}
                          className={`flex w-full items-center justify-between gap-2 rounded-lg px-2.5 py-2 text-left transition cursor-pointer ${
                            active
                              ? "bg-emerald-500/15 text-emerald-200"
                              : "text-zinc-200 hover:bg-white/[0.06]"
                          }`}
                        >
                          <div className="min-w-0">
                            <div className="truncate text-xs font-medium">{m.name}</div>
                            <div className="text-[10px] font-mono text-zinc-500">
                              {m.kind} · {m.size_mib} MiB
                            </div>
                          </div>
                          {active && <Check className="h-3.5 w-3.5 shrink-0 text-emerald-400" />}
                        </button>
                      );
                    })
                  )}
                </div>

                {/* Speculative Decoding Draft Model Selector */}
                {loaded && runnableModels.length > 1 && (
                  <div className="border-t border-white/[0.07] px-3 py-2.5 bg-[#1b1b1f] space-y-1.5">
                    <div className="flex items-center justify-between text-[11px]">
                      <span className="flex items-center gap-1.5 font-medium text-emerald-300">
                        <Zap className="h-3 w-3" />
                        Speculative Draft Model (2× Hız)
                      </span>
                      <span className="font-mono text-[10px] text-zinc-500">M=4 GPU</span>
                    </div>
                    <select
                      value={studio?.draft_model?.path || ""}
                      disabled={loadingModel}
                      onChange={(e) => {
                        setModelMenuOpen(false);
                        void onSetDraftModel(e.target.value ? e.target.value : null);
                      }}
                      className="h-8 w-full rounded-lg border border-white/[0.1] bg-[#141417] px-2 text-xs font-mono text-zinc-200 focus:outline-none"
                    >
                      <option value="">Kapalı (Yalnızca Hedef Model)</option>
                      {runnableModels
                        .filter((m) => m.path !== loaded.path)
                        .map((m) => (
                          <option key={m.path} value={m.path}>
                            {m.name} ({m.size_mib} MiB)
                          </option>
                        ))}
                    </select>
                  </div>
                )}
                <div className="border-t border-white/[0.07] p-1.5 space-y-0.5 bg-[#19191d]">
                  <button
                    type="button"
                    onClick={() => {
                      setModelMenuOpen(false);
                      onSwitchToHub();
                    }}
                    className="flex w-full items-center gap-2 rounded-lg px-2.5 py-2 text-xs text-emerald-300 transition hover:bg-emerald-500/10 cursor-pointer"
                  >
                    <CloudDownload className="h-3.5 w-3.5" />
                    <span>HuggingFace Hub&apos;dan Model İndir &amp; Dönüştür...</span>
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setModelMenuOpen(false);
                      void onPickModelDialog();
                    }}
                    className="flex w-full items-center gap-2 rounded-lg px-2.5 py-2 text-xs text-zinc-300 transition hover:bg-white/[0.06] cursor-pointer"
                  >
                    <FolderOpen className="h-3.5 w-3.5 text-zinc-400" />
                    <span>Diskten .vxt / .gguf Dosyası Aç...</span>
                  </button>
                  {loaded && (
                    <button
                      type="button"
                      onClick={() => {
                        setModelMenuOpen(false);
                        void onUnloadModel();
                      }}
                      className="flex w-full items-center gap-2 rounded-lg px-2.5 py-2 text-xs text-red-400/90 transition hover:bg-red-500/10 cursor-pointer"
                    >
                      <PowerOff className="h-3.5 w-3.5" />
                      <span>Modeli GPU Belleğinden Kaldır</span>
                    </button>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Right Header Controls */}
        <div className="flex items-center gap-2">
          {metrics && (
            <div
              className="hidden md:flex items-center gap-2 rounded-lg border border-white/[0.06] bg-white/[0.025] px-2.5 py-1 text-[11px] font-mono text-zinc-400"
              title={`Prefill: ${metrics.prefill_tps.toFixed(1)} tok/s · Path: ${metrics.decode_path}`}
            >
              <Zap className="h-3 w-3 text-emerald-400" />
              <span className="text-zinc-200">{metrics.decode_tps.toFixed(1)} tok/s</span>
              <span className="text-zinc-600">·</span>
              <span>{metrics.stream_gbs.toFixed(0)} GB/s</span>
            </div>
          )}

          {/* Parameters Popover */}
          <div className="relative" ref={settingsRef}>
            <button
              type="button"
              onClick={() => setSettingsOpen((v) => !v)}
              title="Çıkarım Ayarları"
              className={`flex h-8 items-center gap-1.5 rounded-lg px-2.5 text-xs transition cursor-pointer ${
                settingsOpen
                  ? "bg-white/[0.1] text-zinc-100"
                  : "text-zinc-400 hover:bg-white/[0.06] hover:text-zinc-100"
              }`}
            >
              <SlidersHorizontal className="h-3.5 w-3.5" />
              <span className="hidden sm:inline">Ayarlar</span>
            </button>

            {settingsOpen && (
              <div className="absolute right-0 top-full z-50 mt-1.5 w-[320px] rounded-xl border border-white/[0.1] bg-[#1e1e22] p-4 shadow-2xl space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold text-zinc-100">Çıkarım Parametreleri</span>
                  <button
                    type="button"
                    onClick={() => void handleResetKv()}
                    className="flex items-center gap-1 rounded-md px-2 py-1 text-[11px] text-zinc-400 transition hover:bg-white/[0.06] hover:text-zinc-200 cursor-pointer"
                  >
                    <RotateCcw className="h-3 w-3" />
                    KV Sıfırla ({studio?.cached_tokens ?? 0})
                  </button>
                </div>

                <div className="flex items-center justify-between gap-2 rounded-lg bg-white/[0.03] px-3 py-2">
                  <div>
                    <div className="text-xs font-medium text-zinc-200">Hızlı Greedy (GPU ArgMax)</div>
                    <div className="text-[10px] text-zinc-500">En yüksek tok/s · Deterministik</div>
                  </div>
                  <Switch
                    checked={useFastGreedy}
                    disabled={enableThinking}
                    onCheckedChange={setCb4Greedy}
                  />
                </div>

                <div className="space-y-1.5">
                  <div className="flex justify-between text-xs">
                    <span className="text-zinc-400">Temperature</span>
                    <span className="font-mono text-zinc-200">{effectiveTemp.toFixed(2)}</span>
                  </div>
                  <Slider
                    value={[temperature]}
                    min={0}
                    max={1.5}
                    step={0.05}
                    disabled={useFastGreedy}
                    onValueChange={([v]) => setTemperature(v)}
                  />
                </div>

                <div className="space-y-1.5">
                  <div className="flex justify-between text-xs">
                    <span className="text-zinc-400">Max Tokens</span>
                    <span className="font-mono text-zinc-200">{maxTokens}</span>
                  </div>
                  <Slider
                    value={[maxTokens]}
                    min={128}
                    max={4096}
                    step={128}
                    onValueChange={([v]) => setMaxTokens(v)}
                  />
                </div>

                <div className="grid grid-cols-2 gap-2">
                  <button
                    type="button"
                    onClick={() => setEnableThinking((v) => !v)}
                    className={`flex items-center justify-between rounded-lg border px-2.5 py-2 text-xs transition cursor-pointer ${
                      enableThinking
                        ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-300"
                        : "border-white/[0.07] bg-white/[0.02] text-zinc-400 hover:text-zinc-200"
                    }`}
                  >
                    <span>Thinking</span>
                    <span className="font-mono text-[10px]">{enableThinking ? "ON" : "OFF"}</span>
                  </button>
                  <button
                    type="button"
                    onClick={() => setJsonMode((v) => !v)}
                    className={`flex items-center justify-between rounded-lg border px-2.5 py-2 text-xs transition cursor-pointer ${
                      jsonMode
                        ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-300"
                        : "border-white/[0.07] bg-white/[0.02] text-zinc-400 hover:text-zinc-200"
                    }`}
                  >
                    <span>JSON Mode</span>
                    <span className="font-mono text-[10px]">{jsonMode ? "ON" : "OFF"}</span>
                  </button>
                </div>

                <div className="space-y-1">
                  <label className="text-[11px] text-zinc-400">Stop Sequences (virgülle)</label>
                  <Input
                    value={stopSeqs}
                    onChange={(e) => setStopSeqs(e.target.value)}
                    placeholder="örn. END, ```"
                    className="h-8 bg-[#151518] font-mono text-xs"
                  />
                </div>

                <div className="space-y-1">
                  <label className="text-[11px] text-zinc-400">System Prompt</label>
                  <Textarea
                    value={systemPrompt}
                    onChange={(e) => setSystemPrompt(e.target.value)}
                    placeholder="İsteğe bağlı sistem talimatı..."
                    className="min-h-[64px] bg-[#151518] text-xs"
                  />
                </div>

                {loaded && (
                  <div className="rounded-lg border border-white/[0.06] bg-[#151518] p-2.5 text-[11px] font-mono text-zinc-400 space-y-1">
                    <div className="flex justify-between">
                      <span>Mimari</span>
                      <span className="text-zinc-200">
                        {loaded.arch} ({loaded.n_layer}L)
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>Sıkıştırılmış Boyut</span>
                      <span className="text-zinc-200">{loaded.compressed_mib.toFixed(1)} MiB</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Yükleme</span>
                      <span className="text-zinc-200">{loaded.load_ms} ms</span>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </header>

      {/* ── Centered Editorial Message Stream ──────────────────────────────── */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 sm:px-6">
        <div className="mx-auto flex min-h-full max-w-[760px] flex-col py-8">
          {messages.length === 0 ? (
            <div className="my-auto flex flex-col items-center text-center py-6">
              <div className="mb-5 flex h-12 w-12 items-center justify-center rounded-2xl border border-white/[0.09] bg-gradient-to-b from-white/[0.07] to-white/[0.02] shadow-lg">
                <Sparkles className="h-5 w-5 text-emerald-400" />
              </div>

              <h1 className="text-2xl font-semibold tracking-tight text-zinc-100">
                {loaded ? loaded.name : "VXT Studio"}
              </h1>
              <p className="mt-2 max-w-md text-sm text-zinc-400 leading-relaxed">
                {loaded ? (
                  <>
                    <span className="text-emerald-400 font-medium">
                      {shortBackendLabel(loaded.backend)}
                    </span>{" "}
                    üzerinde doğrudan sıkıştırılmış bellekten çalışıyor ({loaded.compressed_mib.toFixed(0)} MiB
                    {visionSupported ? " · Vision 3D M-RoPE" : ""}).
                  </>
                ) : (
                  "HuggingFace Hub'dan tek tıkla model indirip .vxt formatına dönüştürün veya yerel bir model seçin."
                )}
              </p>

              {loaded ? (
                <div className="mt-8 grid w-full grid-cols-1 gap-2.5 sm:grid-cols-2">
                  {STARTER_PROMPTS.map((p) => (
                    <button
                      key={p.title}
                      type="button"
                      onClick={() => {
                        setInput(p.text);
                        textareaRef.current?.focus();
                      }}
                      className="group rounded-xl border border-white/[0.07] bg-white/[0.02] p-3.5 text-left transition hover:border-white/[0.15] hover:bg-white/[0.045] cursor-pointer"
                    >
                      <div className="text-[13px] font-medium text-zinc-200 group-hover:text-white">
                        {p.title}
                      </div>
                      <div className="mt-0.5 text-xs text-zinc-500 line-clamp-1">{p.subtitle}</div>
                    </button>
                  ))}
                </div>
              ) : (
                <div className="mt-7 flex flex-wrap items-center justify-center gap-2.5">
                  <button
                    type="button"
                    onClick={onSwitchToHub}
                    className="inline-flex items-center gap-2 rounded-xl bg-emerald-500 px-4 py-2.5 text-sm font-medium text-zinc-950 transition hover:bg-emerald-400 cursor-pointer"
                  >
                    <CloudDownload className="h-4 w-4" />
                    HuggingFace&apos;ten Model İndir
                  </button>
                  <button
                    type="button"
                    onClick={() => void onPickModelDialog()}
                    className="inline-flex items-center gap-2 rounded-xl border border-white/[0.1] bg-white/[0.04] px-4 py-2.5 text-sm font-medium text-zinc-200 transition hover:bg-white/[0.08] cursor-pointer"
                  >
                    <FolderOpen className="h-4 w-4" />
                    Yerel Model Aç
                  </button>
                </div>
              )}
            </div>
          ) : (
            <div className="space-y-7 pb-6">
              {messages.map((msg, idx) => {
                const isLast = idx === messages.length - 1;
                if (msg.role === "user") {
                  return (
                    <div key={msg.id} className="flex justify-end">
                      <div className="max-w-[82%] rounded-2xl bg-[#27272d] px-4 py-3 text-[15px] leading-relaxed text-zinc-100 shadow-sm">
                        <MessageImages attachments={msg.experimental_attachments} />
                        <div className="whitespace-pre-wrap">{msg.content}</div>
                      </div>
                    </div>
                  );
                }
                return (
                  <div key={msg.id} className="flex gap-3.5">
                    <div className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-lg border border-white/[0.08] bg-[#222227] text-[10px] font-bold text-emerald-400">
                      VX
                    </div>
                    <div className="min-w-0 flex-1">
                      <AssistantMessage
                        content={msg.content}
                        isLast={isLast}
                        isStreaming={isStreaming}
                        metricsText={isLast ? metricsSummary : null}
                        onRegenerate={() => {
                          if (!isStreaming) void reload();
                        }}
                      />
                    </div>
                  </div>
                );
              })}

              {isStreaming && messages[messages.length - 1]?.role === "user" && (
                <div className="flex gap-3.5">
                  <div className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-lg border border-white/[0.08] bg-[#222227] text-[10px] font-bold text-emerald-400">
                    VX
                  </div>
                  <div className="flex items-center gap-2 py-1.5 text-sm text-zinc-400">
                    <Loader2 className="h-3.5 w-3.5 animate-spin text-emerald-400" />
                    <span>GPU prefill...</span>
                  </div>
                </div>
              )}

              {error && (
                <div className="rounded-xl border border-red-500/30 bg-red-500/10 px-4 py-3 text-xs text-red-300">
                  {error.message}
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* ── Floating Claude / Codex Rounded Composer ───────────────────────── */}
      <div className="shrink-0 px-4 pb-4 pt-1 sm:px-6">
        <div className="mx-auto max-w-[760px]">
          <div className="rounded-2xl border border-white/[0.1] bg-[#222227] p-3 shadow-2xl transition focus-within:border-white/[0.2]">
            <VisionPreviewStrip
              attachments={attachments}
              onRemove={(idx) => setAttachments((prev) => prev.filter((_, i) => i !== idx))}
            />

            <textarea
              ref={textareaRef}
              value={input}
              onChange={handleInputChange}
              onPaste={(e) => {
                if (!visionSupported) return;
                const files = e.clipboardData?.files;
                if (files && files.length > 0) {
                  const imgs = Array.from(files).filter((f) => f.type.startsWith("image/"));
                  if (imgs.length > 0) {
                    e.preventDefault();
                    void addImageFiles(imgs);
                  }
                }
              }}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  sendMessage();
                }
              }}
              rows={1}
              placeholder={
                loaded
                  ? visionSupported
                    ? "Bir mesaj yazın veya görsel yapıştırın..."
                    : `${loaded.name} ile konuşun...`
                  : "Önce bir model seçin veya HuggingFace Hub'dan indirin..."
              }
              disabled={!loaded}
              className="w-full resize-none bg-transparent px-1.5 py-1 text-[15px] leading-relaxed text-zinc-100 placeholder:text-zinc-500 focus:outline-none disabled:opacity-50"
            />

            {/* Inline Composer Bottom Bar */}
            <div className="mt-2 flex items-center justify-between gap-2 pt-1">
              <div className="flex flex-wrap items-center gap-1.5">
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/png,image/jpeg,image/webp,image/bmp,image/gif"
                  multiple
                  className="hidden"
                  onChange={(e) => {
                    if (e.target.files && e.target.files.length > 0) {
                      void addImageFiles(e.target.files);
                      e.target.value = "";
                    }
                  }}
                />
                <button
                  type="button"
                  disabled={!loaded || isStreaming || !visionSupported}
                  onClick={() => fileInputRef.current?.click()}
                  title={
                    visionSupported
                      ? "Görsel ekle (PNG/JPG/WebP)"
                      : "Yüklü model Vision desteklemiyor"
                  }
                  className="flex h-7 items-center gap-1 rounded-lg px-2 text-xs text-zinc-400 transition hover:bg-white/[0.06] hover:text-zinc-200 disabled:opacity-35 cursor-pointer"
                >
                  <ImagePlus className="h-3.5 w-3.5" />
                  <span className="hidden sm:inline">Görsel</span>
                </button>

                <button
                  type="button"
                  onClick={() => setEnableThinking((v) => !v)}
                  className={`flex h-7 items-center gap-1 rounded-lg px-2 text-xs transition cursor-pointer ${
                    enableThinking
                      ? "bg-emerald-500/15 text-emerald-300"
                      : "text-zinc-400 hover:bg-white/[0.06] hover:text-zinc-200"
                  }`}
                >
                  <Brain className="h-3.5 w-3.5" />
                  <span>Think</span>
                </button>

                <button
                  type="button"
                  onClick={() => setJsonMode((v) => !v)}
                  className={`flex h-7 items-center gap-1 rounded-lg px-2 text-xs font-mono transition cursor-pointer ${
                    jsonMode
                      ? "bg-emerald-500/15 text-emerald-300"
                      : "text-zinc-400 hover:bg-white/[0.06] hover:text-zinc-200"
                  }`}
                >
                  <span>{"{}"}</span>
                  <span>JSON</span>
                </button>

                <button
                  type="button"
                  disabled={enableThinking}
                  onClick={() => setCb4Greedy((v) => !v)}
                  className={`flex h-7 items-center gap-1 rounded-lg px-2 text-xs transition cursor-pointer disabled:opacity-40 ${
                    useFastGreedy
                      ? "text-zinc-300 bg-white/[0.05]"
                      : "text-zinc-400 hover:bg-white/[0.06]"
                  }`}
                  title="Hızlı deterministik greedy vs temperature örnekleme"
                >
                  <Cpu className="h-3 w-3 text-emerald-400" />
                  <span>{useFastGreedy ? "Greedy" : `T=${temperature.toFixed(1)}`}</span>
                </button>
              </div>

              <div className="flex items-center gap-2">
                {(studio?.cached_tokens ?? 0) > 0 && (
                  <span className="hidden sm:inline text-[11px] font-mono text-zinc-500">
                    {studio?.cached_tokens} KV
                  </span>
                )}
                {isStreaming ? (
                  <button
                    type="button"
                    onClick={() => void handleStop()}
                    className="flex h-8 w-8 items-center justify-center rounded-full bg-zinc-100 text-zinc-950 transition hover:bg-white cursor-pointer"
                    title="Durdur"
                  >
                    <Square className="h-3.5 w-3.5 fill-current" />
                  </button>
                ) : (
                  <button
                    type="button"
                    disabled={!loaded || (!input.trim() && attachments.length === 0)}
                    onClick={sendMessage}
                    className="flex h-8 w-8 items-center justify-center rounded-full bg-zinc-100 text-zinc-950 transition hover:bg-white disabled:bg-white/[0.08] disabled:text-zinc-500 cursor-pointer"
                    title="Gönder (Enter)"
                  >
                    <ArrowUp className="h-4 w-4 stroke-[2.5]" />
                  </button>
                )}
              </div>
            </div>
          </div>
          <div className="mt-1.5 text-center text-[11px] text-zinc-600">
            Tamamen yerel · Sıfır bulut · OpenAI API: <span className="font-mono">{baseUrl}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
