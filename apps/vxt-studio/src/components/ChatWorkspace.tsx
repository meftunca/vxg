import * as React from "react";
import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import { useChat } from "@ai-sdk/react";
import { convertToCoreMessages, streamText } from "ai";
import {
  Activity,
  Brain,
  ChevronDown,
  ChevronRight,
  Cpu,
  Gauge,
  RotateCcw,
  Send,
  Sparkles,
  Square,
  Zap,
} from "lucide-react";
import {
  Badge,
  Button,
  Card,
  Input,
  Separator,
  Slider,
  Switch,
  Textarea,
} from "@/components/ui/primitives";
import {
  abortGeneration,
  resetChatCache,
  type StudioState,
} from "@/lib/vxt-bridge";

const DEMO_PROMPTS = [
  {
    label: "TR · Sıfır-Kopya Mimari",
    text: "Sıkıştırılmış ağırlıklar (.vxt) üzerinde doğrudan GPU GEMV çalıştırmak bellek bant genişliği darboğazını nasıl çözer? 3 maddede özetle.",
  },
  {
    label: "Rust · Arena Kod Örneği",
    text: "Write a clean, idiomatic Rust bump-arena allocator struct with 64-byte alignment in ~25 lines of code.",
  },
  {
    label: "Matematik · Hız Kazancı",
    text: "Bir model 3.60 GB BF16 yerine 0.99 GB VXT-CB4 olarak saklanıyor ve M2 Max'te 182 GB/s akış hızıyla okunuyor. Saniyede yaklaşık kaç token üretir?",
  },
  {
    label: "EN · MoE + Vision",
    text: "Why do Mixture-of-Experts (MoE) vision-language models benefit disproportionately from 4-bit codebook + sparse FP16 outlier compression on unified-memory laptops?",
  },
];

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

function AssistantBubble({ content }: { content: string }) {
  const { think, answer, open } = splitThinking(content);
  const [expanded, setExpanded] = React.useState(false);
  const showThink = expanded || open;

  return (
    <div className="space-y-2.5 text-sm leading-relaxed">
      {think !== null && (
        <div className="rounded-lg border border-emerald-500/25 bg-emerald-500/5 px-3 py-2">
          <button
            type="button"
            onClick={() => setExpanded((v) => !v)}
            className="flex w-full items-center gap-1.5 text-xs font-medium text-emerald-400 cursor-pointer"
          >
            {showThink ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronRight className="h-3.5 w-3.5" />}
            <Brain className="h-3.5 w-3.5" />
            <span>{open ? "Thinking..." : "Reasoning Trace"}</span>
          </button>
          {showThink && think.length > 0 && (
            <pre className="mt-2 whitespace-pre-wrap font-mono text-xs text-muted-foreground">
              {think}
            </pre>
          )}
        </div>
      )}
      <div className="whitespace-pre-wrap text-foreground/95">{answer || (open ? "" : content)}</div>
    </div>
  );
}

interface ChatWorkspaceProps {
  studio: StudioState | null;
  onRefresh: () => Promise<void>;
  onSwitchToForge: () => void;
}

export function ChatWorkspace({ studio, onRefresh, onSwitchToForge }: ChatWorkspaceProps) {
  const [cb4Greedy, setCb4Greedy] = React.useState(true);
  const [temperature, setTemperature] = React.useState(0.6);
  const [maxTokens, setMaxTokens] = React.useState(1024);
  const [enableThinking, setEnableThinking] = React.useState(false);
  const [systemPrompt, setSystemPrompt] = React.useState("");

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
    systemPrompt,
  });
  cfgRef.current = {
    baseUrl,
    modelId,
    useFastGreedy,
    effectiveTemp,
    maxTokens,
    enableThinking,
    systemPrompt,
  };

  const localAiSdkFetch = React.useCallback(
    async (_input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
      const cfg = cfgRef.current;
      const bodyObj = init?.body ? JSON.parse(String(init.body)) : {};
      const vxtProvider = createOpenAICompatible({
        name: "vxt-local",
        baseURL: cfg.baseUrl,
        apiKey: "vxt-local",
        fetch: async (url, reqInit) => {
          if (reqInit?.body && typeof reqInit.body === "string") {
            const payload = JSON.parse(reqInit.body);
            payload.enable_thinking = cfg.enableThinking;
            if (cfg.useFastGreedy) {
              payload.presence_penalty = 0.0;
            } else {
              delete payload.presence_penalty;
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
    handleInputChange,
    handleSubmit,
    append,
    stop,
    setMessages,
    status,
    error,
  } = useChat({
    api: "/api/vxt-local-chat",
    fetch: localAiSdkFetch,
    onFinish: () => {
      void onRefresh();
    },
  });

  const isStreaming = status === "submitted" || status === "streaming";
  const scrollRef = React.useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, status]);

  const handleStop = async () => {
    stop();
    await abortGeneration(baseUrl);
    await onRefresh();
  };

  const handleClear = async () => {
    stop();
    setMessages([]);
    await resetChatCache(baseUrl);
    await onRefresh();
  };

  const m = studio?.last_metrics;
  const loaded = studio?.model;

  return (
    <div className="grid h-[calc(100vh-88px)] grid-cols-1 gap-4 lg:grid-cols-[1fr_290px]">
      <Card className="flex h-full flex-col overflow-hidden border-border/80 bg-card/80">
        <div className="flex flex-wrap items-center justify-between gap-2 border-b px-4 py-2.5 bg-muted/30">
          <div className="flex flex-wrap items-center gap-2 text-xs">
            <Badge variant={loaded ? "default" : "secondary"} className="gap-1">
              <Zap className="h-3 w-3" />
              {loaded ? `${loaded.name} (${loaded.compressed_mib} MiB)` : "Model Yükleniyor / Seçilmedi"}
            </Badge>
            {m && (
              <>
                <Badge variant="outline" className="gap-1 font-mono">
                  <Gauge className="h-3 w-3 text-emerald-400" />
                  {m.decode_tps.toFixed(1)} tok/s
                </Badge>
                <Badge variant="outline" className="gap-1 font-mono">
                  <Activity className="h-3 w-3 text-cyan-400" />
                  {m.stream_gbs.toFixed(1)} GB/s
                </Badge>
                <Badge variant="outline" className="font-mono text-muted-foreground">
                  prefill {m.prefill_ms.toFixed(0)} ms ({m.cached_tokens} cache + {m.new_prefill_tokens} new)
                </Badge>
              </>
            )}
          </div>
          <Button variant="ghost" size="sm" onClick={handleClear} disabled={isStreaming}>
            <RotateCcw className="h-3.5 w-3.5" />
            Yeni Sohbet
          </Button>
        </div>

        <div ref={scrollRef} className="flex-1 space-y-4 overflow-y-auto p-4">
          {messages.length === 0 ? (
            <div className="flex h-full flex-col items-center justify-center text-center max-w-2xl mx-auto px-4">
              <div className="mb-3 flex h-12 w-12 items-center justify-center rounded-2xl bg-primary/15 text-primary">
                <Sparkles className="h-6 w-6" />
              </div>
              <h2 className="text-lg font-semibold tracking-tight">
                VXT Compressed-Domain AI Playground
              </h2>
              <p className="mt-1 text-xs text-muted-foreground max-w-lg">
                Vercel AI SDK (<code>useChat</code> + <code>streamText</code>) doğrudan cihazdaki{" "}
                <code>.vxt</code> Metal motoruna bağlanır. Ağırlıklar açılmadan (zero-decompression)
                GPU üzerinde çalıştırılır.
              </p>

              {!loaded && (
                <div className="mt-4">
                  <Button variant="outline" size="sm" onClick={onSwitchToForge}>
                    Tensor Forge Sekmesinden .vxt Üret veya Yükle
                  </Button>
                </div>
              )}

              <div className="mt-6 grid w-full grid-cols-1 gap-2.5 sm:grid-cols-2">
                {DEMO_PROMPTS.map((p) => (
                  <button
                    key={p.label}
                    type="button"
                    disabled={!loaded || isStreaming}
                    onClick={() => void append({ role: "user", content: p.text })}
                    className="rounded-xl border border-border/80 bg-background/60 p-3 text-left transition hover:border-primary/50 hover:bg-accent/40 disabled:opacity-50 cursor-pointer"
                  >
                    <div className="text-xs font-semibold text-primary">{p.label}</div>
                    <div className="mt-1 line-clamp-2 text-xs text-muted-foreground">{p.text}</div>
                  </button>
                ))}
              </div>
            </div>
          ) : (
            messages.map((msg) => (
              <div
                key={msg.id}
                className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
              >
                <div
                  className={`max-w-[85%] rounded-2xl px-4 py-3 ${
                    msg.role === "user"
                      ? "bg-primary text-primary-foreground font-medium text-sm"
                      : "border border-border/80 bg-background/80 text-foreground"
                  }`}
                >
                  {msg.role === "user" ? (
                    <div className="whitespace-pre-wrap">{msg.content}</div>
                  ) : (
                    <AssistantBubble content={msg.content} />
                  )}
                </div>
              </div>
            ))
          )}

          {error && (
            <div className="rounded-lg border border-red-500/40 bg-red-500/10 px-3 py-2 text-xs text-red-300">
              {error.message}
            </div>
          )}
        </div>

        <form
          onSubmit={(e) => {
            e.preventDefault();
            if (!input.trim() || !loaded || isStreaming) return;
            handleSubmit(e);
          }}
          className="border-t bg-background/50 p-3"
        >
          <div className="flex items-end gap-2">
            <Textarea
              value={input}
              onChange={handleInputChange}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  if (input.trim() && loaded && !isStreaming) {
                    handleSubmit(e);
                  }
                }
              }}
              placeholder={
                loaded
                  ? "Mesajınızı yazın... (Göndermek için Enter, yeni satır için Shift+Enter)"
                  : "Önce üst bardan bir .vxt / .gguf modeli yükleyin..."
              }
              className="min-h-[46px] max-h-[140px] resize-none bg-card"
            />
            {isStreaming ? (
              <Button type="button" variant="destructive" onClick={() => void handleStop()}>
                <Square className="h-4 w-4" />
                Durdur
              </Button>
            ) : (
              <Button type="submit" disabled={!loaded || !input.trim()}>
                <Send className="h-4 w-4" />
                Gönder
              </Button>
            )}
          </div>
        </form>
      </Card>

      <Card className="flex flex-col justify-between p-4 bg-card/80">
        <div className="space-y-4">
          <div>
            <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
              Çalışma Zamanı Ayarları
            </div>
            <Separator className="mt-2" />
          </div>

          <div className="flex items-center justify-between gap-2">
            <div>
              <div className="text-xs font-medium">CB4-Fast Greedy (0-ULP)</div>
              <div className="text-[11px] text-muted-foreground">
                Pipelined 2×8-tok Metal çekirdeği (~176 tok/s)
              </div>
            </div>
            <Switch
              checked={cb4Greedy}
              onCheckedChange={(checked) => {
                setCb4Greedy(checked);
                if (checked) setEnableThinking(false);
              }}
            />
          </div>

          {!useFastGreedy && (
            <div className="space-y-2">
              <div className="flex justify-between text-xs">
                <span>Temperature</span>
                <span className="font-mono text-muted-foreground">{temperature.toFixed(2)}</span>
              </div>
              <Slider
                value={[temperature]}
                min={0.05}
                max={1.2}
                step={0.05}
                onValueChange={([v]) => setTemperature(v)}
              />
            </div>
          )}

          <div className="flex items-center justify-between gap-2">
            <div>
              <div className="text-xs font-medium">Thinking (&lt;think&gt;)</div>
              <div className="text-[11px] text-muted-foreground">
                Adım adım muhakeme zincirini aç
              </div>
            </div>
            <Switch
              checked={enableThinking}
              onCheckedChange={(checked) => {
                setEnableThinking(checked);
                if (checked) {
                  setCb4Greedy(false);
                  setMaxTokens((prev) => Math.max(prev, 2048));
                }
              }}
            />
          </div>

          <div className="space-y-1.5">
            <div className="flex justify-between text-xs">
              <span>Maksimum Token</span>
              <span className="font-mono text-muted-foreground">{maxTokens}</span>
            </div>
            <Input
              type="number"
              min={32}
              max={4096}
              step={64}
              value={maxTokens}
              onChange={(e) => setMaxTokens(Number(e.target.value) || 512)}
            />
          </div>

          <div className="space-y-1.5">
            <div className="text-xs font-medium">Sistem İstemcisi (Opsiyonel)</div>
            <Textarea
              rows={3}
              value={systemPrompt}
              onChange={(e) => setSystemPrompt(e.target.value)}
              placeholder="Örn: Sen kısa ve net yanıt veren bir sistem mühendisisin."
              className="text-xs"
            />
          </div>
        </div>

        <div className="rounded-lg border border-border/70 bg-muted/30 p-3 text-[11px] space-y-1">
          <div className="flex items-center gap-1.5 font-medium text-foreground">
            <Cpu className="h-3.5 w-3.5 text-primary" />
            <span>KV + DeltaNet Prefix Cache</span>
          </div>
          <div className="text-muted-foreground">
            Önbellekteki token: <span className="font-mono text-foreground">{studio?.cached_tokens ?? 0}</span>
          </div>
          {loaded && (
            <div className="text-muted-foreground truncate" title={loaded.backend}>
              {loaded.backend}
            </div>
          )}
        </div>
      </Card>
    </div>
  );
}
