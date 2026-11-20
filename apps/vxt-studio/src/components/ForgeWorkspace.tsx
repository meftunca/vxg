import * as React from "react";
import { open, save } from "@tauri-apps/plugin-dialog";
import {
  ArrowRight,
  CheckCircle2,
  CloudDownload,
  Download,
  Eye,
  FileBox,
  FolderOpen,
  HardDrive,
  Heart,
  KeyRound,
  Layers,
  Loader2,
  Lock,
  PanelLeft,
  Play,
  RefreshCw,
  Search,
  Sparkles,
  Square,
  Trash2,
  Wand2,
} from "lucide-react";
import { Input, Switch } from "@/components/ui/primitives";
import {
  cancelForgeJob,
  convertToVxt,
  createDemoTensor,
  deleteLocalModel,
  downloadHfAndPack,
  inspectTensorFile,
  isTauriRuntime,
  searchHfModels,
  type ConvertReport,
  type DiscoveredFile,
  type ForgeProgressPayload,
  type HfSearchModel,
  type InspectReport,
} from "@/lib/vxt-bridge";

const PACK_MODES = [
  {
    id: "q6",
    label: "Q6 · E8-HQ",
    bpw: "~6.27 bpw",
    tag: "En İyi Kalite/Boyut",
    desc: "GGUF Q6_K'dan 0.43 GB daha küçük, PPL ve TR KLD (-%32) daha iyi · Metal & CUDA Resident",
  },
  {
    id: "q4",
    label: "Q4 · E8-HQ",
    bpw: "~4.50 bpw",
    tag: "Dengeli",
    desc: "GGUF Q4_K_M'yi ~0.9 bpw daha küçük boyutta geçer · Sıfır-OOM streaming paketleme",
  },
  {
    id: "q8",
    label: "Q8 · E8-HQ",
    bpw: "~7.71 bpw",
    tag: "Stüdyo",
    desc: "GGUF Q8_0'dan 0.89 GB daha küçük, TR KLD (-%55.6) daha isabetli",
  },
  {
    id: "q3",
    label: "Q3 · E8 v5",
    bpw: "~3.05 bpw",
    tag: "Ultra Küçük · 5.2×",
    desc: "En yüksek tok/s (RTX 5090 963 tok/s, M2 Max 300+ tok/s) · Düşük VRAM",
  },
  {
    id: "cb4",
    label: "CB4-v2",
    bpw: "~4.38 bpw",
    tag: "0-ULP Metal",
    desc: "Analitik super-weight kalibrasyonu + Metal CB4-Fast pipelined çekirdek",
  },
  {
    id: "lossless",
    label: "Lossless BF16",
    bpw: "~11.4 bpw",
    tag: "0-ULP Bit-Exact",
    desc: "%100 kayıpsız bit düzlemi sıkıştırması (ham BF16'ya göre 1.41× küçük)",
  },
];

const QUICK_FILTERS = [
  { label: "Qwen3.5", query: "Qwen/Qwen3.5" },
  { label: "Qwen3", query: "Qwen/Qwen3" },
  { label: "Qwen3 MoE", query: "Qwen3-30B-A3B" },
  { label: "Llama 3.2", query: "meta-llama/Llama-3.2" },
  { label: "Gemma", query: "google/gemma-3" },
];

const FEATURED_REPOS: HfSearchModel[] = [
  {
    id: "Qwen/Qwen3.5-0.8B",
    downloads: 0,
    likes: 0,
    pipeline_tag: "image-text-to-text",
    gated: false,
    vision: true,
    format_hint: "SafeTensors",
    arch_hint: "Qwen3.5",
  },
  {
    id: "Qwen/Qwen3.5-2B",
    downloads: 0,
    likes: 0,
    pipeline_tag: "image-text-to-text",
    gated: false,
    vision: true,
    format_hint: "SafeTensors",
    arch_hint: "Qwen3.5",
  },
  {
    id: "Qwen/Qwen3.5-4B",
    downloads: 0,
    likes: 0,
    pipeline_tag: "image-text-to-text",
    gated: false,
    vision: true,
    format_hint: "SafeTensors",
    arch_hint: "Qwen3.5",
  },
  {
    id: "Qwen/Qwen3.5-9B",
    downloads: 0,
    likes: 0,
    pipeline_tag: "image-text-to-text",
    gated: false,
    vision: true,
    format_hint: "SafeTensors",
    arch_hint: "Qwen3.5",
  },
  {
    id: "Qwen/Qwen3-30B-A3B",
    downloads: 0,
    likes: 0,
    pipeline_tag: "text-generation",
    gated: false,
    vision: false,
    format_hint: "SafeTensors",
    arch_hint: "Qwen3-MoE",
  },
];

function fmtCount(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return String(n);
}

function detectPresetBadge(filename: string): string | null {
  const l = filename.toLowerCase();
  if (l.includes("-q8") || l.includes("q8.")) return "Q8 · E8-HQ";
  if (l.includes("-q6") || l.includes("q6.") || l.includes("turbo-hq")) return "Q6 · E8-HQ";
  if (l.includes("-q4") || l.includes("q4.")) return "Q4 · E8-HQ";
  if (l.includes("-q3") || l.includes("q3.") || l.includes("turbo")) return "Q3 · E8";
  if (l.includes("cb4")) return "CB4-v2";
  if (l.includes("lossless")) return "Lossless";
  return null;
}

export interface ForgeWorkspaceProps {
  discovered: DiscoveredFile[];
  activeModelPath?: string | null;
  sidebarCollapsed: boolean;
  onToggleSidebar: () => void;
  busy: boolean;
  activeJobRepo: string | null;
  progress: ForgeProgressPayload | null;
  onStartJob: (repoOrName: string) => void;
  onFinishJob: () => void;
  onLoadIntoChat: (vxtPath: string) => Promise<void>;
  onRefresh: () => Promise<void>;
}

export function ForgeWorkspace({
  discovered,
  activeModelPath,
  sidebarCollapsed,
  onToggleSidebar,
  busy,
  activeJobRepo,
  progress,
  onStartJob,
  onFinishJob,
  onLoadIntoChat,
  onRefresh,
}: ForgeWorkspaceProps) {
  const [subtab, setSubtab] = React.useState<"hub" | "library" | "local">("hub");
  const [mode, setMode] = React.useState("q6");
  const [query, setQuery] = React.useState("Qwen/Qwen3.5");
  const [hfToken, setHfToken] = React.useState(() => localStorage.getItem("vxt_hf_token") || "");
  const [showToken, setShowToken] = React.useState(false);
  const [searching, setSearching] = React.useState(false);
  const [searchResults, setSearchResults] = React.useState<HfSearchModel[]>(FEATURED_REPOS);
  const [searchErr, setSearchErr] = React.useState<string | null>(null);

  const [inputPath, setInputPath] = React.useState("");
  const [outputPath, setOutputPath] = React.useState("");
  const [packAsLlm, setPackAsLlm] = React.useState(true);
  const [inspecting, setInspecting] = React.useState(false);
  const [report, setReport] = React.useState<InspectReport | null>(null);
  const [convResult, setConvResult] = React.useState<ConvertReport | null>(null);
  const [err, setErr] = React.useState<string | null>(null);
  const [deletingPath, setDeletingPath] = React.useState<string | null>(null);

  const vxtLibrary = React.useMemo(
    () => discovered.filter((f) => f.kind === "VXT"),
    [discovered],
  );
  const localSources = React.useMemo(
    () => discovered.filter((f) => f.kind !== "VXT"),
    [discovered],
  );

  const selectedPreset = PACK_MODES.find((m) => m.id === mode) ?? PACK_MODES[0];
  const looksLikeRepoId = /^[a-zA-Z0-9._-]+\/[a-zA-Z0-9._-]+$/.test(query.trim());

  const runSearch = React.useCallback(
    async (q: string) => {
      setSearching(true);
      setSearchErr(null);
      try {
        const res = await searchHfModels(q, hfToken);
        setSearchResults(res.length > 0 ? res : FEATURED_REPOS);
      } catch (e) {
        setSearchErr(e instanceof Error ? e.message : String(e));
      } finally {
        setSearching(false);
      }
    },
    [hfToken],
  );

  React.useEffect(() => {
    void runSearch("Qwen/Qwen3.5");
  }, [runSearch]);

  const handleSaveToken = (val: string) => {
    setHfToken(val);
    if (val.trim()) {
      localStorage.setItem("vxt_hf_token", val.trim());
    } else {
      localStorage.removeItem("vxt_hf_token");
    }
  };

  const handleConverted = async (res: ConvertReport) => {
    setConvResult(res);
    try {
      const outRep = await inspectTensorFile(res.output_path);
      setReport(outRep);
    } catch {
      // Ignore inspect error
    }
    await onRefresh();
  };

  const handleDownloadAndPack = async (repoId: string) => {
    if (!repoId.trim() || !isTauriRuntime() || busy) return;
    setErr(null);
    setConvResult(null);
    onStartJob(repoId.trim());
    try {
      const res = await downloadHfAndPack(repoId.trim(), mode, hfToken);
      await handleConverted(res);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      onFinishJob();
    }
  };

  const suggestOutputPath = (inPath: string, m: string) => {
    const trimmed = inPath.replace(/\/+$/, "");
    const base = trimmed.replace(/\.(gguf|safetensors|vxt)$/i, "");
    return `${base}-${m}.vxt`;
  };

  const handleSelectInput = async (path: string) => {
    setErr(null);
    setConvResult(null);
    setInputPath(path);
    if (!path.toLowerCase().endsWith(".vxt")) {
      setOutputPath(suggestOutputPath(path, mode));
    }
    if (!isTauriRuntime()) return;
    setInspecting(true);
    try {
      const rep = await inspectTensorFile(path);
      setReport(rep);
      setPackAsLlm(
        rep.runnable_llm &&
          (rep.format.startsWith("GGUF") || rep.format.startsWith("HuggingFace")),
      );
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setInspecting(false);
    }
  };

  const handleBrowseInput = async () => {
    if (!isTauriRuntime()) return;
    const picked = await open({
      multiple: false,
      filters: [{ name: "AI Tensors & Models", extensions: ["safetensors", "gguf", "vxt"] }],
    });
    if (typeof picked === "string") {
      await handleSelectInput(picked);
    }
  };

  const handleBrowseHfDir = async () => {
    if (!isTauriRuntime()) return;
    const picked = await open({
      directory: true,
      multiple: false,
    });
    if (typeof picked === "string") {
      await handleSelectInput(picked);
    }
  };

  const handleBrowseOutput = async () => {
    if (!isTauriRuntime()) return;
    const picked = await save({
      defaultPath: outputPath || `model-${mode}.vxt`,
      filters: [{ name: "VXT Compressed Archive", extensions: ["vxt"] }],
    });
    if (typeof picked === "string") {
      setOutputPath(picked);
    }
  };

  const handleGenerateDemoTensor = async () => {
    setErr(null);
    setInspecting(true);
    try {
      const rep = await createDemoTensor();
      setReport(rep);
      setInputPath(rep.path);
      setPackAsLlm(false);
      setOutputPath(suggestOutputPath(rep.path, mode));
      await onRefresh();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setInspecting(false);
    }
  };

  const handleConvertLocal = async () => {
    if (!inputPath || !outputPath || busy) return;
    setErr(null);
    setConvResult(null);
    const shortName = inputPath.split("/").pop() || inputPath;
    onStartJob(shortName);
    try {
      const res = await convertToVxt(inputPath, outputPath, mode, packAsLlm);
      await handleConverted(res);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      onFinishJob();
    }
  };

  const handleDelete = async (path: string) => {
    if (!isTauriRuntime()) return;
    setDeletingPath(path);
    setErr(null);
    try {
      await deleteLocalModel(path);
      await onRefresh();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setDeletingPath(null);
    }
  };

  return (
    <div className="flex h-full flex-col bg-[#19191c]">
      {/* ── Top Bar ────────────────────────────────────────────────────────── */}
      <header className="flex h-[54px] shrink-0 items-center justify-between border-b border-white/[0.06] px-5">
        <div className="flex items-center gap-3">
          {sidebarCollapsed && (
            <button
              type="button"
              onClick={onToggleSidebar}
              className="flex h-8 w-8 items-center justify-center rounded-lg text-zinc-400 transition hover:bg-white/[0.06] hover:text-zinc-100 cursor-pointer"
            >
              <PanelLeft className="h-4 w-4" />
            </button>
          )}
          <div className="flex items-center gap-2">
            <CloudDownload className="h-4 w-4 text-emerald-400" />
            <h1 className="text-sm font-semibold text-zinc-100">Modeller &amp; HuggingFace Hub</h1>
          </div>

          <div className="ml-3 flex items-center rounded-lg border border-white/[0.07] bg-[#141416] p-0.5 text-xs">
            <button
              type="button"
              onClick={() => setSubtab("hub")}
              className={`rounded-md px-3 py-1 transition cursor-pointer ${
                subtab === "hub"
                  ? "bg-white/[0.1] font-medium text-zinc-100"
                  : "text-zinc-400 hover:text-zinc-200"
              }`}
            >
              HuggingFace Hub
            </button>
            <button
              type="button"
              onClick={() => setSubtab("library")}
              className={`flex items-center gap-1.5 rounded-md px-3 py-1 transition cursor-pointer ${
                subtab === "library"
                  ? "bg-white/[0.1] font-medium text-zinc-100"
                  : "text-zinc-400 hover:text-zinc-200"
              }`}
            >
              <span>Kütüphanem</span>
              <span className="rounded bg-white/[0.08] px-1.5 py-0.2 text-[10px] font-mono">
                {vxtLibrary.length}
              </span>
            </button>
            <button
              type="button"
              onClick={() => setSubtab("local")}
              className={`rounded-md px-3 py-1 transition cursor-pointer ${
                subtab === "local"
                  ? "bg-white/[0.1] font-medium text-zinc-100"
                  : "text-zinc-400 hover:text-zinc-200"
              }`}
            >
              Yerel Dönüştürücü
            </button>
          </div>
        </div>

        <button
          type="button"
          onClick={() => setShowToken((v) => !v)}
          className={`flex items-center gap-1.5 rounded-lg border px-2.5 py-1.5 text-xs transition cursor-pointer ${
            hfToken
              ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-300"
              : "border-white/[0.08] bg-white/[0.03] text-zinc-400 hover:text-zinc-200"
          }`}
        >
          <KeyRound className="h-3.5 w-3.5" />
          <span>{hfToken ? "HF Token Kayıtlı" : "HF Token"}</span>
        </button>
      </header>

      {/* ── Main Content ───────────────────────────────────────────────────── */}
      <div className="flex-1 overflow-y-auto px-6 py-6">
        <div className="mx-auto max-w-5xl space-y-5">
          {/* Optional HF Token Bar */}
          {showToken && (
            <div className="flex flex-wrap items-center gap-3 rounded-xl border border-white/[0.08] bg-[#202025] px-4 py-3">
              <Lock className="h-4 w-4 text-zinc-400" />
              <div className="flex-1 min-w-[220px]">
                <Input
                  type="password"
                  value={hfToken}
                  onChange={(e) => handleSaveToken(e.target.value)}
                  placeholder="hf_... (Llama / Gemma gibi izinli gated modeller için)"
                  className="h-8 bg-[#151518] font-mono text-xs"
                />
              </div>
              <span className="text-[11px] text-zinc-500">
                Yalnızca yerel cihazınızda saklanır ve HuggingFace API isteklerinde kullanılır.
              </span>
            </div>
          )}

          {/* Active Download / Pack Progress Banner */}
          {(busy || progress) && (
            <div className="rounded-xl border border-emerald-500/30 bg-emerald-500/[0.06] p-4 space-y-2.5">
              <div className="flex items-center justify-between gap-3">
                <div className="flex items-center gap-2.5 min-w-0">
                  <Loader2 className="h-4 w-4 shrink-0 animate-spin text-emerald-400" />
                  <span className="rounded-md bg-emerald-500/20 px-2 py-0.5 text-[11px] font-medium text-emerald-300">
                    {progress?.phase === "download" ? "HuggingFace İndiriliyor" : ".vxt Paketleniyor"}
                  </span>
                  <span className="truncate text-xs font-mono text-zinc-200">
                    {activeJobRepo || ""}
                  </span>
                </div>
                <div className="flex items-center gap-3 shrink-0">
                  <span className="font-mono text-xs font-medium text-emerald-300">
                    {(progress?.percent ?? 0).toFixed(1)}%
                  </span>
                  {busy && (
                    <button
                      type="button"
                      onClick={() => void cancelForgeJob()}
                      className="flex items-center gap-1 rounded-lg border border-red-500/30 bg-red-500/10 px-2.5 py-1 text-[11px] text-red-300 transition hover:bg-red-500/20 cursor-pointer"
                    >
                      <Square className="h-3 w-3 fill-current" />
                      İptal
                    </button>
                  )}
                </div>
              </div>
              <div className="h-1.5 w-full overflow-hidden rounded-full bg-white/[0.08]">
                <div
                  className="h-full rounded-full bg-emerald-400 transition-all duration-200"
                  style={{ width: `${Math.max(2, progress?.percent ?? 4)}%` }}
                />
              </div>
              <div className="text-[11px] font-mono text-zinc-400 truncate">
                {progress?.detail || "Başlatılıyor..."}
              </div>
            </div>
          )}

          {/* Error Banner */}
          {err && (
            <div className="rounded-xl border border-red-500/30 bg-red-500/10 px-4 py-3 text-xs text-red-300">
              {err}
            </div>
          )}

          {/* Success Conversion Banner */}
          {convResult && (
            <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-emerald-500/30 bg-emerald-500/10 px-4 py-3.5">
              <div className="space-y-0.5">
                <div className="flex items-center gap-2 text-sm font-medium text-emerald-300">
                  <CheckCircle2 className="h-4 w-4" />
                  <span>
                    .vxt Hazır — {convResult.input_mib} MiB → {convResult.output_mib} MiB (
                    {convResult.ratio}× sıkıştırma · {convResult.elapsed_s}s)
                  </span>
                </div>
                <div className="font-mono text-[11px] text-zinc-400 break-all">
                  {convResult.output_path}
                </div>
              </div>
              {convResult.runnable_llm && (
                <button
                  type="button"
                  onClick={() => void onLoadIntoChat(convResult.output_path)}
                  className="inline-flex items-center gap-1.5 rounded-xl bg-emerald-500 px-3.5 py-2 text-xs font-semibold text-zinc-950 transition hover:bg-emerald-400 cursor-pointer"
                >
                  <Play className="h-3.5 w-3.5 fill-current" />
                  Sohbette Aç
                </button>
              )}
            </div>
          )}

          {/* ── Preset Selector Strip (Shared across Hub & Local) ─────────── */}
          {subtab !== "library" && (
            <div className="rounded-2xl border border-white/[0.07] bg-[#202025] p-4 space-y-3">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="text-xs font-medium text-zinc-200">
                  Hedef Sıkıştırma Profili (`.vxt` Preset)
                </div>
                <div className="text-[11px] text-zinc-400">{selectedPreset.desc}</div>
              </div>
              <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-6">
                {PACK_MODES.map((m) => {
                  const active = mode === m.id;
                  return (
                    <button
                      key={m.id}
                      type="button"
                      disabled={busy}
                      onClick={() => {
                        setMode(m.id);
                        if (inputPath && !inputPath.toLowerCase().endsWith(".vxt")) {
                          setOutputPath(suggestOutputPath(inputPath, m.id));
                        }
                      }}
                      className={`rounded-xl border p-2.5 text-left transition cursor-pointer disabled:opacity-50 ${
                        active
                          ? "border-emerald-500/60 bg-emerald-500/12 text-zinc-100"
                          : "border-white/[0.07] bg-[#17171b] text-zinc-400 hover:border-white/[0.14] hover:text-zinc-200"
                      }`}
                    >
                      <div className="flex items-center justify-between gap-1">
                        <span className="text-xs font-semibold">{m.label}</span>
                      </div>
                      <div className="mt-1 font-mono text-[10px] text-emerald-400/90">
                        {m.bpw}
                      </div>
                      <div className="mt-0.5 text-[10px] text-zinc-500 truncate">{m.tag}</div>
                    </button>
                  );
                })}
              </div>
            </div>
          )}

          {/* ── SUBTAB 1: HUGGINGFACE HUB ─────────────────────────────────── */}
          {subtab === "hub" && (
            <div className="space-y-4">
              {/* Search Bar */}
              <form
                onSubmit={(e) => {
                  e.preventDefault();
                  void runSearch(query);
                }}
                className="flex flex-wrap gap-2"
              >
                <div className="relative flex-1 min-w-[260px]">
                  <Search className="pointer-events-none absolute left-3.5 top-1/2 h-4 w-4 -translate-y-1/2 text-zinc-500" />
                  <input
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="HuggingFace'te ara (örn. Qwen3.5, Llama-3.2) veya tam org/model-adi yazın..."
                    className="h-11 w-full rounded-xl border border-white/[0.1] bg-[#222227] pl-10 pr-4 text-sm text-zinc-100 placeholder:text-zinc-500 focus:border-white/[0.22] focus:outline-none"
                  />
                </div>
                <button
                  type="submit"
                  disabled={searching}
                  className="inline-flex h-11 items-center gap-2 rounded-xl border border-white/[0.1] bg-white/[0.05] px-4 text-xs font-medium text-zinc-200 transition hover:bg-white/[0.1] cursor-pointer"
                >
                  {searching ? (
                    <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  ) : (
                    <Search className="h-3.5 w-3.5" />
                  )}
                  Ara
                </button>
                {looksLikeRepoId && (
                  <button
                    type="button"
                    disabled={busy || !isTauriRuntime()}
                    onClick={() => void handleDownloadAndPack(query.trim())}
                    className="inline-flex h-11 items-center gap-2 rounded-xl bg-emerald-500 px-4 text-xs font-semibold text-zinc-950 transition hover:bg-emerald-400 disabled:opacity-50 cursor-pointer"
                  >
                    <Download className="h-3.5 w-3.5" />
                    Doğrudan İndir → {mode.toUpperCase()}
                  </button>
                )}
              </form>

              {/* Quick Filter Chips */}
              <div className="flex flex-wrap items-center gap-1.5">
                <span className="mr-1 text-[11px] text-zinc-500">Popüler:</span>
                {QUICK_FILTERS.map((f) => (
                  <button
                    key={f.label}
                    type="button"
                    onClick={() => {
                      setQuery(f.query);
                      void runSearch(f.query);
                    }}
                    className="rounded-lg border border-white/[0.07] bg-white/[0.03] px-2.5 py-1 text-[11px] text-zinc-300 transition hover:border-white/[0.16] hover:bg-white/[0.07] cursor-pointer"
                  >
                    {f.label}
                  </button>
                ))}
              </div>

              {searchErr && (
                <div className="rounded-xl border border-amber-500/30 bg-amber-500/10 px-4 py-2.5 text-xs text-amber-200">
                  Arama uyarısı: {searchErr} (Yine de yukarıya tam <code className="font-mono">org/model</code> yazıp doğrudan indirebilirsiniz.)
                </div>
              )}

              {/* Live Search Results */}
              <div className="overflow-hidden rounded-2xl border border-white/[0.07] bg-[#1e1e23] divide-y divide-white/[0.05]">
                {searchResults.map((m) => (
                  <div
                    key={m.id}
                    className="flex flex-wrap items-center justify-between gap-3 px-4 py-3.5 transition hover:bg-white/[0.025]"
                  >
                    <div className="min-w-0 flex-1 space-y-1">
                      <div className="flex flex-wrap items-center gap-2">
                        <span className="truncate font-mono text-sm font-medium text-zinc-100">
                          {m.id}
                        </span>
                        <span className="rounded bg-white/[0.06] px-1.5 py-0.5 text-[10px] font-mono text-zinc-400">
                          {m.format_hint}
                        </span>
                        {m.arch_hint && (
                          <span className="rounded bg-white/[0.06] px-1.5 py-0.5 text-[10px] text-zinc-300">
                            {m.arch_hint}
                          </span>
                        )}
                        {m.vision && (
                          <span className="inline-flex items-center gap-1 rounded bg-sky-500/15 px-1.5 py-0.5 text-[10px] text-sky-300">
                            <Eye className="h-2.5 w-2.5" />
                            Vision
                          </span>
                        )}
                        {m.gated && (
                          <span className="inline-flex items-center gap-1 rounded bg-amber-500/15 px-1.5 py-0.5 text-[10px] text-amber-300">
                            <Lock className="h-2.5 w-2.5" />
                            Gated
                          </span>
                        )}
                      </div>
                      <div className="flex items-center gap-4 text-[11px] text-zinc-500">
                        {m.downloads > 0 && (
                          <span className="inline-flex items-center gap-1">
                            <Download className="h-3 w-3" />
                            {fmtCount(m.downloads)} indirme
                          </span>
                        )}
                        {m.likes > 0 && (
                          <span className="inline-flex items-center gap-1">
                            <Heart className="h-3 w-3" />
                            {fmtCount(m.likes)}
                          </span>
                        )}
                        <span>
                          Çıktı: <code className="text-zinc-400">~/Models/{m.id.split("/")[1]}-{mode}.vxt</code>
                        </span>
                      </div>
                    </div>

                    <button
                      type="button"
                      disabled={busy || !isTauriRuntime()}
                      onClick={() => void handleDownloadAndPack(m.id)}
                      className="inline-flex shrink-0 items-center gap-1.5 rounded-xl bg-zinc-100 px-3.5 py-2 text-xs font-semibold text-zinc-950 transition hover:bg-white disabled:bg-white/[0.08] disabled:text-zinc-500 cursor-pointer"
                    >
                      {busy && activeJobRepo === m.id ? (
                        <Loader2 className="h-3.5 w-3.5 animate-spin" />
                      ) : (
                        <CloudDownload className="h-3.5 w-3.5" />
                      )}
                      <span>İndir → {mode.toUpperCase()}</span>
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* ── SUBTAB 2: LOCAL .VXT LIBRARY ──────────────────────────────── */}
          {subtab === "library" && (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="text-xs text-zinc-400">
                  <code className="font-mono text-zinc-300">~/Models</code> ve yerel klasörlerde bulunan{" "}
                  <code className="font-mono text-zinc-300">.vxt</code> modelleri
                </div>
                <button
                  type="button"
                  onClick={() => void onRefresh()}
                  className="inline-flex items-center gap-1.5 rounded-lg border border-white/[0.08] bg-white/[0.03] px-3 py-1.5 text-xs text-zinc-300 transition hover:bg-white/[0.07] cursor-pointer"
                >
                  <RefreshCw className="h-3.5 w-3.5" />
                  Yenile
                </button>
              </div>

              {vxtLibrary.length === 0 ? (
                <div className="rounded-2xl border border-dashed border-white/[0.1] bg-white/[0.015] py-14 text-center space-y-3">
                  <HardDrive className="mx-auto h-8 w-8 text-zinc-600" />
                  <div className="text-sm font-medium text-zinc-300">Henüz yerel .vxt modeli yok</div>
                  <p className="text-xs text-zinc-500">
                    HuggingFace Hub sekmesinden tek tıkla bir model indirip dönüştürebilirsiniz.
                  </p>
                  <button
                    type="button"
                    onClick={() => setSubtab("hub")}
                    className="inline-flex items-center gap-2 rounded-xl bg-emerald-500 px-4 py-2 text-xs font-semibold text-zinc-950 transition hover:bg-emerald-400 cursor-pointer"
                  >
                    <CloudDownload className="h-3.5 w-3.5" />
                    HuggingFace Hub&apos;a Git
                  </button>
                </div>
              ) : (
                <div className="overflow-hidden rounded-2xl border border-white/[0.07] bg-[#1e1e23] divide-y divide-white/[0.05]">
                  {vxtLibrary.map((f) => {
                    const isLoaded = activeModelPath === f.path;
                    const preset = detectPresetBadge(f.name);
                    return (
                      <div
                        key={f.path}
                        className="flex flex-wrap items-center justify-between gap-3 px-4 py-3.5 transition hover:bg-white/[0.025]"
                      >
                        <div className="min-w-0 flex-1 space-y-1">
                          <div className="flex flex-wrap items-center gap-2">
                            <span
                              className={`h-2 w-2 rounded-full ${
                                isLoaded ? "bg-emerald-400" : "bg-zinc-600"
                              }`}
                            />
                            <span className="truncate font-mono text-sm font-medium text-zinc-100">
                              {f.name}
                            </span>
                            {preset && (
                              <span className="rounded bg-emerald-500/15 px-1.5 py-0.5 text-[10px] font-mono text-emerald-300">
                                {preset}
                              </span>
                            )}
                            {isLoaded && (
                              <span className="rounded bg-white/[0.08] px-1.5 py-0.5 text-[10px] text-zinc-200">
                                GPU&apos;da Yüklü
                              </span>
                            )}
                          </div>
                          <div className="flex items-center gap-3 text-[11px] font-mono text-zinc-500">
                            <span>{f.size_mib} MiB</span>
                            <span className="truncate">{f.path}</span>
                          </div>
                        </div>

                        <div className="flex items-center gap-1.5 shrink-0">
                          {f.runnable_hint && (
                            <button
                              type="button"
                              disabled={busy}
                              onClick={() => void onLoadIntoChat(f.path)}
                              className="inline-flex items-center gap-1.5 rounded-xl bg-zinc-100 px-3 py-1.5 text-xs font-semibold text-zinc-950 transition hover:bg-white cursor-pointer"
                            >
                              <Play className="h-3 w-3 fill-current" />
                              Sohbette Aç
                            </button>
                          )}
                          <button
                            type="button"
                            onClick={() => {
                              setSubtab("local");
                              void handleSelectInput(f.path);
                            }}
                            className="inline-flex items-center gap-1 rounded-xl border border-white/[0.08] bg-white/[0.03] px-2.5 py-1.5 text-xs text-zinc-300 transition hover:bg-white/[0.08] cursor-pointer"
                          >
                            <FileBox className="h-3.5 w-3.5" />
                            İncele
                          </button>
                          <button
                            type="button"
                            disabled={busy || deletingPath === f.path}
                            onClick={() => void handleDelete(f.path)}
                            title="Dosyayı diskten sil"
                            className="flex h-8 w-8 items-center justify-center rounded-xl text-zinc-500 transition hover:bg-red-500/15 hover:text-red-400 cursor-pointer"
                          >
                            {deletingPath === f.path ? (
                              <Loader2 className="h-3.5 w-3.5 animate-spin" />
                            ) : (
                              <Trash2 className="h-3.5 w-3.5" />
                            )}
                          </button>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          )}

          {/* ── SUBTAB 3: LOCAL FILE / FOLDER CONVERTER & INSPECTOR ───────── */}
          {subtab === "local" && (
            <div className="grid grid-cols-1 gap-5 lg:grid-cols-[400px_1fr]">
              <div className="rounded-2xl border border-white/[0.07] bg-[#1e1e23] p-4 space-y-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2 text-sm font-medium text-zinc-100">
                    <Wand2 className="h-4 w-4 text-emerald-400" />
                    Yerel Dönüştürücü
                  </div>
                  <button
                    type="button"
                    onClick={() => void handleGenerateDemoTensor()}
                    className="inline-flex items-center gap-1 rounded-lg border border-white/[0.08] bg-white/[0.03] px-2.5 py-1 text-[11px] text-zinc-300 transition hover:bg-white/[0.08] cursor-pointer"
                  >
                    <Sparkles className="h-3 w-3 text-emerald-400" />
                    Demo Üret
                  </button>
                </div>

                <div className="space-y-1.5">
                  <label className="text-[11px] text-zinc-400">
                    Kaynak (HF Klasörü / .gguf / .safetensors / .vxt)
                  </label>
                  <Input
                    value={inputPath}
                    onChange={(e) => setInputPath(e.target.value)}
                    placeholder="/Users/.../Qwen3.5-9B"
                    className="bg-[#151518] font-mono text-xs"
                  />
                  <div className="flex gap-2 pt-1">
                    <button
                      type="button"
                      onClick={() => void handleBrowseHfDir()}
                      className="flex-1 inline-flex items-center justify-center gap-1.5 rounded-lg border border-white/[0.08] bg-white/[0.04] py-1.5 text-xs text-zinc-200 transition hover:bg-white/[0.08] cursor-pointer"
                    >
                      <FolderOpen className="h-3.5 w-3.5" />
                      HF Klasörü Seç
                    </button>
                    <button
                      type="button"
                      onClick={() => void handleBrowseInput()}
                      className="flex-1 inline-flex items-center justify-center gap-1.5 rounded-lg border border-white/[0.08] bg-white/[0.04] py-1.5 text-xs text-zinc-200 transition hover:bg-white/[0.08] cursor-pointer"
                    >
                      <FileBox className="h-3.5 w-3.5" />
                      Dosya Seç
                    </button>
                  </div>
                </div>

                <div className="space-y-1.5">
                  <label className="text-[11px] text-zinc-400">Çıktı `.vxt` Yolu</label>
                  <div className="flex gap-2">
                    <Input
                      value={outputPath}
                      onChange={(e) => setOutputPath(e.target.value)}
                      placeholder="/Users/.../model-q6.vxt"
                      className="bg-[#151518] font-mono text-xs"
                    />
                    <button
                      type="button"
                      onClick={() => void handleBrowseOutput()}
                      className="shrink-0 rounded-lg border border-white/[0.08] bg-white/[0.04] px-3 text-xs text-zinc-300 transition hover:bg-white/[0.08] cursor-pointer"
                    >
                      Seç
                    </button>
                  </div>
                </div>

                <div className="flex items-center justify-between rounded-xl bg-[#151518] px-3 py-2.5">
                  <div>
                    <div className="text-xs font-medium text-zinc-200">Tam Çalıştırılabilir LLM/VLM</div>
                    <div className="text-[10px] text-zinc-500">Tokenizer + config + Vision gömülü</div>
                  </div>
                  <Switch checked={packAsLlm} onCheckedChange={setPackAsLlm} />
                </div>

                <button
                  type="button"
                  disabled={!inputPath || !outputPath || busy || inputPath.toLowerCase().endsWith(".vxt")}
                  onClick={() => void handleConvertLocal()}
                  className="flex w-full items-center justify-center gap-2 rounded-xl bg-emerald-500 py-2.5 text-xs font-semibold text-zinc-950 transition hover:bg-emerald-400 disabled:bg-white/[0.08] disabled:text-zinc-500 cursor-pointer"
                >
                  {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <ArrowRight className="h-4 w-4" />}
                  <span>.vxt Olarak Paketle ({mode.toUpperCase()})</span>
                </button>

                {localSources.length > 0 && (
                  <div className="space-y-1.5 pt-2 border-t border-white/[0.06]">
                    <div className="text-[11px] text-zinc-500">Keşfedilen Yerel HF / GGUF Kaynakları</div>
                    <div className="max-h-40 overflow-y-auto space-y-1">
                      {localSources.map((s) => (
                        <button
                          key={s.path}
                          type="button"
                          onClick={() => void handleSelectInput(s.path)}
                          className="flex w-full items-center justify-between gap-2 rounded-lg px-2.5 py-1.5 text-left text-xs text-zinc-300 transition hover:bg-white/[0.05] cursor-pointer"
                        >
                          <span className="truncate font-mono">{s.name}</span>
                          <span className="shrink-0 font-mono text-[10px] text-zinc-500">
                            {s.kind} · {s.size_mib} MiB
                          </span>
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              {/* Tensor Inspector */}
              <div className="rounded-2xl border border-white/[0.07] bg-[#1e1e23] p-4 flex flex-col min-h-[420px]">
                <div className="flex items-center justify-between border-b border-white/[0.06] pb-3">
                  <div className="flex items-center gap-2 text-sm font-medium text-zinc-100">
                    <Layers className="h-4 w-4 text-emerald-400" />
                    Tensör İnceleyici
                  </div>
                  {report && (
                    <span className="rounded-md bg-white/[0.06] px-2 py-0.5 font-mono text-[11px] text-zinc-300">
                      {report.format} · {report.tensor_count} tensör · {report.size_mib} MiB
                    </span>
                  )}
                </div>

                {inspecting ? (
                  <div className="m-auto flex items-center gap-2 text-xs text-zinc-400">
                    <Loader2 className="h-4 w-4 animate-spin text-emerald-400" />
                    Dosya inceleniyor...
                  </div>
                ) : report ? (
                  <div className="mt-3 flex-1 overflow-auto">
                    <table className="w-full text-left font-mono text-[11px]">
                      <thead className="sticky top-0 bg-[#1e1e23] text-zinc-500">
                        <tr className="border-b border-white/[0.07]">
                          <th className="py-2 pr-3 font-medium">Tensör</th>
                          <th className="py-2 pr-3 font-medium">Boyut</th>
                          <th className="py-2 pr-3 font-medium">Codec</th>
                          <th className="py-2 pr-3 text-right font-medium">BPW</th>
                          <th className="py-2 text-right font-medium">KiB</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-white/[0.04] text-zinc-300">
                        {report.tensors.slice(0, 120).map((t) => (
                          <tr key={t.index} className="hover:bg-white/[0.02]">
                            <td className="py-1.5 pr-3 max-w-[260px] truncate">{t.name}</td>
                            <td className="py-1.5 pr-3 text-zinc-400">{t.shape}</td>
                            <td className="py-1.5 pr-3 text-emerald-300/90">{t.codec_or_dtype}</td>
                            <td className="py-1.5 pr-3 text-right">{t.bpw.toFixed(2)}</td>
                            <td className="py-1.5 text-right text-zinc-400">{t.size_kib.toFixed(1)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <div className="m-auto text-center text-xs text-zinc-500">
                    İncelemek için soldan bir dosya seçin veya Kütüphanem&apos;den İncele&apos;ye tıklayın.
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
