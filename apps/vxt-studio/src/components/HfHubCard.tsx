import * as React from "react";
import {
  CloudDownload,
  HardDrive,
  KeyRound,
  Loader2,
  Play,
  Search,
  Square,
  Trash2,
} from "lucide-react";
import {
  Badge,
  Button,
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  Input,
  Separator,
} from "@/components/ui/primitives";
import {
  cancelForgeJob,
  deleteLocalModel,
  downloadHfAndPack,
  isTauriRuntime,
  type ConvertReport,
  type DiscoveredFile,
  type ForgeProgressPayload,
} from "@/lib/vxt-bridge";

const HF_PRESETS = [
  { repo: "Qwen/Qwen3.5-0.8B", tag: "Qwen3.5 · 0.8B" },
  { repo: "Qwen/Qwen3.5-4B", tag: "Qwen3.5 · 4B" },
  { repo: "Qwen/Qwen3.5-9B", tag: "Qwen3.5 · 9B" },
  { repo: "Qwen/Qwen3-30B-A3B", tag: "Qwen3 · 30B MoE" },
  { repo: "meta-llama/Llama-3.2-3B-Instruct", tag: "Llama-3.2 · 3B" },
];

interface HfHubCardProps {
  mode: string;
  busy: boolean;
  progress: ForgeProgressPayload | null;
  discovered: DiscoveredFile[];
  onBusyChange: (busy: boolean) => void;
  onProgressReset: () => void;
  onConverted: (res: ConvertReport) => Promise<void>;
  onError: (msg: string | null) => void;
  onInspectPath: (path: string) => Promise<void>;
  onLoadIntoChat: (path: string) => Promise<void>;
  onRefresh: () => Promise<void>;
}

export function HfHubCard({
  mode,
  busy,
  progress,
  discovered,
  onBusyChange,
  onProgressReset,
  onConverted,
  onError,
  onInspectPath,
  onLoadIntoChat,
  onRefresh,
}: HfHubCardProps) {
  const [repoId, setRepoId] = React.useState("Qwen/Qwen3.5-0.8B");
  const [hfToken, setHfToken] = React.useState("");
  const [showToken, setShowToken] = React.useState(false);
  const [deletingPath, setDeletingPath] = React.useState<string | null>(null);

  const vxtLibrary = React.useMemo(
    () => discovered.filter((f) => f.kind === "VXT"),
    [discovered],
  );

  const handleDownloadAndPack = async () => {
    if (!repoId.trim() || !isTauriRuntime()) return;
    onError(null);
    onProgressReset();
    onBusyChange(true);
    try {
      const res = await downloadHfAndPack(repoId.trim(), mode, hfToken);
      await onConverted(res);
    } catch (e) {
      onError(e instanceof Error ? e.message : String(e));
    } finally {
      onBusyChange(false);
    }
  };

  const handleDelete = async (path: string) => {
    if (!isTauriRuntime()) return;
    setDeletingPath(path);
    onError(null);
    try {
      await deleteLocalModel(path);
      await onRefresh();
    } catch (e) {
      onError(e instanceof Error ? e.message : String(e));
    } finally {
      setDeletingPath(null);
    }
  };

  return (
    <Card className="bg-card/80">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between gap-2">
          <CardTitle className="flex items-center gap-2 text-base">
            <CloudDownload className="h-4 w-4 text-primary" />
            Model Yöneticisi &amp; HuggingFace Hub
          </CardTitle>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setShowToken((v) => !v)}
            className="h-7 px-2 text-[11px]"
          >
            <KeyRound className="h-3 w-3" />
            {showToken ? "Token Gizle" : "HF Token"}
          </Button>
        </div>
        <CardDescription>
          HuggingFace Hub&apos;dan doğrudan indirip seçili modda (<code>{mode.toUpperCase()}</code>)
          otomatik <code>.vxt</code> arşivine paketleyin veya yerel kütüphaneyi yönetin.
        </CardDescription>
      </CardHeader>

      <CardContent className="space-y-3">
        <div className="flex flex-wrap gap-1.5">
          {HF_PRESETS.map((p) => (
            <button
              key={p.repo}
              type="button"
              disabled={busy}
              onClick={() => setRepoId(p.repo)}
              className={`rounded-md border px-2 py-1 text-[11px] font-mono transition cursor-pointer ${
                repoId === p.repo
                  ? "border-primary bg-primary/15 text-foreground"
                  : "border-border/70 bg-background/50 text-muted-foreground hover:border-border"
              }`}
            >
              {p.tag}
            </button>
          ))}
        </div>

        <div className="flex gap-2">
          <Input
            value={repoId}
            disabled={busy}
            onChange={(e) => setRepoId(e.target.value)}
            placeholder="org/model-adi (örn. Qwen/Qwen3-4B)"
            className="font-mono text-xs"
          />
          <Button
            size="sm"
            disabled={!repoId.trim() || busy || !isTauriRuntime()}
            onClick={() => void handleDownloadAndPack()}
          >
            {busy ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            ) : (
              <CloudDownload className="h-3.5 w-3.5" />
            )}
            İndir &amp; Paketle
          </Button>
        </div>

        {showToken && (
          <Input
            type="password"
            value={hfToken}
            disabled={busy}
            onChange={(e) => setHfToken(e.target.value)}
            placeholder="hf_... (Llama / Gemma gibi korumalı modeller için isteğe bağlı)"
            className="font-mono text-xs"
          />
        )}

        {busy && (
          <div className="space-y-2 rounded-lg border border-primary/40 bg-primary/5 p-3">
            <div className="flex items-center justify-between gap-2 text-xs">
              <div className="flex items-center gap-2 font-medium">
                <Loader2 className="h-3.5 w-3.5 animate-spin text-primary" />
                <span>
                  {progress?.phase === "download"
                    ? "HuggingFace İndiriliyor"
                    : ".vxt Sıkıştırılıyor"}
                </span>
                {progress && (
                  <Badge variant="outline" className="font-mono text-[10px]">
                    %{progress.percent.toFixed(1)}
                  </Badge>
                )}
              </div>
              <Button
                variant="destructive"
                size="sm"
                className="h-6 px-2 text-[11px]"
                onClick={() => void cancelForgeJob()}
              >
                <Square className="h-3 w-3" />
                İptal Et
              </Button>
            </div>
            <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
              <div
                className="h-full bg-primary transition-all duration-150"
                style={{ width: `${Math.min(100, Math.max(2, progress?.percent ?? 4))}%` }}
              />
            </div>
            <div className="truncate font-mono text-[11px] text-muted-foreground">
              {progress?.detail ?? "Hazırlanıyor..."}
            </div>
          </div>
        )}

        {vxtLibrary.length > 0 && (
          <>
            <Separator />
            <div className="space-y-1.5">
              <div className="flex items-center justify-between text-[11px] font-medium text-muted-foreground">
                <span className="flex items-center gap-1.5">
                  <HardDrive className="h-3.5 w-3.5" />
                  Yerel .vxt Model Kütüphanesi ({vxtLibrary.length})
                </span>
              </div>
              <div className="max-h-44 space-y-1.5 overflow-y-auto pr-1">
                {vxtLibrary.map((m) => (
                  <div
                    key={m.path}
                    className="flex items-center justify-between gap-2 rounded-md border border-border/70 bg-background/50 px-2.5 py-1.5 text-xs"
                  >
                    <div className="min-w-0 flex-1">
                      <div className="truncate font-mono font-medium">{m.name}</div>
                      <div className="text-[10px] text-muted-foreground">
                        {m.size_mib} MiB {m.runnable_hint ? "· Çalıştırılabilir LLM" : "· Tensör Arşivi"}
                      </div>
                    </div>
                    <div className="flex items-center gap-1">
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-7 w-7"
                        title="Tensörleri İncele"
                        onClick={() => void onInspectPath(m.path)}
                      >
                        <Search className="h-3.5 w-3.5" />
                      </Button>
                      {m.runnable_hint && (
                        <Button
                          variant="secondary"
                          size="sm"
                          className="h-7 px-2 text-[11px]"
                          onClick={() => void onLoadIntoChat(m.path)}
                        >
                          <Play className="h-3 w-3" />
                          Yükle
                        </Button>
                      )}
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-7 w-7 text-red-400 hover:text-red-300"
                        title=".vxt Dosyasını Sil"
                        disabled={deletingPath === m.path || busy}
                        onClick={() => void handleDelete(m.path)}
                      >
                        {deletingPath === m.path ? (
                          <Loader2 className="h-3.5 w-3.5 animate-spin" />
                        ) : (
                          <Trash2 className="h-3.5 w-3.5" />
                        )}
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}
