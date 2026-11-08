import * as React from "react";
import { open, save } from "@tauri-apps/plugin-dialog";
import {
  ArrowRight,
  CheckCircle2,
  FileBox,
  FolderOpen,
  Layers,
  Loader2,
  Play,
  Sparkles,
  Wand2,
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
  Switch,
} from "@/components/ui/primitives";
import {
  convertToVxt,
  createDemoTensor,
  inspectTensorFile,
  isTauriRuntime,
  type ConvertReport,
  type DiscoveredFile,
  type InspectReport,
} from "@/lib/vxt-bridge";

const PACK_MODES = [
  {
    id: "cb4",
    title: "CB4-v2 + Sparse Outlier",
    badge: "Önerilen · 3.64×",
    desc: "~4.38 bpw · Orijinal BF16 kalitesi (36/40) + 176 tok/s CB4-Fast Metal",
  },
  {
    id: "lossless",
    title: "Lossless BF16 / FP16",
    badge: "0-ULP Bit-Exact",
    desc: "~11.4 bpw · %100 kayıpsız bit düzlemi + üs tablosu sıkıştırması",
  },
  {
    id: "turbo",
    title: "VXT-Turbo (WHT + EC)",
    badge: "Ultra Küçük · 5.97×",
    desc: "~2.68 bpw · Hadamard + entropi kodlamalı maksimum sıkıştırma",
  },
  {
    id: "cb3",
    title: "CB3 + Sparse Outlier",
    badge: "4.6× Küçük",
    desc: "~3.45 bpw · 3-bit kübik Lloyd-Max + seyrek FP16 outlier akışı",
  },
];

interface ForgeWorkspaceProps {
  discovered: DiscoveredFile[];
  onLoadIntoChat: (vxtPath: string) => Promise<void>;
  onRefresh: () => Promise<void>;
}

export function ForgeWorkspace({
  discovered,
  onLoadIntoChat,
  onRefresh,
}: ForgeWorkspaceProps) {
  const [inputPath, setInputPath] = React.useState("");
  const [outputPath, setOutputPath] = React.useState("");
  const [mode, setMode] = React.useState("cb4");
  const [packAsLlm, setPackAsLlm] = React.useState(true);
  const [inspecting, setInspecting] = React.useState(false);
  const [converting, setConverting] = React.useState(false);
  const [report, setReport] = React.useState<InspectReport | null>(null);
  const [convResult, setConvResult] = React.useState<ConvertReport | null>(null);
  const [err, setErr] = React.useState<string | null>(null);

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
      defaultPath: outputPath || "model-cb4.vxt",
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

  const handleConvert = async () => {
    if (!inputPath || !outputPath) return;
    setErr(null);
    setConverting(true);
    try {
      const res = await convertToVxt(inputPath, outputPath, mode, packAsLlm);
      setConvResult(res);
      const outRep = await inspectTensorFile(res.output_path);
      setReport(outRep);
      await onRefresh();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setConverting(false);
    }
  };

  return (
    <div className="grid h-[calc(100vh-88px)] grid-cols-1 gap-4 lg:grid-cols-[430px_1fr] overflow-y-auto">
      <div className="space-y-4">
        <Card className="bg-card/80">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="flex items-center gap-2 text-base">
                <Wand2 className="h-4 w-4 text-primary" />
                Tensor → .vxt Dönüştürücü
              </CardTitle>
              <Button variant="outline" size="sm" onClick={() => void handleGenerateDemoTensor()}>
                <Sparkles className="h-3.5 w-3.5 text-primary" />
                Örnek .safetensors Üret
              </Button>
            </div>
            <CardDescription>
              Cihazdaki <code>HuggingFace</code> model klasörlerini (<code>config.json</code> +{" "}
              <code>.safetensors</code>) veya <code>.gguf</code> ağırlıklarını sıkıştırılmış{" "}
              <code>.vxt</code> arşivine dönüştürün.
            </CardDescription>
          </CardHeader>

          <CardContent className="space-y-4">
            <div className="space-y-1.5">
              <div className="text-xs font-medium">
                Kaynak Model / Dosya (HF Klasörü / .safetensors / .gguf / .vxt)
              </div>
              <div className="flex gap-2">
                <Input
                  value={inputPath}
                  onChange={(e) => setInputPath(e.target.value)}
                  placeholder="/path/to/HF-Model-Dir veya model.gguf"
                  className="font-mono text-xs"
                />
                <Button
                  variant="secondary"
                  size="sm"
                  title="Dosya Seç (.safetensors / .gguf / .vxt)"
                  onClick={() => void handleBrowseInput()}
                >
                  <FileBox className="h-3.5 w-3.5" />
                  Dosya
                </Button>
                <Button
                  variant="secondary"
                  size="sm"
                  title="HuggingFace Model Klasörü Seç (config.json + tokenizer.json + .safetensors)"
                  onClick={() => void handleBrowseHfDir()}
                >
                  <FolderOpen className="h-3.5 w-3.5" />
                  HF Klasör
                </Button>
              </div>
            </div>

            {discovered.length > 0 && (
              <div className="space-y-1.5">
                <div className="text-[11px] font-medium text-muted-foreground">
                  Cihazda Bulunan Hazır Dosyalar (Tıkla &amp; İncele)
                </div>
                <div className="flex max-h-28 flex-wrap gap-1.5 overflow-y-auto">
                  {discovered.map((f) => (
                    <button
                      key={f.path}
                      type="button"
                      onClick={() => void handleSelectInput(f.path)}
                      className="inline-flex items-center gap-1.5 rounded-md border border-border/80 bg-background/60 px-2 py-1 text-[11px] hover:border-primary/50 cursor-pointer"
                    >
                      <Badge variant="secondary" className="px-1 py-0 text-[10px]">
                        {f.kind}
                      </Badge>
                      <span className="max-w-[190px] truncate font-mono">{f.name}</span>
                      <span className="text-muted-foreground">{f.size_mib}M</span>
                    </button>
                  ))}
                </div>
              </div>
            )}

            <Separator />

            <div className="space-y-2">
              <div className="text-xs font-medium">Sıkıştırma Modu</div>
              <div className="grid grid-cols-1 gap-2">
                {PACK_MODES.map((m) => (
                  <button
                    key={m.id}
                    type="button"
                    onClick={() => {
                      setMode(m.id);
                      if (inputPath && !inputPath.toLowerCase().endsWith(".vxt")) {
                        setOutputPath(suggestOutputPath(inputPath, m.id));
                      }
                    }}
                    className={`rounded-lg border p-2.5 text-left transition cursor-pointer ${
                      mode === m.id
                        ? "border-primary bg-primary/10"
                        : "border-border/70 bg-background/40 hover:border-border"
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-semibold">{m.title}</span>
                      <Badge variant={mode === m.id ? "default" : "outline"}>{m.badge}</Badge>
                    </div>
                    <div className="mt-1 text-[11px] text-muted-foreground">{m.desc}</div>
                  </button>
                ))}
              </div>
            </div>

            {report && (report.format.startsWith("GGUF") || report.format.startsWith("HuggingFace")) && (
              <div className="flex items-center justify-between rounded-lg border bg-muted/30 px-3 py-2">
                <div>
                  <div className="text-xs font-medium">Tam Çalıştırılabilir LLM Paketi</div>
                  <div className="text-[11px] text-muted-foreground">
                    Tokenizer + ChatML + CB4-v2 analitik kalibrasyon dahil edilir
                  </div>
                </div>
                <Switch checked={packAsLlm} onCheckedChange={setPackAsLlm} />
              </div>
            )}

            <div className="space-y-1.5">
              <div className="text-xs font-medium">Hedef .vxt Çıktı Yolu</div>
              <div className="flex gap-2">
                <Input
                  value={outputPath}
                  onChange={(e) => setOutputPath(e.target.value)}
                  placeholder="/path/to/output.vxt"
                  className="font-mono text-xs"
                />
                <Button variant="secondary" size="icon" onClick={() => void handleBrowseOutput()}>
                  <FolderOpen className="h-4 w-4" />
                </Button>
              </div>
            </div>

            <Button
              className="w-full"
              disabled={!inputPath || !outputPath || converting || inputPath.endsWith(".vxt")}
              onClick={() => void handleConvert()}
            >
              {converting ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Sıkıştırılıyor (.vxt)...
                </>
              ) : (
                <>
                  <FileBox className="h-4 w-4" />
                  .vxt Olarak Sıkıştır ve Kaydet
                </>
              )}
            </Button>

            {err && (
              <div className="rounded-lg border border-red-500/40 bg-red-500/10 p-2.5 text-xs text-red-300">
                {err}
              </div>
            )}
          </CardContent>
        </Card>

        {convResult && (
          <Card className="border-emerald-500/40 bg-emerald-500/5">
            <CardContent className="p-4 space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 text-sm font-semibold text-emerald-400">
                  <CheckCircle2 className="h-4 w-4" />
                  .vxt Arşivi Hazır!
                </div>
                <Badge>{convResult.ratio}× Küçüldü</Badge>
              </div>
              <div className="flex items-center gap-2 font-mono text-xs">
                <span>{convResult.input_mib} MiB</span>
                <ArrowRight className="h-3.5 w-3.5 text-primary" />
                <span className="font-bold text-primary">{convResult.output_mib} MiB</span>
                <span className="text-muted-foreground">
                  ({convResult.tensor_count} tensör · {convResult.elapsed_s}s)
                </span>
              </div>
              <div className="truncate font-mono text-[11px] text-muted-foreground">
                {convResult.output_path}
              </div>
              {convResult.runnable_llm && (
                <Button
                  size="sm"
                  className="w-full"
                  onClick={() => void onLoadIntoChat(convResult.output_path)}
                >
                  <Play className="h-3.5 w-3.5" />
                  Bu .vxt Modelini Chat Motoruna Yükle
                </Button>
              )}
            </CardContent>
          </Card>
        )}
      </div>

      <Card className="flex flex-col overflow-hidden bg-card/80">
        <CardHeader className="pb-3 border-b bg-muted/20">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              <CardTitle className="flex items-center gap-2 text-base">
                <Layers className="h-4 w-4 text-primary" />
                Tensör &amp; Arşiv Müfettişi
              </CardTitle>
              <CardDescription>
                {report
                  ? `${report.file_name} — ${report.format} · ${report.size_mib} MiB · ${report.tensor_count} tensör`
                  : "İncelemek için soldan bir .safetensors, .gguf veya .vxt seçin"}
              </CardDescription>
            </div>
            {report?.runnable_llm &&
              (report.format === "VXT1" || report.format.startsWith("GGUF")) && (
                <Button size="sm" onClick={() => void onLoadIntoChat(report.path)}>
                  <Play className="h-3.5 w-3.5" />
                  Chat&apos;te Aç
                </Button>
              )}
          </div>
        </CardHeader>

        <div className="flex-1 overflow-auto">
          {inspecting ? (
            <div className="flex h-64 items-center justify-center gap-2 text-sm text-muted-foreground">
              <Loader2 className="h-4 w-4 animate-spin text-primary" />
              Tensör tablosu okunuyor...
            </div>
          ) : report ? (
            <table className="w-full text-left text-xs">
              <thead className="sticky top-0 bg-card border-b text-muted-foreground">
                <tr>
                  <th className="px-3 py-2 font-medium">#</th>
                  <th className="px-3 py-2 font-medium">Tensör Adı</th>
                  <th className="px-3 py-2 font-medium">Boyut</th>
                  <th className="px-3 py-2 font-medium">Mod / DType</th>
                  <th className="px-3 py-2 text-right font-medium">BPW</th>
                  <th className="px-3 py-2 text-right font-medium">KB</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border/50 font-mono">
                {report.tensors.map((t) => (
                  <tr key={t.index} className="hover:bg-muted/30">
                    <td className="px-3 py-1.5 text-muted-foreground">{t.index}</td>
                    <td className="px-3 py-1.5 text-foreground">{t.name}</td>
                    <td className="px-3 py-1.5 text-muted-foreground">{t.shape}</td>
                    <td className="px-3 py-1.5">
                      <Badge variant="outline" className="font-mono text-[10px]">
                        {t.codec_or_dtype}
                      </Badge>
                    </td>
                    <td className="px-3 py-1.5 text-right text-primary">{t.bpw.toFixed(2)}</td>
                    <td className="px-3 py-1.5 text-right text-muted-foreground">
                      {t.size_kib.toFixed(1)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <div className="flex h-64 flex-col items-center justify-center text-center p-6 text-xs text-muted-foreground">
              <FileBox className="mb-2 h-8 w-8 opacity-40" />
              <div>
                Hızlı sahne demosu için soldaki <b>&ldquo;Örnek .safetensors Üret&rdquo;</b> butonuna
                basıp saniyesinde <code>CB4-v2</code> veya <code>Lossless</code> <code>.vxt</code>{" "}
                çıktısı alabilirsiniz.
              </div>
            </div>
          )}
        </div>
      </Card>
    </div>
  );
}
