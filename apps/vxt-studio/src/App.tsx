import * as React from "react";
import { open } from "@tauri-apps/plugin-dialog";
import {
  Cpu,
  FolderOpen,
  Layers,
  Loader2,
  MessageSquare,
  Radio,
  Server,
} from "lucide-react";
import { ApiHubWorkspace } from "@/components/ApiHubWorkspace";
import { ChatWorkspace } from "@/components/ChatWorkspace";
import { ForgeWorkspace } from "@/components/ForgeWorkspace";
import {
  Badge,
  Button,
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/primitives";
import {
  fetchStudioState,
  isTauriRuntime,
  loadModel,
  type StudioState,
} from "@/lib/vxt-bridge";

export default function App() {
  const [tab, setTab] = React.useState("chat");
  const [studio, setStudio] = React.useState<StudioState | null>(null);
  const [loadingModel, setLoadingModel] = React.useState(false);
  const [loadErr, setLoadErr] = React.useState<string | null>(null);

  const refresh = React.useCallback(async () => {
    try {
      const st = await fetchStudioState();
      setStudio(st);
    } catch {
      // Server still starting
    }
  }, []);

  React.useEffect(() => {
    void refresh();
    const timer = setInterval(() => {
      void refresh();
    }, 1800);
    return () => clearInterval(timer);
  }, [refresh]);

  const handleLoadModelPath = async (path: string) => {
    setLoadErr(null);
    setLoadingModel(true);
    try {
      const st = await loadModel(path, true);
      setStudio(st);
      setTab("chat");
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

  const runnableModels = (studio?.discovered_files ?? []).filter((f) => f.runnable_hint);

  return (
    <div className="min-h-screen bg-background text-foreground px-4 py-3">
      <Tabs value={tab} onValueChange={setTab} className="w-full">
        <header className="mb-3 flex flex-wrap items-center justify-between gap-3 rounded-xl border bg-card/90 px-4 py-2.5">
          <div className="flex items-center gap-3">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary/20 font-bold text-primary text-sm">
              VX
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="text-sm font-semibold tracking-tight">VXT Studio</span>
                <Badge variant="outline" className="text-[10px]">
                  CB4-v2 · Vision · Metal 0-ULP
                </Badge>
              </div>
              <div className="text-[11px] text-muted-foreground">
                Compressed-Domain AI Chat &amp; Local .vxt Tensor Forge
              </div>
            </div>

            <TabsList className="ml-3">
              <TabsTrigger value="chat">
                <MessageSquare className="h-3.5 w-3.5" />
                Chat
              </TabsTrigger>
              <TabsTrigger value="forge">
                <Layers className="h-3.5 w-3.5" />
                Tensor → .vxt Forge
              </TabsTrigger>
              <TabsTrigger value="api">
                <Server className="h-3.5 w-3.5" />
                OpenAI API
              </TabsTrigger>
            </TabsList>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            {runnableModels.length > 0 && (
              <select
                value={studio?.model?.path || ""}
                disabled={loadingModel}
                onChange={(e) => {
                  if (e.target.value) void handleLoadModelPath(e.target.value);
                }}
                className="h-8 max-w-[260px] rounded-md border border-input bg-background px-2.5 text-xs font-mono"
              >
                <option value="" disabled>
                  Model Seç (.vxt / .gguf)...
                </option>
                {runnableModels.map((m) => (
                  <option key={m.path} value={m.path}>
                    {m.name} ({m.size_mib} MiB)
                  </option>
                ))}
              </select>
            )}

            <Button
              variant="outline"
              size="sm"
              disabled={loadingModel}
              onClick={() => void handlePickModelDialog()}
            >
              {loadingModel || studio?.busy && !studio?.model ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin text-primary" />
              ) : (
                <FolderOpen className="h-3.5 w-3.5" />
              )}
              Model Aç
            </Button>

            <Badge variant={studio?.model ? "default" : "secondary"} className="gap-1.5 py-1">
              {studio?.model ? (
                <>
                  <Cpu className="h-3 w-3" />
                  <span>Apple Metal Hazır</span>
                </>
              ) : (
                <>
                  <Radio className="h-3 w-3 animate-pulse" />
                  <span>{studio?.busy ? "GPU'ya Yükleniyor..." : "Model Bekleniyor"}</span>
                </>
              )}
            </Badge>
          </div>
        </header>

        {loadErr && (
          <div className="mb-3 rounded-lg border border-red-500/40 bg-red-500/10 px-3 py-2 text-xs text-red-300">
            {loadErr}
          </div>
        )}

        <TabsContent value="chat" className="mt-0">
          <ChatWorkspace
            studio={studio}
            onRefresh={refresh}
            onSwitchToForge={() => setTab("forge")}
          />
        </TabsContent>

        <TabsContent value="forge" className="mt-0">
          <ForgeWorkspace
            discovered={studio?.discovered_files ?? []}
            onLoadIntoChat={handleLoadModelPath}
            onRefresh={refresh}
          />
        </TabsContent>

        <TabsContent value="api" className="mt-0">
          <ApiHubWorkspace studio={studio} />
        </TabsContent>
      </Tabs>
    </div>
  );
}
