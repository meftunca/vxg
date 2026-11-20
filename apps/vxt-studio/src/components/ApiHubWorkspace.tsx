import * as React from "react";
import { Activity, Globe, PanelLeft, Radio, Server, Terminal } from "lucide-react";
import { HighlightedSnippet } from "@/components/MarkdownRenderer";
import type { StudioState } from "@/lib/vxt-bridge";

export function ApiHubWorkspace({
  studio,
  sidebarCollapsed,
  onToggleSidebar,
}: {
  studio: StudioState | null;
  sidebarCollapsed: boolean;
  onToggleSidebar: () => void;
}) {
  const baseUrl = studio?.base_url || "http://127.0.0.1:11435/v1";
  const rootUrl = baseUrl.replace(/\/v1$/, "");
  const modelName = studio?.model?.name || "qwen35-9b";
  const [pingLabel, setPingLabel] = React.useState<string>("");
  const [pingResult, setPingResult] = React.useState<string | null>(null);

  const probeEndpoint = async (label: string, url: string, asText = false) => {
    setPingLabel(label);
    try {
      const res = await fetch(url);
      if (asText) {
        setPingResult(await res.text());
      } else {
        const json = await res.json();
        setPingResult(JSON.stringify(json, null, 2));
      }
    } catch (e) {
      setPingResult(String(e));
    }
  };

  const aiSdkCode = `import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import { streamText } from "ai";

const vxt = createOpenAICompatible({
  name: "vxt-local",
  baseURL: "${baseUrl}",
});

const { textStream } = streamText({
  model: vxt("${modelName}"),
  prompt: "Explain compressed-domain VXT-Turbo E8 inference in 2 sentences.",
  temperature: 0,
});`;

  const pythonCode = `from openai import OpenAI

client = OpenAI(base_url="${baseUrl}", api_key="vxt-local")
resp = client.chat.completions.create(
    model="${modelName}",
    messages=[{"role": "user", "content": "Return JSON with city and country for Tokyo."}],
    response_format={"type": "json_object"},
    temperature=0,
)
print(resp.choices[0].message.content)`;

  const toolsCode = `curl -s ${baseUrl}/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "${modelName}",
    "messages": [{"role": "user", "content": "What is the weather in Istanbul?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
      }
    }],
    "stop": ["\\n\\n"],
    "temperature": 0
  }'`;

  const legacyCode = `# Legacy text completions (/v1/completions) + Prometheus telemetry (/metrics)
curl -s ${baseUrl}/completions \\
  -H "Content-Type: application/json" \\
  -d '{"model": "${modelName}", "prompt": "fn fibonacci(n: u64) -> u64 {", "max_tokens": 64, "temperature": 0}'

curl -s ${rootUrl}/metrics`;

  return (
    <div className="flex h-full flex-col bg-[#19191c]">
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
          <Globe className="h-4 w-4 text-emerald-400" />
          <h1 className="text-sm font-semibold text-zinc-100">OpenAI API &amp; Telemetri</h1>
        </div>
        <div className="flex items-center gap-2 rounded-lg border border-emerald-500/30 bg-emerald-500/10 px-3 py-1 font-mono text-xs text-emerald-300">
          <Radio className="h-3 w-3 animate-pulse" />
          <span>{baseUrl}</span>
        </div>
      </header>

      <div className="flex-1 overflow-y-auto px-6 py-6">
        <div className="mx-auto max-w-4xl space-y-5">
          {/* Live Endpoint Probe Card */}
          <div className="rounded-2xl border border-white/[0.07] bg-[#1e1e23] p-5 space-y-4">
            <div className="flex flex-wrap items-start justify-between gap-4">
              <div className="space-y-1">
                <h2 className="text-sm font-semibold text-zinc-100">
                  Yerel Production Sunucu (Cursor · OpenAI SDK · Vercel AI SDK · Prometheus)
                </h2>
                <p className="text-xs text-zinc-400 leading-relaxed max-w-2xl">
                  Gömülü Rust HTTP/SSE motoru <code className="text-zinc-300">/v1/chat/completions</code>{" "}
                  (<code className="text-zinc-300">tools</code>,{" "}
                  <code className="text-zinc-300">response_format</code> JSON/Schema,{" "}
                  <code className="text-zinc-300">stop</code>, Vision{" "}
                  <code className="text-zinc-300">image_url</code>),{" "}
                  <code className="text-zinc-300">/v1/completions</code>,{" "}
                  <code className="text-zinc-300">/v1/models</code>,{" "}
                  <code className="text-zinc-300">/health</code> ve Prometheus{" "}
                  <code className="text-zinc-300">/metrics</code> uçlarını sunar.
                </p>
              </div>
              <div className="flex flex-wrap items-center gap-2">
                <button
                  type="button"
                  onClick={() => void probeEndpoint("GET /v1/models", `${baseUrl}/models`)}
                  className="inline-flex items-center gap-1.5 rounded-xl border border-white/[0.08] bg-white/[0.04] px-3 py-1.5 text-xs text-zinc-200 transition hover:bg-white/[0.08] cursor-pointer"
                >
                  <Terminal className="h-3.5 w-3.5 text-zinc-400" />
                  /v1/models
                </button>
                <button
                  type="button"
                  onClick={() => void probeEndpoint("GET /health", `${rootUrl}/health`)}
                  className="inline-flex items-center gap-1.5 rounded-xl border border-white/[0.08] bg-white/[0.04] px-3 py-1.5 text-xs text-zinc-200 transition hover:bg-white/[0.08] cursor-pointer"
                >
                  <Server className="h-3.5 w-3.5 text-zinc-400" />
                  /health
                </button>
                <button
                  type="button"
                  onClick={() =>
                    void probeEndpoint("GET /metrics (Prometheus)", `${rootUrl}/metrics`, true)
                  }
                  className="inline-flex items-center gap-1.5 rounded-xl border border-emerald-500/30 bg-emerald-500/10 px-3 py-1.5 text-xs text-emerald-300 transition hover:bg-emerald-500/15 cursor-pointer"
                >
                  <Activity className="h-3.5 w-3.5" />
                  /metrics
                </button>
              </div>
            </div>

            {pingResult && (
              <div className="overflow-hidden rounded-xl border border-white/[0.07] bg-[#121215]">
                <div className="flex items-center justify-between border-b border-white/[0.06] px-3.5 py-2 text-[11px] font-mono text-zinc-400">
                  <span>{pingLabel}</span>
                  <button
                    type="button"
                    onClick={() => setPingResult(null)}
                    className="text-zinc-500 hover:text-zinc-200 cursor-pointer"
                  >
                    Kapat
                  </button>
                </div>
                <pre className="max-h-64 overflow-auto p-3.5 font-mono text-xs text-emerald-300/90 leading-relaxed">
                  {pingResult}
                </pre>
              </div>
            )}
          </div>

          {/* Syntax-Highlighted Integration Snippets */}
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
            <HighlightedSnippet title="Vercel AI SDK (TypeScript)" lang="typescript" code={aiSdkCode} />
            <HighlightedSnippet
              title="OpenAI Python SDK (JSON Mode)"
              lang="python"
              code={pythonCode}
            />
            <HighlightedSnippet
              title="cURL · Function Calling / Tools & Stop"
              lang="bash"
              code={toolsCode}
            />
            <HighlightedSnippet
              title="cURL · Legacy /v1/completions & /metrics"
              lang="bash"
              code={legacyCode}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
