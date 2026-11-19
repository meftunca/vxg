import * as React from "react";
import { Activity, Check, Copy, Globe, Radio, Server, Terminal } from "lucide-react";
import {
  Badge,
  Button,
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/primitives";
import type { StudioState } from "@/lib/vxt-bridge";

function CodeSnippet({ title, code }: { title: string; code: string }) {
  const [copied, setCopied] = React.useState(false);
  return (
    <Card className="bg-card/80">
      <CardHeader className="flex flex-row items-center justify-between py-3 px-4 border-b">
        <CardTitle className="text-xs font-semibold">{title}</CardTitle>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => {
            void navigator.clipboard.writeText(code);
            setCopied(true);
            setTimeout(() => setCopied(false), 1500);
          }}
        >
          {copied ? <Check className="h-3.5 w-3.5 text-primary" /> : <Copy className="h-3.5 w-3.5" />}
          {copied ? "Kopyalandı" : "Kopyala"}
        </Button>
      </CardHeader>
      <CardContent className="p-4">
        <pre className="overflow-x-auto font-mono text-xs text-muted-foreground leading-relaxed">
          {code}
        </pre>
      </CardContent>
    </Card>
  );
}

export function ApiHubWorkspace({ studio }: { studio: StudioState | null }) {
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
    <div className="h-[calc(100vh-88px)] space-y-4 overflow-y-auto pr-1">
      <Card className="bg-card/80">
        <CardHeader className="pb-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              <CardTitle className="flex items-center gap-2 text-base">
                <Globe className="h-4 w-4 text-primary" />
                Production OpenAI Uyumlu Sunucu (Cursor / AI SDK / OpenAI / Prometheus)
              </CardTitle>
              <CardDescription>
                Gömülü Rust sunucusu SSE <code>/v1/chat/completions</code> (<code>tools</code>,{" "}
                <code>response_format</code> JSON/Schema, <code>stop</code>, Vision{" "}
                <code>image_url</code>), <code>/v1/completions</code>, <code>/v1/models</code>,{" "}
                <code>/health</code> ve Prometheus <code>/metrics</code> uçlarını sunar.
              </CardDescription>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <Badge className="gap-1.5 font-mono">
                <Radio className="h-3 w-3 animate-pulse" />
                {baseUrl}
              </Badge>
              <Button
                variant="outline"
                size="sm"
                onClick={() => void probeEndpoint("GET /v1/models", `${baseUrl}/models`)}
              >
                <Terminal className="h-3.5 w-3.5" />
                /v1/models
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => void probeEndpoint("GET /health", `${rootUrl}/health`)}
              >
                <Server className="h-3.5 w-3.5" />
                /health
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => void probeEndpoint("GET /metrics (Prometheus)", `${rootUrl}/metrics`, true)}
              >
                <Activity className="h-3.5 w-3.5 text-emerald-400" />
                /metrics
              </Button>
            </div>
          </div>
        </CardHeader>
        {pingResult && (
          <CardContent className="space-y-1.5">
            {pingLabel && (
              <div className="font-mono text-[11px] text-muted-foreground">{pingLabel}</div>
            )}
            <pre className="max-h-64 overflow-auto rounded-lg border bg-background/70 p-3 font-mono text-xs text-emerald-400">
              {pingResult}
            </pre>
          </CardContent>
        )}
      </Card>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <CodeSnippet title="Vercel AI SDK (@ai-sdk/openai-compatible)" code={aiSdkCode} />
        <CodeSnippet title="Python OpenAI SDK (JSON Mode)" code={pythonCode} />
      </div>
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <CodeSnippet title="cURL · OpenAI Tool Calling & Stop Sequences" code={toolsCode} />
        <CodeSnippet title="cURL · /v1/completions & Prometheus /metrics" code={legacyCode} />
      </div>
    </div>
  );
}
