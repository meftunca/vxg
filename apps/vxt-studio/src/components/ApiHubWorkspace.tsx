import * as React from "react";
import { Check, Copy, Globe, Radio, Terminal } from "lucide-react";
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
  const modelName = studio?.model?.name || "Qwen3.5-2B";
  const [pingResult, setPingResult] = React.useState<string | null>(null);

  const testModelsEndpoint = async () => {
    try {
      const res = await fetch(`${baseUrl}/models`);
      const json = await res.json();
      setPingResult(JSON.stringify(json, null, 2));
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
  prompt: "Explain compressed-domain Metal GEMV in 2 sentences.",
  temperature: 0,
});`;

  const pythonCode = `from openai import OpenAI

client = OpenAI(base_url="${baseUrl}", api_key="vxt-local")
stream = client.chat.completions.create(
    model="${modelName}",
    messages=[{"role": "user", "content": "Merhaba VXT!"}],
    temperature=0,
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)`;

  const curlCode = `curl -N ${baseUrl}/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "${modelName}",
    "messages": [{"role": "user", "content": "Selam! Kısaca kendini tanıt."}],
    "temperature": 0,
    "stream": true
  }'`;

  return (
    <div className="h-[calc(100vh-88px)] space-y-4 overflow-y-auto pr-1">
      <Card className="bg-card/80">
        <CardHeader className="pb-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              <CardTitle className="flex items-center gap-2 text-base">
                <Globe className="h-4 w-4 text-primary" />
                Yerel OpenAI Uyumlu Sunucu (LM Studio / Cursor / AI SDK Drop-In)
              </CardTitle>
              <CardDescription>
                Uygulama açıkken gömülü Rust sunucusu SSE streaming destekli{" "}
                <code>/v1/chat/completions</code> ve <code>/v1/models</code> uçlarını sunar.
              </CardDescription>
            </div>
            <div className="flex items-center gap-2">
              <Badge className="gap-1.5 font-mono">
                <Radio className="h-3 w-3 animate-pulse" />
                {baseUrl}
              </Badge>
              <Button variant="outline" size="sm" onClick={() => void testModelsEndpoint()}>
                <Terminal className="h-3.5 w-3.5" />
                /v1/models Test Et
              </Button>
            </div>
          </div>
        </CardHeader>
        {pingResult && (
          <CardContent>
            <pre className="rounded-lg border bg-background/70 p-3 font-mono text-xs text-emerald-400">
              {pingResult}
            </pre>
          </CardContent>
        )}
      </Card>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <CodeSnippet title="Vercel AI SDK (@ai-sdk/openai-compatible)" code={aiSdkCode} />
        <CodeSnippet title="Python OpenAI SDK" code={pythonCode} />
      </div>
      <CodeSnippet title="cURL (SSE Token Streaming)" code={curlCode} />
    </div>
  );
}
