import * as React from "react";
import ReactMarkdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import { Check, Copy, Terminal } from "lucide-react";

function extractText(node: React.ReactNode): string {
  if (typeof node === "string" || typeof node === "number") {
    return String(node);
  }
  if (Array.isArray(node)) {
    return node.map(extractText).join("");
  }
  if (React.isValidElement<{ children?: React.ReactNode }>(node)) {
    return extractText(node.props.children);
  }
  return "";
}

function CodeBlock({
  className,
  children,
}: {
  className?: string;
  children?: React.ReactNode;
}) {
  const [copied, setCopied] = React.useState(false);
  const langMatch = /language-([a-zA-Z0-9_+-]+)/.exec(className || "");
  const lang = langMatch?.[1] || "code";
  const rawCode = React.useMemo(() => extractText(children).replace(/\n$/, ""), [children]);

  const handleCopy = () => {
    void navigator.clipboard.writeText(rawCode);
    setCopied(true);
    setTimeout(() => setCopied(false), 1600);
  };

  return (
    <div className="my-3.5 overflow-hidden rounded-xl border border-white/[0.08] bg-[#111114]">
      <div className="flex items-center justify-between border-b border-white/[0.06] bg-white/[0.025] px-3.5 py-2">
        <div className="flex items-center gap-1.5 text-[11px] font-mono text-zinc-400">
          <Terminal className="h-3 w-3 text-zinc-500" />
          <span>{lang}</span>
        </div>
        <button
          type="button"
          onClick={handleCopy}
          className="flex items-center gap-1 rounded-md px-2 py-0.5 text-[11px] text-zinc-400 transition hover:bg-white/[0.07] hover:text-zinc-100 cursor-pointer"
        >
          {copied ? (
            <>
              <Check className="h-3 w-3 text-emerald-400" />
              <span className="text-emerald-400">Kopyalandı</span>
            </>
          ) : (
            <>
              <Copy className="h-3 w-3" />
              <span>Kopyala</span>
            </>
          )}
        </button>
      </div>
      <div className="overflow-x-auto px-4 py-3.5">
        <code className={`${className || ""} block font-mono text-[13px] leading-[1.6]`}>
          {children}
        </code>
      </div>
    </div>
  );
}

const REMARK_PLUGINS = [remarkGfm, remarkMath];
const REHYPE_PLUGINS = [
  rehypeKatex,
  [rehypeHighlight, { detect: true, ignoreMissing: true }] as never,
];

export const MarkdownRenderer = React.memo(function MarkdownRenderer({
  content,
}: {
  content: string;
}) {
  return (
    <div className="vxt-prose">
      <ReactMarkdown
        remarkPlugins={REMARK_PLUGINS}
        rehypePlugins={REHYPE_PLUGINS}
        components={{
          pre({ children }) {
            if (
              React.isValidElement<{ className?: string; children?: React.ReactNode }>(children)
            ) {
              return (
                <CodeBlock className={children.props.className}>
                  {children.props.children}
                </CodeBlock>
              );
            }
            return <pre className="my-3 overflow-x-auto rounded-xl bg-[#111114] p-4">{children}</pre>;
          },
          table({ children }) {
            return (
              <div className="my-3.5 overflow-x-auto rounded-xl border border-white/[0.08] bg-[#141418]">
                <table className="my-0">{children}</table>
              </div>
            );
          },
          a({ href, children }) {
            return (
              <a href={href} target="_blank" rel="noreferrer">
                {children}
              </a>
            );
          },
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
});

export function HighlightedSnippet({
  title,
  lang,
  code,
}: {
  title: string;
  lang: string;
  code: string;
}) {
  const [copied, setCopied] = React.useState(false);
  const fenced = React.useMemo(() => `\`\`\`${lang}\n${code}\n\`\`\``, [lang, code]);

  return (
    <div className="overflow-hidden rounded-xl border border-white/[0.08] bg-[#131316]">
      <div className="flex items-center justify-between border-b border-white/[0.06] bg-white/[0.02] px-4 py-2.5">
        <span className="text-xs font-medium text-zinc-200">{title}</span>
        <button
          type="button"
          onClick={() => {
            void navigator.clipboard.writeText(code);
            setCopied(true);
            setTimeout(() => setCopied(false), 1600);
          }}
          className="flex items-center gap-1.5 rounded-md px-2.5 py-1 text-[11px] text-zinc-400 transition hover:bg-white/[0.07] hover:text-zinc-100 cursor-pointer"
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
      </div>
      <div className="px-4 py-3 [&>div>div]:my-0 [&>div>div]:border-0 [&>div>div]:bg-transparent [&>div>div>div:first-child]:hidden [&>div>div>div:last-child]:p-0">
        <MarkdownRenderer content={fenced} />
      </div>
    </div>
  );
}
