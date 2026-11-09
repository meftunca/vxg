import * as React from "react";
import { ImagePlus, X } from "lucide-react";
import { Button } from "@/components/ui/primitives";

export interface VisionAttachment {
  name: string;
  contentType: string;
  url: string;
}

export async function readImageFiles(files: FileList | File[]): Promise<VisionAttachment[]> {
  const list = Array.from(files).filter((f) => f.type.startsWith("image/"));
  const results: VisionAttachment[] = [];
  for (const file of list) {
    const url = await new Promise<string>((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result || ""));
      reader.onerror = () => reject(new Error(`Görsel okunamadı: ${file.name}`));
      reader.readAsDataURL(file);
    });
    if (url.startsWith("data:image/")) {
      results.push({
        name: file.name || "clipboard.png",
        contentType: file.type || "image/png",
        url,
      });
    }
  }
  return results;
}

export function VisionPreviewStrip({
  attachments,
  onRemove,
}: {
  attachments: VisionAttachment[];
  onRemove: (idx: number) => void;
}) {
  if (attachments.length === 0) return null;
  return (
    <div className="mb-2.5 flex flex-wrap gap-2">
      {attachments.map((att, idx) => (
        <div
          key={`${att.name}-${idx}`}
          className="group relative h-16 w-16 overflow-hidden rounded-lg border border-border bg-muted/40"
          title={att.name}
        >
          <img src={att.url} alt={att.name} className="h-full w-full object-cover" />
          <button
            type="button"
            onClick={() => onRemove(idx)}
            className="absolute right-1 top-1 flex h-5 w-5 items-center justify-center rounded-full bg-black/75 text-white opacity-90 transition hover:bg-red-500 cursor-pointer"
          >
            <X className="h-3 w-3" />
          </button>
        </div>
      ))}
    </div>
  );
}

export function MessageImages({
  attachments,
}: {
  attachments?: Array<{ name?: string; contentType?: string; url: string }>;
}) {
  const imgs = (attachments ?? []).filter((a) => a.url?.startsWith("data:image/") || a.contentType?.startsWith("image/"));
  if (imgs.length === 0) return null;
  return (
    <div className="mb-2 flex flex-wrap gap-2">
      {imgs.map((img, i) => (
        <img
          key={`${img.name || "img"}-${i}`}
          src={img.url}
          alt={img.name || "attached"}
          className="max-h-52 max-w-[260px] rounded-xl border border-white/20 object-contain bg-black/20"
        />
      ))}
    </div>
  );
}

export function AttachImageButton({
  disabled,
  visionSupported,
  onFiles,
}: {
  disabled: boolean;
  visionSupported: boolean;
  onFiles: (files: FileList) => void;
}) {
  const inputRef = React.useRef<HTMLInputElement>(null);
  return (
    <>
      <input
        ref={inputRef}
        type="file"
        accept="image/png,image/jpeg,image/webp,image/bmp,image/gif"
        multiple
        className="hidden"
        onChange={(e) => {
          if (e.target.files && e.target.files.length > 0) {
            onFiles(e.target.files);
            e.target.value = "";
          }
        }}
      />
      <Button
        type="button"
        variant="outline"
        size="icon"
        disabled={disabled || !visionSupported}
        title={
          visionSupported
            ? "Görsel ekle (PNG/JPG/WebP) veya panodan Cmd+V ile yapıştır"
            : "Yüklü model Vision desteklemiyor"
        }
        onClick={() => inputRef.current?.click()}
        className="h-[46px] w-[46px] shrink-0"
      >
        <ImagePlus className="h-4 w-4" />
      </Button>
    </>
  );
}
