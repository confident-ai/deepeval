import {
  checkIfMultimodal,
  convertToMultiModalArray,
  MLLMImage,
} from "@/test-case/mllm-image";

// Per-provider multimodal message builders. A prompt is a plain string that may
// contain image slugs (`[DEEPEVAL:IMAGE|PDF:{id}]`); when it does, we split it
// into text + image parts in each provider's expected shape. Mirrors the
// `check_if_multimodal` / `convert_to_multi_modal_array` flow in Python's models.

type Part = Record<string, unknown>;

/** Image as a URL (remote) or data URI (local/base64) — for OpenAI & AI SDK. */
function imageSource(img: MLLMImage): string {
  if (img.url && !img.local) return img.url;
  img.ensureImagesLoaded();
  return img.asDataUri() ?? img.url ?? "";
}

/** Raw bytes for providers that require base64 (Gemini); fetches remote URLs. */
async function imageBytes(
  img: MLLMImage,
): Promise<{ mimeType: string; data: string }> {
  img.ensureImagesLoaded();
  if (img.dataBase64 && img.mimeType) {
    return { mimeType: img.mimeType, data: img.dataBase64 };
  }
  const resp = await fetch(img.url as string);
  const data = Buffer.from(await resp.arrayBuffer()).toString("base64");
  const mimeType =
    img.mimeType ?? resp.headers.get("content-type") ?? "image/jpeg";
  return { mimeType, data };
}

/** OpenAI Chat Completions `content` (string when no images). */
export function openAIContent(prompt: string): string | Part[] {
  if (!checkIfMultimodal(prompt)) return prompt;
  return convertToMultiModalArray(prompt).map((p) =>
    typeof p === "string"
      ? { type: "text", text: p }
      : { type: "image_url", image_url: { url: imageSource(p) } },
  );
}

/** AI SDK message `content` parts (string when no images). */
export function aiSdkContent(prompt: string): string | Part[] {
  if (!checkIfMultimodal(prompt)) return prompt;
  return convertToMultiModalArray(prompt).map((p) =>
    typeof p === "string"
      ? { type: "text", text: p }
      : { type: "image", image: imageSource(p) },
  );
}

/** Anthropic message `content` (string when no images). */
export function anthropicContent(prompt: string): string | Part[] {
  if (!checkIfMultimodal(prompt)) return prompt;
  return convertToMultiModalArray(prompt).map((p) => {
    if (typeof p === "string") return { type: "text", text: p };
    if (p.url && !p.local) {
      return { type: "image", source: { type: "url", url: p.url } };
    }
    p.ensureImagesLoaded();
    return {
      type: "image",
      source: { type: "base64", media_type: p.mimeType, data: p.dataBase64 },
    };
  });
}

/** Gemini `contents` (the raw prompt string when no images; else Content[]). */
export async function geminiContents(prompt: string): Promise<unknown> {
  if (!checkIfMultimodal(prompt)) return prompt;
  const parts: Part[] = [];
  for (const p of convertToMultiModalArray(prompt)) {
    if (typeof p === "string") {
      parts.push({ text: p });
    } else {
      const { mimeType, data } = await imageBytes(p);
      parts.push({ inlineData: { mimeType, data } });
    }
  }
  return [{ role: "user", parts }];
}

/**
 * Ollama `messages` — one message per part, images carried in their own
 * message's `images` array as base64 (mirrors `generate_messages` in Python's
 * `OllamaModel`).
 */
export async function ollamaMessages(prompt: string): Promise<Part[]> {
  if (!checkIfMultimodal(prompt)) {
    return [{ role: "user", content: prompt }];
  }
  const messages: Part[] = [];
  for (const p of convertToMultiModalArray(prompt)) {
    if (typeof p === "string") {
      messages.push({ role: "user", content: p });
    } else {
      const { data } = await imageBytes(p);
      messages.push({ role: "user", images: [data] });
    }
  }
  return messages;
}

/**
 * The Converse API rejects anything outside this set, and it is case-sensitive.
 * Python uppercases the subtype here, which the service refuses; TS sends the
 * documented lowercase value.
 */
const BEDROCK_IMAGE_FORMATS = new Set(["png", "jpeg", "gif", "webp"]);

function bedrockImageFormat(mimeType: string | undefined): string {
  const subtype = (mimeType ?? "image/jpeg").split("/").pop()?.toLowerCase();
  const normalized = subtype === "jpg" ? "jpeg" : subtype;
  if (!normalized || !BEDROCK_IMAGE_FORMATS.has(normalized)) {
    throw new Error(
      `Amazon Bedrock does not accept '${mimeType}' images. Supported formats: ${[...BEDROCK_IMAGE_FORMATS].join(", ")}.`,
    );
  }
  return normalized;
}

/**
 * Bedrock Converse `content` blocks (mirrors `generate_payload` in Python's
 * `AmazonBedrockModel`). Bedrock takes no external URLs, so every image is
 * inlined as raw bytes.
 */
export async function bedrockContent(prompt: string): Promise<Part[]> {
  if (!checkIfMultimodal(prompt)) return [{ text: prompt }];
  const content: Part[] = [];
  for (const p of convertToMultiModalArray(prompt)) {
    if (typeof p === "string") {
      content.push({ text: p });
    } else {
      const { mimeType, data } = await imageBytes(p);
      content.push({
        image: {
          format: bedrockImageFormat(mimeType),
          source: { bytes: Buffer.from(data, "base64") },
        },
      });
    }
  }
  return content;
}
