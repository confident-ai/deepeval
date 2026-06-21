import { MLLMImage } from "../../test-case";

type Part = string | MLLMImage;

/** Indices of the image elements in a parsed multimodal array. */
export function getImageIndices(parts: Part[]): number[] {
  const indices: number[] = [];
  parts.forEach((p, i) => {
    if (p instanceof MLLMImage) indices.push(i);
  });
  return indices;
}

/** Split a multimodal array into its text strings + images (mirrors `separate_images_from_text`). */
export function separateImagesFromText(parts: Part[]): [string[], MLLMImage[]] {
  const texts: string[] = [];
  const images: MLLMImage[] = [];
  for (const p of parts) {
    if (p instanceof MLLMImage) images.push(p);
    else texts.push(p);
  }
  return [texts, images];
}

/**
 * The nearest text above/below an image (mirrors `get_image_context`). Each is
 * clipped to `maxContextSize` chars (above keeps the tail, below the head).
 */
export function getImageContext(
  imageIndex: number,
  parts: Part[],
  maxContextSize?: number,
): [string | undefined, string | undefined] {
  let above: string | undefined;
  let below: string | undefined;
  for (let i = imageIndex - 1; i >= 0; i--) {
    if (typeof parts[i] === "string") {
      above = parts[i] as string;
      if (maxContextSize) above = above.slice(-maxContextSize);
      break;
    }
  }
  for (let i = imageIndex + 1; i < parts.length; i++) {
    if (typeof parts[i] === "string") {
      below = parts[i] as string;
      if (maxContextSize) below = below.slice(0, maxContextSize);
      break;
    }
  }
  return [above, below];
}
