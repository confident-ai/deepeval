import { config } from "dotenv";

/**
 * Utility functions for Confident AI integration
 */

// Load environment variables from .env file
try {
  // Using require instead of import for conditional loading
  config();
} catch (_error) {
  // Silently continue if dotenv is not available
  // This allows the package to work without dotenv as a dependency
}

/**
 * Check if the user is authenticated with Confident AI
 * @returns boolean indicating if the user is authenticated
 */
export function isConfident(): boolean {
  // Check for API key in environment variables
  const confidentApiKey = process.env.CONFIDENT_API_KEY;
  const isConfident = !!confidentApiKey && confidentApiKey.trim() !== "";
  if (!isConfident) {
    console.error("Confident AI API key not found.");
  }
  return isConfident;
}

/**
 * Get the Confident AI API key from environment variables
 * @returns The API key or null if not found
 */
export function getConfidentApiKey(): string | null {
  const apiKey = process.env.CONFIDENT_API_KEY;
  return apiKey && apiKey.trim() !== "" ? apiKey : null;
}

/**
 * Convert object keys from camelCase to snake_case
 * @param data - The data to convert
 * @returns The data with keys converted to snake_case
 */
export function convertKeysToSnakeCase(data: any): any {
  if (data === null || data === undefined) {
    return data;
  }

  if (typeof data === "object" && !Array.isArray(data)) {
    const newObj: Record<string, any> = {};
    for (const [key, value] of Object.entries(data)) {
      const newKey = camelToSnake(key);
      if (key === "additionalMetadata") {
        // Convert key but do not recurse into value
        newObj[newKey] = value;
      } else {
        newObj[newKey] = convertKeysToSnakeCase(value);
      }
    }
    return newObj;
  } else if (Array.isArray(data)) {
    return data.map((item) => convertKeysToSnakeCase(item));
  }

  return data;
}

/**
 * Convert a camelCase string to snake_case
 * @param str - The string to convert
 * @returns The snake_case version of the string
 */
function camelToSnake(str: string): string {
  return str.replace(/[A-Z]/g, (letter) => `_${letter.toLowerCase()}`);
}

/**
 * Utility function to wait for a specified time
 * @param ms - Milliseconds to wait
 * @returns Promise that resolves after the specified time
 */
export function wait(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
