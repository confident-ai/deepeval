// Jest setup file for handling missing API keys in tests
// This file is automatically loaded by Jest before running tests

// Check if CONFIDENT_API_KEY is set
const hasConfidentApiKey = !!process.env.CONFIDENT_API_KEY;

// Store the original console.error to avoid duplicate messages
const originalConsoleError = console.error;

// If no API key is present, provide a setup message
if (!hasConfidentApiKey) {
  console.info(
    "Note: CONFIDENT_API_KEY not set. Tests requiring API calls will fail gracefully or be skipped.",
  );

  // Optional: Suppress specific error messages to reduce noise
  console.error = (...args) => {
    // Only suppress the "API key not found" message once
    const message = args[0]?.toString?.() || "";
    if (
      message.includes("Confident AI API key not found") &&
      message.includes("Please provide")
    ) {
      return; // Skip the specific error message about API key
    }
    return originalConsoleError.apply(console, args);
  };
}
