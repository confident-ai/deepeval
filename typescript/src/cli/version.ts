import * as fs from "fs";
import * as path from "path";

export function getVersion(): string {
  for (const candidate of ["../../package.json", "../package.json"]) {
    try {
      const raw = fs.readFileSync(path.join(__dirname, candidate), "utf-8");
      const version = (JSON.parse(raw) as { version?: string }).version;
      if (version) return version;
    } catch {}
  }
  return "unknown";
}
