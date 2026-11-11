import { PrismaPg } from "@prisma/adapter-pg";
import { PrismaClient } from "@/generated/prisma/client";

// Kept on `globalThis` so dev-server reloads reuse one connection pool
// instead of opening a new one per reload.
const globalForDb = globalThis as typeof globalThis & {
  benchmarkDb?: PrismaClient;
};

// Created on first use, so importing this module never requires
// DATABASE_URL (type generation and builds do not connect).
export function getDb(): PrismaClient {
  if (!globalForDb.benchmarkDb) {
    const connectionString = process.env.DATABASE_URL;
    if (!connectionString) {
      throw new Error("DATABASE_URL is not set.");
    }
    globalForDb.benchmarkDb = new PrismaClient({
      adapter: new PrismaPg(connectionString),
    });
  }
  return globalForDb.benchmarkDb;
}
