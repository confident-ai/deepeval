import { defineConfig } from "prisma/config";

// The Prisma CLI does not read .env files. Export DATABASE_URL in the shell
// (or use `node --env-file=.env.local`) before running migrate commands;
// `prisma generate` does not need it.
export default defineConfig({
  schema: "prisma/schema.prisma",
  migrations: {
    path: "prisma/migrations",
  },
  datasource: {
    url: process.env.DATABASE_URL,
  },
});
