import { Command } from "commander";
import { editSettings, getSettings } from "@/config/settings";
import { postPersistedTestRun } from "@/evaluate/confident";
import {
  readLatestTestRun,
  saveLatestTestRun,
} from "@/evaluate/test-run/local";
import { coerceBlankToNull, prompt } from "@/cli/utils";
import { openBrowser, PROD, withUtm } from "@/cli/utm";

async function promptLogin(): Promise<boolean> {
  const loginUrl = withUtm(PROD, { content: "upload_and_open_link" });
  console.log(
    "🥳 Welcome to Confident AI, the evals cloud platform 🏡❤️\n" +
      `🔑 You'll need an API key from ${loginUrl} to view your results (free)`,
  );
  await openBrowser(loginUrl);

  for (;;) {
    const apiKey = coerceBlankToNull(
      await prompt("🔐 Enter your API Key: ", true),
    );
    if (apiKey) {
      editSettings(
        (draft) => {
          draft.CONFIDENT_API_KEY = apiKey;
        },
        { save: "dotenv:.env.local" },
      );
      console.log("\n🎉🥳 Congratulations! You've successfully logged in! 🙌");
      return true;
    }
    console.log("❌ API Key cannot be empty. Please try again.\n");
  }
}

export function registerViewCommand(program: Command): void {
  program
    .command("view")
    .description("Open the latest test run on Confident AI.")
    .action(async () => {
      const latest = readLatestTestRun();
      if (!latest) {
        console.log(
          "❌ No test run found in cache. Run `deepeval login` + an evaluation " +
            "to get started 🚀.",
        );
        process.exitCode = 1;
        return;
      }

      if (latest.link) {
        console.log(`🔗 View test run: ${latest.link}`);
        await openBrowser(latest.link);
        return;
      }

      const apiKey = getSettings().CONFIDENT_API_KEY;
      if (!apiKey || apiKey.trim() === "") {
        await promptLogin();
      }

      console.log("📤 Uploading test run to Confident AI...");
      const posted = await postPersistedTestRun(
        latest.cases,
        latest.runDuration,
        {
          official: latest.official,
          identifier: latest.identifier,
          hyperparameters: latest.hyperparameters,
        },
      );
      if (!posted.link) {
        console.log("❌ The test run could not be uploaded.");
        process.exitCode = 1;
        return;
      }

      latest.link = posted.link;
      latest.testRunId = posted.testRunId;
      saveLatestTestRun(latest);
      await openBrowser(posted.link);
    });
}
