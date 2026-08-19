import { Command } from "commander";
import { assessGovernance } from "@/governance";

export function registerGateCommand(program: Command): void {
  program
    .command("gate")
    .description(
      "Check your project against its governance policy and exit with a " +
        "non-zero code if it doesn't pass.",
    )
    .option(
      "-q, --quiet",
      "Suppress output. The exit code still reflects the verdict.",
    )
    .action(async (options: { quiet?: boolean }) => {
      const quiet = !!options.quiet;
      try {
        const { passed, governancePolicy } = await assessGovernance();
        const policyName = governancePolicy.name || "governance policy";

        if (passed) {
          if (!quiet) {
            console.log(`✅ Governance gate passed against ${policyName}.`);
          }
          process.exit(0);
        }

        if (!quiet) {
          console.error(
            `❌ Governance gate failed against ${policyName}. ` +
              "One or more controls did not pass.",
          );
        }
        process.exit(1);
      } catch (error) {
        if (!quiet) {
          const message =
            error instanceof Error ? error.message : String(error);
          console.error(
            `❌ Could not assess governance for your project: ${message}\n` +
              "Make sure your project is associated with a governance policy. " +
              "If it isn't, please contact your organization administrator.",
          );
        }
        process.exit(1);
      }
    });
}
