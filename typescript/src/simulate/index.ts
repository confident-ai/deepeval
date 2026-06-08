import { ConversationalGolden } from "../dataset";
import { ConversationalTestCase, Turn } from "../test-case/llm-test-case";
import { Api, HttpMethods, Endpoints } from "../confident/api";
import * as cliProgress from "cli-progress";

export interface SimulateHttpResponse {
  userResponse: string;
  completed: boolean;
  [key: string]: any;
}

type ModelCallback = (args: {
  input: string;
  turns: Turn[];
  threadId: string;
}) => Promise<Turn>;

export class ConversationSimulator {
  private readonly modelCallback: ModelCallback;

  constructor(opts: { modelCallback: ModelCallback }) {
    this.modelCallback = opts.modelCallback;
  }

  async simulate(args: {
    conversationalGoldens: ConversationalGolden[];
    maxUserSimulations?: number;
    showProgress?: boolean;
  }): Promise<ConversationalTestCase[]> {
    const maxUserSimulations = args.maxUserSimulations || 10;
    const showProgress = args.showProgress !== false;

    const multibar = showProgress
      ? new cliProgress.MultiBar(
          {
            clearOnComplete: true,
            hideCursor: true,
            format: "{label} {bar} | {percentage}%",
          },
          cliProgress.Presets.shades_classic,
        )
      : null;

    const overallProgressBar = multibar
      ? multibar.create(args.conversationalGoldens.length, 0, {
          label: "Simulating conversations:",
        })
      : null;

    const conversationPromises = args.conversationalGoldens.map((golden, i) => {
      return this.simulateSingleConversation(
        golden,
        maxUserSimulations,
        multibar || undefined,
        i + 1,
        overallProgressBar || undefined,
      );
    });

    const results = await Promise.all(conversationPromises);

    if (multibar) {
      multibar.stop();
    }

    console.log(
      `\u2705 Successfully simulated ${args.conversationalGoldens.length} conversations`,
    );

    return results;
  }

  private async simulateSingleConversation(
    golden: ConversationalGolden,
    maxUserSimulations: number,
    multibar?: cliProgress.MultiBar,
    conversationNumber?: number,
    overallProgressBar?: cliProgress.SingleBar,
  ): Promise<ConversationalTestCase> {
    if (maxUserSimulations <= 0) {
      throw new Error("maxUserSimulations must be greater than 0");
    }

    let turns: Turn[] = [];
    const threadId = crypto.randomUUID();
    let simulationCounter = 0;

    if (golden.turns) {
      turns = turns.concat(golden.turns);
      const lastRole = turns[turns.length - 1].role;
      if (lastRole === "user") {
        const lastUserInput = turns[turns.length - 1].content;
        const assistantTurn: Turn = await this.modelCallback({
          input: lastUserInput,
          turns,
          threadId,
        });
        turns.push(assistantTurn);
      }
    }

    const conversationProgressBar = multibar
      ? multibar.create(maxUserSimulations, 0, {
          label: `  Simulating conversation #${conversationNumber || 1}:`,
        })
      : null;

    while (true) {
      const shouldStop = await this.stopConversation(turns, golden);
      if (shouldStop) {
        break;
      }
      if (simulationCounter >= maxUserSimulations) {
        break;
      }

      if (turns.length === 0 || turns[turns.length - 1].role !== "user") {
        const userInput = await this.generateNextUserInput(golden, turns);
        turns.push(new Turn({ role: "user", content: userInput }));
        simulationCounter++;

        if (conversationProgressBar) {
          conversationProgressBar.update(simulationCounter);
        }
      }

      const lastUserInput = turns[turns.length - 1].content;
      const assistantTurn: Turn = await this.modelCallback({
        input: lastUserInput,
        turns,
        threadId,
      });
      turns.push(assistantTurn);
    }

    if (conversationProgressBar) {
      if (multibar) {
        multibar.remove(conversationProgressBar);
      }
    }

    if (overallProgressBar) {
      overallProgressBar.increment();
    }

    return new ConversationalTestCase({
      turns,
      scenario: golden.scenario,
      userDescription: golden.userDescription,
      expectedOutcome: golden.expectedOutcome,
      context: golden.context,
      name: golden.name,
      additionalMetadata: golden.additionalMetadata,
      comments: golden.comments,
      _datasetRank: golden._datasetRank,
      _datasetAlias: golden._datasetAlias,
      _datasetId: golden._datasetId,
    });
  }

  private async generateNextUserInput(
    golden: ConversationalGolden,
    turns: Turn[],
  ): Promise<string> {
    const payload = this.dumpConversationalGolden({ ...golden, turns });
    const res = await this.callApi<SimulateHttpResponse>(payload);
    return res.data.userResponse;
  }

  private async stopConversation(
    turns: Turn[],
    golden: ConversationalGolden,
  ): Promise<boolean> {
    const payload = this.dumpConversationalGolden({ ...golden, turns });
    const res = await this.callApi<SimulateHttpResponse>(payload);
    return res.data.completed;
  }

  private dumpConversationalGolden(golden: ConversationalGolden) {
    const body = {
      conversationalGolden: {
        scenario: golden.scenario,
        expectedOutcome: golden.expectedOutcome,
        userDescription: golden.userDescription,
        turns: golden.turns?.map((turn) => {
          return {
            role: turn.role,
            content: turn.content,
            userId: turn.userId,
            retrievalContext: turn.retrievalContext,
            toolsCalled: turn.toolsCalled,
          };
        }),
      },
    };
    return body;
  }

  private async callApi<T>(body: any): Promise<T> {
    const api = new Api();
    const response = await api.sendRequest(
      HttpMethods.POST,
      Endpoints.SIMULATE_ENDPOINT,
      body,
    );
    return response as T;
  }
}
