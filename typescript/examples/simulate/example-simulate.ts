import { ConversationSimulator } from "../../src/simulate";
import { ConversationalGolden } from "../../src/dataset";
import { Turn } from "../../src/test-case/llm-test-case";

const mockModelCallback = async (_args: {
  input: string;
  turns: Turn[];
  threadId: string;
}): Promise<Turn> => {
  return new Turn({
    role: "assistant",
    content:
      "I understand. How else can I assist you with your Coldplay concert ticket needs?",
  });
};

async function runSimulationExample() {
  const simulator = new ConversationSimulator({
    modelCallback: mockModelCallback,
  });
  const golden1 = new ConversationalGolden({
    scenario:
      "Andy Byron wants to purchase a VIP ticket to a cold play concert.",
    expectedOutcome: "Successful purchase of a ticket.",
    userDescription: "Andy Byron is the CEO of Astronomer.",
  });
  const golden2 = new ConversationalGolden({
    scenario:
      "Andy Byron wants to purchase a VIP ticket to a cold play concert.",
    expectedOutcome: "Successful purchase of a ticket.",
    userDescription: "Andy Byron is the CEO of Astronomer.",
    turns: [
      new Turn({
        role: "assistant",
        content:
          "Hi there, I'm interested in purchasing a VIP ticket to the upcoming Coldplay concert. Could you help me with the process?",
        retrievalContext: ["asdf", "asdf"],
      }),
    ],
  });

  try {
    const results1 = await simulator.simulate({
      conversationalGoldens: [golden1],
      maxUserSimulations: 3,
    });
    console.log("✅ Test Case 1 completed!");
    console.log(`Generated ${results1[0].turns?.length || 0} turns`);
    if (results1[0].turns) {
      console.log("\nConversation turns:");
      results1[0].turns.forEach((turn, index) => {
        console.log(`${index + 1}. ${turn.role}: ${turn.content}`);
      });
    }

    console.log("\n🚀 Running simulation for Test Case 2...");
    const results2 = await simulator.simulate({
      conversationalGoldens: [golden2],
      maxUserSimulations: 3,
    });
    console.log("✅ Test Case 2 completed!");
    console.log(`Generated ${results2[0].turns?.length || 0} turns`);

    if (results2[0].turns) {
      console.log("\nConversation turns:");
      results2[0].turns.forEach((turn, index) => {
        console.log(`${index + 1}. ${turn.role}: ${turn.content}`);
      });
    }

    console.log("\n🎉 All simulation examples completed successfully!");
  } catch (error) {
    console.error("❌ Error running simulation:", error);
  }
}

runSimulationExample();
