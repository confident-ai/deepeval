import { config } from "dotenv";
import { randomUUID } from "crypto";
import { Prompt } from "../../src";

config();

interface PromptMessage {
  role: string;
  content: string;
}

export const runSharedBranchingTests = (alias: string) => {
  test("Should create a new branch", async () => {
    const uuid = randomUUID();
    const newBranchName = `new-branch-${uuid}`;
    const prompt = new Prompt({ alias });

    await prompt.createBranch(newBranchName);

    // Pull all branches
    const branches = await prompt.getBranches();
    const branchNames = branches.map((branch) => branch.name);

    expect(branchNames).toContain(newBranchName);
  });

  test("Should update an existing branch name", async () => {
    const uuid = randomUUID();
    const oldBranchName = `old-branch-${uuid}`;
    const newBranchName = `new-branch-${uuid}`;

    const prompt = new Prompt({ alias });

    // Create the initial branch
    await prompt.createBranch(oldBranchName);

    // Verify it was created
    let branches = await prompt.getBranches();
    let branchNames = branches.map((branch) => branch.name);
    expect(branchNames).toContain(oldBranchName);

    // Update the branch
    await prompt.updateBranch(newBranchName, oldBranchName);

    // Verify the update
    branches = await prompt.getBranches();
    branchNames = branches.map((branch) => branch.name);

    expect(branchNames).toContain(newBranchName);
    expect(branchNames).not.toContain(oldBranchName);
  });

  test("Should delete a branch", async () => {
    const uuid = randomUUID();
    const branchToDelete = `delete-branch-${uuid}`;

    const prompt = new Prompt({ alias });

    // Create the branch first
    await prompt.createBranch(branchToDelete);

    // Verify it exists
    let branches = await prompt.getBranches();
    let branchNames = branches.map((branch) => branch.name);
    expect(branchNames).toContain(branchToDelete);

    // Delete the branch
    await prompt.deleteBranch(branchToDelete);

    // Verify it was deleted
    branches = await prompt.getBranches();
    branchNames = branches.map((branch) => branch.name);

    expect(branchNames).not.toContain(branchToDelete);
  });
};

describe("Prompt Module", () => {
  test("Should pull a prompt from Confident AI", async () => {
    const alias = "asd";
    const prompt = new Prompt({ alias: alias });

    try {
      const response = await prompt.pull();

      expect(response).toBeDefined();
      // FIX: Access nested data object from API response
      const promptData = (response as any).data;
      expect(promptData.messages).toBeDefined();

      const message = promptData.messages[0] as PromptMessage;
      expect(typeof message.content).toBe("string");
    } catch (error) {
      console.error("Test asd failed with an error:", error);
      throw error;
    }
  });

  test("Should pull a prompt with version from Confident AI", async () => {
    const alias = "asd";
    const version = "00.00.01";
    const prompt = new Prompt({ alias: alias });

    try {
      const response = await prompt.pull({ version: version });
      expect(response).toBeDefined();
    } catch (error) {
      console.error("Test asd with version failed with an error:", error);
      throw error;
    }
  });

  test("Should interpolate a pulled prompt from Confident AI", async () => {
    const alias = "TEXT";
    const version = "00.00.01";
    const prompt = new Prompt({ alias: alias });

    try {
      await prompt.pull({ version: version });
      const kwargs = { x: "World" };
      const interpolatedResult = prompt.interpolate(kwargs);

      expect(interpolatedResult).toBeDefined();
      // FIX: Cast to PromptMessage to resolve TS7053 indexing error
      const message = interpolatedResult[0] as PromptMessage;
      expect(typeof message.content).toBe("string");
      expect(message.content).toContain("Hello, World!");
    } catch (error) {
      console.error("Test TEXT failed with an error:", error);
      throw error;
    }
  });

  test("Should interpolate a pulled prompt from Confident AI (dollar brackets)", async () => {
    const alias = "DOLLAR";
    const version = "00.00.01";
    const prompt = new Prompt({ alias: alias });

    try {
      await prompt.pull({ version: version });
      const kwargs = { x: "World" };
      const interpolatedResult = prompt.interpolate(kwargs);

      expect(interpolatedResult).toBeDefined();
      const message = interpolatedResult[0] as PromptMessage;
      expect(typeof message.content).toBe("string");
      expect(message.content).toContain("Hello, World!");
    } catch (error) {
      console.error("Test DOLLAR failed with an error:", error);
      throw error;
    }
  });

  test("Should interpolate a pulled prompt from Confident AI (mustache)", async () => {
    const alias = "MUSTACHE";
    const version = "00.00.01";
    const prompt = new Prompt({ alias: alias });

    try {
      await prompt.pull({ version: version });
      const kwargs = { x: "World" };
      const interpolatedResult = prompt.interpolate(kwargs);

      expect(interpolatedResult).toBeDefined();
      const message = interpolatedResult[0] as PromptMessage;
      expect(typeof message.content).toBe("string");
      expect(message.content).toContain("Hello, World!");
    } catch (error) {
      console.error("Test MUSTACHE failed with an error:", error);
      throw error;
    }
  });

  test("Should interpolate a pulled prompt from Confident AI (mustache with space)", async () => {
    const alias = "MUSTACHE_WITH_SPACE";
    const version = "00.00.01";
    const prompt = new Prompt({ alias: alias });

    try {
      await prompt.pull({ version: version });
      const kwargs = { x: "World" };
      const interpolatedResult = prompt.interpolate(kwargs);

      expect(interpolatedResult).toBeDefined();
      const message = interpolatedResult[0] as PromptMessage;
      expect(typeof message.content).toBe("string");
      expect(message.content).toContain("Hello, World!");
    } catch (error) {
      console.error("Test MUSTACHE_WITH_SPACE failed with an error:", error);
      throw error;
    }
  });

  test("Should interpolate a pulled prompt from Confident AI (list)", async () => {
    const alias = "LIST";
    const version = "00.00.01";
    const prompt = new Prompt({ alias: alias });

    try {
      const response = await prompt.pull({ version: version });
      const promptData = (response as any).data;

      const kwargs = { x: "World" };
      const interpolatedResult = prompt.interpolate(kwargs);

      expect(promptData).toBeDefined();
      expect(promptData.type).toEqual("LIST");

      const message = interpolatedResult[0] as PromptMessage;
      expect(typeof message.content).toBe("string");
      expect(message.content).toContain("Hello, World!");
    } catch (error) {
      console.error("Test LIST failed with an error:", error);
      throw error;
    }
  });

  describe("Push with array schema definitions", () => {
    test("Should push and pull a prompt with array of strings schema", async () => {
      const ALIAS = "ts_test_prompt_list_strings_schema";
      const prompt = new Prompt({ alias: ALIAS });
      const uuid = randomUUID();

      await prompt.push({
        text: `Generate tags ${uuid}`,
        outputType: "SCHEMA",
        outputSchema: {
          name: "ListOfStringsSchema",
          fields: {
            tags: ["string"],
          },
        },
      });

      const pulledPrompt = new Prompt({ alias: ALIAS });
      const response = await pulledPrompt.pull();
      const data = (response as any).data;

      expect(data.outputType).toBe("SCHEMA");
      expect(data.outputSchema).toBeDefined();
      expect(data.outputSchema.fields).toBeDefined();

      const arrayField = data.outputSchema.fields.find(
        (f: any) => f.name === "tags" && !f.parentId,
      );
      expect(arrayField).toBeDefined();
      expect(arrayField.type).toBe("ARRAY");
    });

    test("Should push and pull a prompt with array of ints schema", async () => {
      const ALIAS = "ts_test_prompt_list_ints_schema";
      const prompt = new Prompt({ alias: ALIAS });
      const uuid = randomUUID();

      await prompt.push({
        text: `Generate scores ${uuid}`,
        outputType: "SCHEMA",
        outputSchema: {
          name: "ListOfIntsSchema",
          fields: {
            scores: ["integer"],
          },
        },
      });

      const pulledPrompt = new Prompt({ alias: ALIAS });
      const response = await pulledPrompt.pull();
      const data = (response as any).data;

      expect(data.outputType).toBe("SCHEMA");
      expect(data.outputSchema).toBeDefined();

      const arrayField = data.outputSchema.fields.find(
        (f: any) => f.name === "scores" && !f.parentId,
      );
      expect(arrayField).toBeDefined();
      expect(arrayField.type).toBe("ARRAY");
    });

    test("Should push and pull a prompt with array of floats schema", async () => {
      const ALIAS = "ts_test_prompt_list_floats_schema";
      const prompt = new Prompt({ alias: ALIAS });
      const uuid = randomUUID();

      await prompt.push({
        text: `Generate values ${uuid}`,
        outputType: "SCHEMA",
        outputSchema: {
          name: "ListOfFloatsSchema",
          fields: {
            values: ["float"],
          },
        },
      });

      const pulledPrompt = new Prompt({ alias: ALIAS });
      const response = await pulledPrompt.pull();
      const data = (response as any).data;

      expect(data.outputType).toBe("SCHEMA");
      expect(data.outputSchema).toBeDefined();

      const arrayField = data.outputSchema.fields.find(
        (f: any) => f.name === "values" && !f.parentId,
      );
      expect(arrayField).toBeDefined();
      expect(arrayField.type).toBe("ARRAY");
    });

    test("Should push and pull a prompt with array of objects schema", async () => {
      const ALIAS = "ts_test_prompt_list_objects_schema";
      const prompt = new Prompt({ alias: ALIAS });
      const uuid = randomUUID();

      await prompt.push({
        text: `Generate sources ${uuid}`,
        outputType: "SCHEMA",
        outputSchema: {
          name: "ListOfObjectsSchema",
          fields: {
            sources: [{ url: "string", title: "string" }],
          },
        },
      });

      const pulledPrompt = new Prompt({ alias: ALIAS });
      const response = await pulledPrompt.pull();
      const data = (response as any).data;

      expect(data.outputType).toBe("SCHEMA");
      expect(data.outputSchema).toBeDefined();

      const arrayField = data.outputSchema.fields.find(
        (f: any) => f.name === "sources" && !f.parentId,
      );
      expect(arrayField).toBeDefined();
      expect(arrayField.type).toBe("ARRAY");

      const objectItem = data.outputSchema.fields.find(
        (f: any) => f.parentId === arrayField.id && f.type === "OBJECT",
      );
      expect(objectItem).toBeDefined();

      const nestedFields = data.outputSchema.fields.filter(
        (f: any) => f.parentId === objectItem.id,
      );
      const nestedNames = nestedFields.map((f: any) => f.name);
      expect(nestedNames).toContain("url");
      expect(nestedNames).toContain("title");
    });

    test("Should push and pull a prompt with mixed schema including lists", async () => {
      const ALIAS = "ts_test_prompt_mixed_list_schema";
      const prompt = new Prompt({ alias: ALIAS });
      const uuid = randomUUID();

      await prompt.push({
        text: `Generate mixed data ${uuid}`,
        outputType: "SCHEMA",
        outputSchema: {
          name: "MixedSchemaWithLists",
          fields: {
            name: "string",
            count: "integer",
            tags: ["string"],
            sources: [{ url: "string", title: "string" }],
          },
        },
      });

      const pulledPrompt = new Prompt({ alias: ALIAS });
      const response = await pulledPrompt.pull();
      const data = (response as any).data;

      expect(data.outputType).toBe("SCHEMA");
      expect(data.outputSchema).toBeDefined();

      const rootFields = data.outputSchema.fields.filter(
        (f: any) => !f.parentId,
      );
      const rootNames = rootFields.map((f: any) => f.name);
      expect(rootNames).toContain("name");
      expect(rootNames).toContain("count");
      expect(rootNames).toContain("tags");
      expect(rootNames).toContain("sources");

      const tagsField = rootFields.find((f: any) => f.name === "tags");
      expect(tagsField.type).toBe("ARRAY");

      const sourcesField = rootFields.find((f: any) => f.name === "sources");
      expect(sourcesField.type).toBe("ARRAY");
    });

    test("Should push and pull a prompt with nested object containing a list", async () => {
      const ALIAS = "ts_test_prompt_nested_obj_with_list_schema";
      const prompt = new Prompt({ alias: ALIAS });
      const uuid = randomUUID();

      await prompt.push({
        text: `Generate nested list data ${uuid}`,
        outputType: "SCHEMA",
        outputSchema: {
          name: "SchemaWithNestedObjectContainingList",
          fields: {
            id: "string",
            details: {
              title: "string",
              items: [{ label: "string", score: "float" }],
            },
          },
        },
      });

      const pulledPrompt = new Prompt({ alias: ALIAS });
      const response = await pulledPrompt.pull();
      const data = (response as any).data;

      expect(data.outputType).toBe("SCHEMA");
      expect(data.outputSchema).toBeDefined();

      const rootFields = data.outputSchema.fields.filter(
        (f: any) => !f.parentId,
      );
      const rootNames = rootFields.map((f: any) => f.name);
      expect(rootNames).toContain("id");
      expect(rootNames).toContain("details");

      const detailsField = rootFields.find((f: any) => f.name === "details");
      expect(detailsField.type).toBe("OBJECT");
    });
  });

  describe("Prompt Branching - Text Type", () => {
    const BRANCH_ALIAS = "test_branch";
    const BRANCH_NAME = "test_branch_name";

    test("Should push to a new branch and main branch by default", async () => {
      const prompt = new Prompt({ alias: BRANCH_ALIAS });

      // 1. Push to main branch
      await prompt.push({ text: "Main branch push" });
      const firstBranchHash = prompt.hash;

      // 2. Push to a different branch
      await prompt.push({
        text: "Different branch push",
        branch: BRANCH_NAME,
      });
      const secondBranchHash = prompt.hash;

      // 3. Fetch commits for main branch
      const mainCommits = await prompt.getCommits("main");
      const mainBranchHashes = mainCommits.map((commit: any) => commit.hash);

      // 4. Fetch commits for the new branch
      const branchCommits = await prompt.getCommits(BRANCH_NAME);
      const branchHashes = branchCommits.map((commit: any) => commit.hash);

      // 5. Assertions
      expect(mainBranchHashes).toContain(firstBranchHash);
      expect(branchHashes).toContain(secondBranchHash);
    });

    runSharedBranchingTests(BRANCH_ALIAS);
  });

  describe("Prompt Branching - List Type", () => {
    const BRANCH_ALIAS = "test_branch_messages";
    const BRANCH_NAME = "test_branch_name";

    test("Should push messages to a new branch and main branch by default", async () => {
      const prompt = new Prompt({ alias: BRANCH_ALIAS });

      // 1. Push to main branch
      await prompt.push({
        messages: [{ role: "user", content: "New branch push" }],
      });
      const firstBranchHash = prompt.hash;

      // 2. Push to a different branch
      await prompt.push({
        messages: [{ role: "user", content: "New branch push" }],
        branch: BRANCH_NAME,
      });
      const secondBranchHash = prompt.hash;

      // 3. Fetch commits for main branch
      const mainCommits = await prompt.getCommits("main");
      const mainBranchHashes = mainCommits.map((commit: any) => commit.hash);

      // 4. Fetch commits for the new branch
      const branchCommits = await prompt.getCommits(BRANCH_NAME);
      const branchHashes = branchCommits.map((commit: any) => commit.hash);

      // 5. Assertions
      expect(mainBranchHashes).toContain(firstBranchHash);
      expect(branchHashes).toContain(secondBranchHash);
    });

    runSharedBranchingTests(BRANCH_ALIAS);
  });
});
