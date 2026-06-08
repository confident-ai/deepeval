import { Prompt } from "../../src";
import OpenAI from "openai";

async function main() {
  const messagesPrompt = new Prompt({ alias: "Message Prompt" });
  await messagesPrompt.pull({ version: "00.00.01" });

  console.log(messagesPrompt.messagesTemplate);
  console.log(messagesPrompt.textTemplate);

  const textPrompt = new Prompt({ alias: "Text Prompt" });
  await textPrompt.pull({ version: "00.00.01" });
  const interpolated_prompt = textPrompt.interpolate({ name: "Joe" });

  console.log(messagesPrompt.messagesTemplate);
  console.log(messagesPrompt.textTemplate);
  console.log(interpolated_prompt);

  const openai = new OpenAI();
  const response = await openai.chat.completions.create({
    model: "gpt-3.5-turbo",
    messages: messagesPrompt.messagesTemplate as any[],
  });

  console.log(response.choices[0].message.content);
}

main();
