const fs = require("fs");

const dataset = JSON.parse(
    fs.readFileSync(
        "examples/rag_evaluation/browser_chatbot_evaluation/dataset.json",
        "utf-8"
    )
);

function fakeChatbot(input) {
    if (input.includes("Dubai")) return "Dubai is a city in the UAE";
    if (input.includes("UAE")) return "Abu Dhabi is the capital of UAE";
    if (input.includes("2+2")) return "2+2 equals 4";
    return "I don't know";
}

const outputs = dataset.map(item => ({
    input: item.input,
    actual_output: fakeChatbot(item.input)
}));

fs.writeFileSync(
    "examples/rag_evaluation/browser_chatbot_evaluation/outputs.json",
    JSON.stringify(outputs, null, 2)
);

console.log("✅ outputs.json generated");