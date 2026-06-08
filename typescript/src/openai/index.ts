import { patchOpenAI } from "./patch";

let registered = false;

function instrumentOpenAI(client: any) {
  if (registered) {
    return;
  }

  patchOpenAI(client);
  registered = true;
}

export { instrumentOpenAI };
