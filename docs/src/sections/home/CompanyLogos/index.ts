import type { LogoProps } from "./types";
import Adobe from "./Adobe";
import Aws from "./Aws";
import Axa from "./Axa";
import Benz from "./Benz";
import Bosch from "./Bosch";
import CvsHealth from "./CvsHealth";
import Ey from "./Ey";
import Google from "./Google";
import Lego from "./Lego";
import Mastercard from "./Mastercard";
import Microsoft from "./Microsoft";
import Nvidia from "./Nvidia";
import OpenAI from "./OpenAI";
import Pfizer from "./Pfizer";
import Samsung from "./Samsung";
import Siemens from "./Siemens";
import Toyota from "./Toyota";
import Uber from "./Uber";
import Visa from "./Visa";
import Walmart from "./Walmart";

export type { LogoProps } from "./types";

export const DYNAMIC_LOGOS: Record<string, React.FC<LogoProps>> = {
  adobe: Adobe,
  aws: Aws,
  axa: Axa,
  benz: Benz,
  bosch: Bosch,
  "cvs-health": CvsHealth,
  ey: Ey,
  google: Google,
  lego: Lego,
  mastercard: Mastercard,
  microsoft: Microsoft,
  nvidia: Nvidia,
  openai: OpenAI,
  pfizer: Pfizer,
  samsung: Samsung,
  siemens: Siemens,
  toyota: Toyota,
  uber: Uber,
  visa: Visa,
  walmart: Walmart,
};
