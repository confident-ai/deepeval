import type { LogoProps } from "./types";
import Aws from "./Aws";
import Benz from "./Benz";
import Bosch from "./Bosch";
import CvsHealth from "./CvsHealth";
import Ey from "./Ey";
import Mastercard from "./Mastercard";
import Nvidia from "./Nvidia";
import OpenAI from "./OpenAI";
import Toyota from "./Toyota";
import Uber from "./Uber";

export type { LogoProps } from "./types";

export const DYNAMIC_LOGOS: Record<string, React.FC<LogoProps>> = {
  aws: Aws,
  benz: Benz,
  bosch: Bosch,
  "cvs-health": CvsHealth,
  ey: Ey,
  mastercard: Mastercard,
  nvidia: Nvidia,
  openai: OpenAI,
  toyota: Toyota,
  uber: Uber,
};
