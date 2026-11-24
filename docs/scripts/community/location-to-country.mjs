/**
 * Free-text location -> ISO 3166-1 alpha-2 country code, for GitHub
 * profile locations. Shared by generate-contributor-countries.mjs and
 * generate-stargazer-countries.mjs.
 */

const norm = (s) =>
  s
    .normalize("NFD")
    .replace(/[̀-ͯ]/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9 ]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();

// Retired or grouping codes that Intl still names like a current country
// (DD is "Germany", FX "France", VD "Vietnam", ...); never map to them.
const RETIRED = new Set(["AN", "BU", "CS", "DD", "DY", "EU", "EZ", "FX", "HV", "NH", "NT", "QO", "RH", "SU", "TP", "UK", "UN", "VD", "XA", "XB", "YD", "YU", "ZR", "ZZ"]);

const BY_NAME = new Map();
{
  const names = new Intl.DisplayNames(["en"], { type: "region" });
  for (let i = 0; i < 26; i++) {
    for (let j = 0; j < 26; j++) {
      const code = String.fromCharCode(65 + i, 65 + j);
      try {
        const name = names.of(code);
        if (name && name !== code && !RETIRED.has(code)) BY_NAME.set(norm(name), code);
      } catch {
        // not a region
      }
    }
  }
}

// Common ways people write a country, and cities and US states that are
// often written alone.
const ALIASES = {
  usa: "US", us: "US", "u s a": "US", "united states of america": "US", america: "US",
  uk: "GB", "u k": "GB", england: "GB", scotland: "GB", wales: "GB", "great britain": "GB", britain: "GB",
  deutschland: "DE", espana: "ES", italia: "IT", nederland: "NL", holland: "NL", "the netherlands": "NL",
  schweiz: "CH", suisse: "CH", osterreich: "AT", polska: "PL", turkiye: "TR", korea: "KR",
  "south korea": "KR", "republic of korea": "KR", uae: "AE", prc: "CN", "hong kong sar": "HK",
  "viet nam": "VN", "czech republic": "CZ", russia: "RU", "brasil": "BR", "mexico": "MX",
  // cities
  "san francisco": "US", "bay area": "US", "sf": "US", "new york": "US", nyc: "US", seattle: "US",
  "los angeles": "US", boston: "US", austin: "US", chicago: "US", "palo alto": "US", "mountain view": "US",
  "san jose": "US", "menlo park": "US", berkeley: "US", oakland: "US", denver: "US", atlanta: "US",
  "silicon valley": "US", "washington dc": "US", "san diego": "US", miami: "US", "pittsburgh": "US",
  london: "GB", cambridge: "GB", oxford: "GB", manchester: "GB", edinburgh: "GB",
  bangalore: "IN", bengaluru: "IN", mumbai: "IN", delhi: "IN", "new delhi": "IN", hyderabad: "IN",
  pune: "IN", chennai: "IN", kolkata: "IN", gurgaon: "IN", gurugram: "IN", noida: "IN", ahmedabad: "IN",
  berlin: "DE", munich: "DE", hamburg: "DE", paris: "FR", amsterdam: "NL", madrid: "ES", barcelona: "ES",
  lisbon: "PT", zurich: "CH", stockholm: "SE", copenhagen: "DK", oslo: "NO", helsinki: "FI",
  dublin: "IE", warsaw: "PL", prague: "CZ", vienna: "AT", milan: "IT", rome: "IT", "tel aviv": "IL",
  toronto: "CA", vancouver: "CA", montreal: "CA", waterloo: "CA", sydney: "AU", melbourne: "AU",
  singapore: "SG", tokyo: "JP", seoul: "KR", beijing: "CN", shanghai: "CN", shenzhen: "CN",
  hangzhou: "CN", taipei: "TW", "hong kong": "HK", jakarta: "ID", manila: "PH", "kuala lumpur": "MY",
  bangkok: "TH", "ho chi minh city": "VN", hanoi: "VN", dubai: "AE", "abu dhabi": "AE", riyadh: "SA",
  istanbul: "TR", cairo: "EG", lagos: "NG", nairobi: "KE", "cape town": "ZA", johannesburg: "ZA",
  "sao paulo": "BR", "rio de janeiro": "BR", "buenos aires": "AR", bogota: "CO", "mexico city": "MX",
  santiago: "CL", lima: "PE", karachi: "PK", lahore: "PK", islamabad: "PK", dhaka: "BD", colombo: "LK",
  kathmandu: "NP", kyiv: "UA", kiev: "UA", bucharest: "RO", budapest: "HU", athens: "GR",
};

const US_STATES = new Set(
  "al ak az ar ca co ct de fl ga hi id il in ia ks ky la me md ma mi mn ms mo mt ne nv nh nj nm ny nc nd oh ok or pa ri sc sd tn tx ut vt va wa wv wi wy dc"
    .split(" ")
    .concat(
      "alabama alaska arizona arkansas california colorado connecticut delaware florida georgia hawaii idaho illinois indiana iowa kansas kentucky louisiana maine maryland massachusetts michigan minnesota mississippi missouri montana nebraska nevada new hampshire new jersey new mexico north carolina north dakota ohio oklahoma oregon pennsylvania rhode island south carolina south dakota tennessee texas utah vermont virginia washington west virginia wisconsin wyoming".split(
        / (?=[a-z])/
      )
    )
);

// Names long enough to search for inside longer text, compiled once.
const ANYWHERE = [...BY_NAME, ...Object.entries(ALIASES)]
  .filter(([name]) => name.length > 3)
  .map(([name, code]) => [new RegExp(`\\b${name}\\b`), code]);

/** A free-text location ("Pune, India", "SF Bay Area") -> alpha-2 code, or null. */
export function resolve(location) {
  if (!location) return null;
  const parts = location.split(/[,/|·•\-–]+/).map(norm).filter(Boolean);
  // Most specific last: "Pune, Maharashtra, India" -> try "india" first.
  for (const p of [...parts].reverse()) {
    if (ALIASES[p]) return ALIASES[p];
    if (BY_NAME.has(p)) return BY_NAME.get(p);
    if (US_STATES.has(p) && parts.length > 1) return "US";
  }
  const whole = norm(location);
  if (ALIASES[whole]) return ALIASES[whole];
  if (BY_NAME.has(whole)) return BY_NAME.get(whole);
  // A known name anywhere in the text, e.g. "Working from India".
  for (const [re, code] of ANYWHERE) if (re.test(whole)) return code;
  return null;
}

