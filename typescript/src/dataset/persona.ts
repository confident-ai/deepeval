/**
 * Who the simulated user is, as opposed to what they are trying to do.
 *
 * A persona pairs with a `ConversationalGolden`: the persona is *who* the user
 * is, while the golden's `scenario` and `expectedOutcome` are *what* they want.
 * Keep behavioral traits here and task instructions on the golden.
 */
export class Persona {
  name?: string;
  /**
   * The free-form persona prompt: demographics, personality, emotional arc,
   * speaking style — who the user is, never what they want. Successor to
   * `ConversationalGolden.userDescription`.
   */
  characteristics: string;

  constructor(params: { characteristics: string; name?: string }) {
    this.characteristics = params.characteristics;
    this.name = params.name;
  }
}

/**
 * Reconcile the `persona` field with the deprecated `userDescription`.
 *
 * Both are kept in sync so `userDescription` keeps flowing to Confident AI and
 * through CSV/JSON datasets while `userDescription` is being retired.
 */
export function resolvePersona(
  persona?: Persona,
  userDescription?: string,
): { persona?: Persona; userDescription?: string } {
  if (persona === undefined) {
    if (userDescription === undefined) {
      return { persona: undefined, userDescription: undefined };
    }
    console.warn(
      "'userDescription' is deprecated and will be removed in a future release. Use 'persona: new Persona({ characteristics: ... })' instead.",
    );
    return {
      persona: new Persona({ characteristics: userDescription }),
      userDescription,
    };
  }
  if (
    userDescription !== undefined &&
    userDescription !== persona.characteristics
  ) {
    throw new Error(
      "Pass either 'persona' or the deprecated 'userDescription', not both with conflicting text.",
    );
  }
  return { persona, userDescription: persona.characteristics };
}

/** Dump a persona for file output, omitting defaults to keep rows small. */
export function serializePersona(
  persona?: Persona,
): Record<string, unknown> | null {
  if (!persona) return null;
  return persona.name
    ? { name: persona.name, characteristics: persona.characteristics }
    : { characteristics: persona.characteristics };
}

/**
 * Build the `ConversationalGolden` persona field from one loaded row.
 *
 * Files written before personas existed only carry `userDescription`, which the
 * golden upgrades (with a deprecation warning) on construction.
 */
export function personaFromRecord(rawPersona: unknown): Persona | undefined {
  if (rawPersona instanceof Persona) return rawPersona;
  if (rawPersona && typeof rawPersona === "object") {
    const { characteristics, name } = rawPersona as {
      characteristics?: string;
      name?: string;
    };
    if (characteristics) return new Persona({ characteristics, name });
  }
  return undefined;
}
