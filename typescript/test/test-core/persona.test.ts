import {
  ConversationalGolden,
  DEFAULT_GOLDEN_KEY_NAMES,
  Persona,
  goldenFromRecord,
} from "@/dataset";

const DELIMITERS = { context: "|", retrievalContext: "|" };

describe("Persona on ConversationalGolden", () => {
  let warn: jest.SpyInstance;

  beforeEach(() => {
    warn = jest.spyOn(console, "warn").mockImplementation(() => {});
  });

  afterEach(() => {
    warn.mockRestore();
  });

  test("keeps userDescription in sync so the Confident AI payload is unchanged", () => {
    const golden = new ConversationalGolden({
      scenario: "Buying a ticket.",
      persona: new Persona({ characteristics: "The CEO of Astronomer." }),
    });

    expect(golden.persona?.characteristics).toBe("The CEO of Astronomer.");
    expect(golden.userDescription).toBe("The CEO of Astronomer.");
    expect(warn).not.toHaveBeenCalled();
  });

  test("promotes a deprecated userDescription to a persona, with a warning", () => {
    const golden = new ConversationalGolden({
      scenario: "Buying a ticket.",
      userDescription: "The CEO of Astronomer.",
    });

    expect(golden.persona?.characteristics).toBe("The CEO of Astronomer.");
    expect(golden.userDescription).toBe("The CEO of Astronomer.");
    expect(warn).toHaveBeenCalledTimes(1);
  });

  test("rejects a persona and userDescription that disagree", () => {
    expect(
      () =>
        new ConversationalGolden({
          scenario: "Buying a ticket.",
          persona: new Persona({ characteristics: "Impatient." }),
          userDescription: "Patient.",
        }),
    ).toThrow(/not both with conflicting text/);
  });

  test("reads a persona object off a loaded record", () => {
    const golden = goldenFromRecord(
      {
        scenario: "Buying a ticket.",
        persona: { name: "Andy", characteristics: "The CEO of Astronomer." },
      },
      DEFAULT_GOLDEN_KEY_NAMES,
      DELIMITERS,
    ) as ConversationalGolden;

    expect(golden.persona?.name).toBe("Andy");
    expect(golden.userDescription).toBe("The CEO of Astronomer.");
    expect(warn).not.toHaveBeenCalled();
  });

  test("upgrades a legacy user_description row", () => {
    const golden = goldenFromRecord(
      { scenario: "Buying a ticket.", user_description: "An old row." },
      DEFAULT_GOLDEN_KEY_NAMES,
      DELIMITERS,
    ) as ConversationalGolden;

    expect(golden.persona?.characteristics).toBe("An old row.");
    expect(warn).toHaveBeenCalledTimes(1);
  });
});
