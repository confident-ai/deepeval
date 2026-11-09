import type {
  Onboarding,
  QuestionnaireAnswer,
  QuestionnaireAnswers,
} from "./api.js";

export const selectProject = (
  onboarding: Extract<Onboarding, { state: "existing_user" }>,
  projectId?: string,
): { id: string; name: string } | undefined => {
  const manageable = onboarding.projects.filter(
    (project) => project.canCreateApiKey,
  );
  if (!projectId) {
    if (manageable.length === 1) {
      const project = manageable[0]!;
      return { id: project.id, name: project.name };
    }
    return undefined;
  }
  const project = onboarding.projects.find(
    (candidate) => candidate.id === projectId,
  );
  if (!project) throw new Error(`Project ${projectId} is not available.`);
  if (!project.canCreateApiKey) {
    throw new Error(
      `You do not have permission to create an API key for ${project.name}.`,
    );
  }
  return { id: project.id, name: project.name };
};

export const validateOrganization = (
  onboarding: Onboarding,
  organizationId?: string,
): void => {
  if (!organizationId) return;
  if (
    onboarding.state !== "existing_user" ||
    onboarding.organization?.id !== organizationId
  ) {
    throw new Error(
      `Browser authorization did not resolve organization ${organizationId}.`,
    );
  }
};

export const setQuestionnaireAnswer = (
  answers: QuestionnaireAnswers,
  questionId: string,
  value: QuestionnaireAnswer,
): QuestionnaireAnswers => ({ ...answers, [questionId]: value });

export const applyCustomSelection = (
  selections: string[],
  customMarker: string,
  customValue: string,
): string[] =>
  selections.map((value) =>
    value === customMarker ? customValue.trim() : value,
  );

export const enforceExclusiveSelection = (
  selections: string[],
  exclusiveValues: string[],
): string[] => {
  const selectedExclusive = selections.find((value) =>
    exclusiveValues.includes(value),
  );
  return selectedExclusive ? [selectedExclusive] : selections;
};
