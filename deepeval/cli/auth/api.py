"""Pydantic models for the Confident AI CLI login/onboarding API.

These mirror the backend contracts:
- request bodies validated by the backend's Zod schemas
  (`DeviceCodeRequestSchema`, `UserCodeRequestSchema`,
  `CompleteCliOnboardingRequestSchema`), and
- the response payloads returned by /cli/auth/sessions* and /cli/onboarding*.

Field names are snake_case in Python and map to the camelCase wire format
through validation/serialization aliases: responses are parsed with
`Model.model_validate(data)` and request bodies serialize via
`to_payload()`. Fields without a default are required — if the server omits
them, parsing fails with a `pydantic.ValidationError` instead of silently
substituting a value.
"""

from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

DEFAULT_EXPIRES_IN_SECONDS = 600
DEFAULT_POLL_INTERVAL_SECONDS = 3


class DevicePairing(BaseModel):
    """Parsed POST /cli/auth/sessions response."""

    model_config = ConfigDict(populate_by_name=True)

    user_code: str = Field(validation_alias="userCode")
    device_code: str = Field(validation_alias="deviceCode")
    verification_url: str = Field(validation_alias="verificationUriComplete")
    expires_in: int = Field(
        DEFAULT_EXPIRES_IN_SECONDS, validation_alias="expiresIn"
    )
    interval: int = DEFAULT_POLL_INTERVAL_SECONDS


class CliAuthorization(BaseModel):
    """Parsed POST /cli/auth/sessions/token response once the browser has
    authorized the pairing."""

    model_config = ConfigDict(populate_by_name=True)

    setup_token: str = Field(validation_alias="setupToken")
    email: Optional[str] = None


class CliOnboardingUser(BaseModel):
    """`user` in the GET /cli/onboarding response."""

    name: Optional[str] = None
    email: Optional[str] = None


class CliOnboardingProject(BaseModel):
    """One entry of `projects` in the GET /cli/onboarding response."""

    model_config = ConfigDict(populate_by_name=True)

    id: str
    name: str
    can_create_api_key: bool = Field(validation_alias="canCreateApiKey")


class CliOnboardingOrganization(BaseModel):
    """`organization` in an existing user's GET /cli/onboarding response."""

    id: str
    name: str


QuestionnaireAnswer = Union[str, bool, List[str]]


class CliQuestionnaireOption(BaseModel):
    """One selectable answer in a server-provided CLI question."""

    model_config = ConfigDict(populate_by_name=True)

    label: str
    value: Union[str, bool]
    exclusive: bool = False
    accepts_custom_value: bool = Field(
        False, validation_alias="acceptsCustomValue"
    )
    custom_prompt: Optional[str] = Field(None, validation_alias="customPrompt")


class CliQuestionnaireQuestionBase(BaseModel):
    id: str
    prompt: str
    required: bool


class CliTextQuestion(CliQuestionnaireQuestionBase):
    model_config = ConfigDict(populate_by_name=True)

    type: Literal["text"]
    default_value: Optional[str] = Field(None, validation_alias="defaultValue")
    max_length: Optional[int] = Field(None, validation_alias="maxLength")


class CliSingleSelectQuestion(CliQuestionnaireQuestionBase):
    model_config = ConfigDict(populate_by_name=True)

    type: Literal["single_select"]
    options: List[CliQuestionnaireOption]
    default_value: Optional[Union[str, bool]] = Field(
        None, validation_alias="defaultValue"
    )


class CliMultiSelectQuestion(CliQuestionnaireQuestionBase):
    model_config = ConfigDict(populate_by_name=True)

    type: Literal["multi_select"]
    options: List[CliQuestionnaireOption]
    min_selections: Optional[int] = Field(
        None, validation_alias="minSelections", ge=0
    )


CliQuestionnaireQuestion = Annotated[
    Union[
        CliTextQuestion,
        CliSingleSelectQuestion,
        CliMultiSelectQuestion,
    ],
    Field(discriminator="type"),
]


class CliQuestionnaire(BaseModel):
    """Versioned questions returned for a new user's CLI setup."""

    version: int
    questions: List[CliQuestionnaireQuestion]


class CliOnboardingContext(BaseModel):
    """Typed GET /cli/onboarding response.

    `user` is set for new users; `projects` is populated for existing users.
    """

    state: Literal["new_user", "existing_user"]
    user: Optional[CliOnboardingUser] = None
    organization: Optional[CliOnboardingOrganization] = None
    projects: List[CliOnboardingProject] = Field(default_factory=list)
    questionnaire: Optional[CliQuestionnaire] = None


class NewUserOnboardingRequest(BaseModel):
    """POST /cli/onboarding/complete body for a first-time user; the backend
    creates their organization and first project."""

    user_name: str = Field(serialization_alias="userName")
    organization_name: str = Field(serialization_alias="organizationName")
    project_name: str = Field(serialization_alias="projectName")
    development_stage: Literal["IDEATION", "DEVELOPMENT", "PRODUCTION"] = Field(
        serialization_alias="developmentStage"
    )
    interaction_type: Literal["SINGLE_TURN", "MULTI_TURN"] = Field(
        serialization_alias="interactionType"
    )
    modalities: List[Literal["TEXT", "IMAGE", "AUDIO"]]
    user_facing: bool = Field(serialization_alias="userFacing")
    external_resources: List[Literal["TOOL_CALL", "MCP", "RAG"]] = Field(
        serialization_alias="externalResources"
    )
    description: str

    def to_payload(self) -> Dict[str, Any]:
        return self.model_dump(by_alias=True)


class DynamicNewUserOnboardingRequest(BaseModel):
    """POST /cli/onboarding/complete body driven by the server questionnaire."""

    questionnaire_version: int = Field(
        serialization_alias="questionnaireVersion"
    )
    questionnaire_answers: Dict[str, QuestionnaireAnswer] = Field(
        serialization_alias="questionnaireAnswers"
    )

    def to_payload(self) -> Dict[str, Any]:
        return self.model_dump(by_alias=True)


class ExistingProjectKeyRequest(BaseModel):
    """POST /cli/onboarding/complete body for an existing user; the backend
    issues a key for one of their projects."""

    project_id: str = Field(serialization_alias="projectId")

    def to_payload(self) -> Dict[str, str]:
        return self.model_dump(by_alias=True)
