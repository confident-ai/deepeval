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

from typing import Dict, List, Literal, Optional

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


class CliOnboardingContext(BaseModel):
    """Typed GET /cli/onboarding response.

    `user` is set for new users; `projects` is populated for existing users.
    """

    state: Literal["new_user", "existing_user"]
    user: Optional[CliOnboardingUser] = None
    projects: List[CliOnboardingProject] = Field(default_factory=list)


class NewUserOnboardingRequest(BaseModel):
    """POST /cli/onboarding/complete body for a first-time user; the backend
    creates their organization and first project."""

    user_name: str = Field(serialization_alias="userName")
    organization_name: str = Field(serialization_alias="organizationName")
    project_name: str = Field(serialization_alias="projectName")

    def to_payload(self) -> Dict[str, str]:
        return self.model_dump(by_alias=True)


class ExistingProjectKeyRequest(BaseModel):
    """POST /cli/onboarding/complete body for an existing user; the backend
    issues a key for one of their projects."""

    project_id: str = Field(serialization_alias="projectId")

    def to_payload(self) -> Dict[str, str]:
        return self.model_dump(by_alias=True)
