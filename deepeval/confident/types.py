from datetime import datetime
from typing import Any, List, Optional

from pydantic import BaseModel, Field

from deepeval.utils import make_model_config


############################################
### API
############################################


class ApiResponse(BaseModel):
    model_config = make_model_config(extra="ignore")

    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    deprecated: Optional[bool] = None
    link: Optional[str] = None


class ConfidentApiError(Exception):
    """Custom exception that preserves API response metadata"""

    def __init__(self, message: str, link: Optional[str] = None):
        super().__init__(message)
        self.link = link


############################################
### Models
############################################


class Organization(BaseModel):
    model_config = make_model_config(extra="ignore")

    id: str
    name: str
    plan: Optional[str] = None
    created_at: Optional[datetime] = None


class Project(BaseModel):
    model_config = make_model_config(extra="ignore")

    id: str
    name: str
    description: Optional[str] = None
    organization_id: Optional[str] = Field(None, alias="organizationId")
    created_at: Optional[datetime] = None


class RoleRef(BaseModel):
    model_config = make_model_config(extra="ignore")

    id: str
    name: str


class PolicyRef(BaseModel):
    model_config = make_model_config(extra="ignore")

    id: str
    name: str


class PermissionRef(BaseModel):
    model_config = make_model_config(extra="ignore")

    id: str
    name: str


class Member(BaseModel):
    model_config = make_model_config(extra="ignore")

    id: str
    email: str
    name: Optional[str] = None
    image: Optional[str] = None
    organization_role: Optional[RoleRef] = Field(None, alias="organizationRole")
    project_role: Optional[RoleRef] = Field(None, alias="projectRole")


class Invitation(BaseModel):
    model_config = make_model_config(extra="ignore")

    id: int
    email: str
    status: Optional[str] = None
    created_at: Optional[datetime] = None
    organization_role_id: Optional[str] = Field(
        None, alias="organizationRoleId"
    )
    project_role_id: Optional[str] = Field(None, alias="projectRoleId")


class Role(BaseModel):
    model_config = make_model_config(extra="ignore")

    id: str
    name: str
    description: Optional[str] = None
    policies: List[PolicyRef] = Field(default_factory=list)
    organization_id: Optional[str] = Field(None, alias="organizationId")
    project_id: Optional[str] = Field(None, alias="projectId")


class Policy(BaseModel):
    model_config = make_model_config(extra="ignore")

    id: str
    name: str
    description: Optional[str] = None
    permissions: List[PermissionRef] = Field(default_factory=list)


class Permission(BaseModel):
    model_config = make_model_config(extra="ignore")

    id: str
    name: str
    description: Optional[str] = None


class ApiKey(BaseModel):
    model_config = make_model_config(extra="ignore")

    id: int
    name: str
    valid: Optional[bool] = None
    value: str
    created_at: Optional[datetime] = None
    last_used: Optional[datetime] = Field(None, alias="lastUsed")


############################################
### Requests
############################################


class UpdateOrganizationRequest(BaseModel):
    name: str


class CreateProjectRequest(BaseModel):
    name: str
    description: Optional[str] = None
    email: Optional[str] = None


class UpdateProjectRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None


class UpdateMemberRoleRequest(BaseModel):
    role_id: str = Field(serialization_alias="roleId")


class CreateOrganizationInvitationsRequest(BaseModel):
    emails: List[str]
    role_id: Optional[str] = Field(
        default=None, serialization_alias="organizationRoleId"
    )


class CreateProjectInvitationsRequest(BaseModel):
    emails: List[str]
    role_id: Optional[str] = Field(
        default=None, serialization_alias="projectRoleId"
    )


class RoleRequest(BaseModel):
    name: str
    policy_ids: List[str] = Field(serialization_alias="policyIds")
    description: Optional[str] = None


class PolicyRequest(BaseModel):
    name: str
    permission_ids: List[str] = Field(serialization_alias="permissionIds")
    description: Optional[str] = None


class CreateApiKeyRequest(BaseModel):
    name: str


class UpdateApiKeyRequest(BaseModel):
    valid: bool


############################################
### Responses
############################################


class OrganizationHttpResponse(BaseModel):
    organization: Organization


class ProjectHttpResponse(BaseModel):
    project: Project


class ProjectsHttpResponse(BaseModel):
    projects: List[Project]


class MemberHttpResponse(BaseModel):
    member: Member


class MembersHttpResponse(BaseModel):
    members: List[Member]


class InvitationHttpResponse(BaseModel):
    invitation: Invitation


class InvitationsHttpResponse(BaseModel):
    invitations: List[Invitation]


class RoleHttpResponse(BaseModel):
    role: Role


class RolesHttpResponse(BaseModel):
    roles: List[Role]


class PolicyHttpResponse(BaseModel):
    policy: Policy


class PoliciesHttpResponse(BaseModel):
    policies: List[Policy]


class PermissionsHttpResponse(BaseModel):
    permissions: List[Permission]


class ApiKeyHttpResponse(BaseModel):
    api_key: ApiKey = Field(alias="apiKey")


class ApiKeysHttpResponse(BaseModel):
    api_keys: List[ApiKey] = Field(alias="apiKeys")
