from typing import List, Optional

from deepeval.confident.api import Api, Endpoints, HttpMethods
from deepeval.confident.types import (
    ApiKey,
    ApiKeyHttpResponse,
    ApiKeysHttpResponse,
    CreateApiKeyRequest,
    CreateOrganizationInvitationsRequest,
    CreateProjectInvitationsRequest,
    CreateProjectRequest,
    Invitation,
    InvitationHttpResponse,
    InvitationsHttpResponse,
    Member,
    MemberHttpResponse,
    MembersHttpResponse,
    Organization,
    OrganizationHttpResponse,
    Permission,
    PermissionsHttpResponse,
    PoliciesHttpResponse,
    Policy,
    PolicyHttpResponse,
    PolicyRequest,
    Project,
    ProjectHttpResponse,
    ProjectsHttpResponse,
    Role,
    RoleHttpResponse,
    RoleRequest,
    RolesHttpResponse,
    UpdateApiKeyRequest,
    UpdateMemberRoleRequest,
    UpdateOrganizationRequest,
    UpdateProjectRequest,
)


class ConfidentClient:
    def __init__(self, api_key: Optional[str] = None):
        self._api = Api(api_key=api_key)

    ############################################
    ### Organization
    ############################################

    def get_organization(self) -> Organization:
        data, _ = self._api.send_request(
            method=HttpMethods.GET,
            endpoint=Endpoints.ORGANIZATION_ENDPOINT,
        )
        return OrganizationHttpResponse(**data).organization

    def update_organization(self, name: str) -> Organization:
        body = UpdateOrganizationRequest(name=name).model_dump(
            by_alias=True, exclude_none=True
        )
        data, _ = self._api.send_request(
            method=HttpMethods.PUT,
            endpoint=Endpoints.ORGANIZATION_ENDPOINT,
            body=body,
        )
        return OrganizationHttpResponse(**data).organization

    ############################################
    ### Projects
    ############################################

    def list_projects(self) -> List[Project]:
        data, _ = self._api.send_request(
            method=HttpMethods.GET,
            endpoint=Endpoints.PROJECTS_ENDPOINT,
        )
        return ProjectsHttpResponse(**data).projects

    def create_project(
        self,
        name: str,
        description: Optional[str] = None,
        email: Optional[str] = None,
    ) -> Project:
        body = CreateProjectRequest(
            name=name, description=description, email=email
        ).model_dump(by_alias=True, exclude_none=True)
        data, _ = self._api.send_request(
            method=HttpMethods.POST,
            endpoint=Endpoints.PROJECTS_ENDPOINT,
            body=body,
        )
        return ProjectHttpResponse(**data).project

    def get_project(self, project_id: str) -> Project:
        data, _ = self._api.send_request(
            method=HttpMethods.GET,
            endpoint=Endpoints.PROJECT_ENDPOINT,
            url_params={"projectId": project_id},
        )
        return ProjectHttpResponse(**data).project

    def update_project(
        self,
        project_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Project:
        body = UpdateProjectRequest(
            name=name, description=description
        ).model_dump(by_alias=True, exclude_none=True)
        data, _ = self._api.send_request(
            method=HttpMethods.PUT,
            endpoint=Endpoints.PROJECT_ENDPOINT,
            url_params={"projectId": project_id},
            body=body,
        )
        return ProjectHttpResponse(**data).project

    def delete_project(self, project_id: str) -> None:
        self._api.send_request(
            method=HttpMethods.DELETE,
            endpoint=Endpoints.PROJECT_ENDPOINT,
            url_params={"projectId": project_id},
        )

    ############################################
    ### Members
    ############################################

    def list_organization_members(
        self, page: int = 1, page_size: int = 25
    ) -> List[Member]:
        data, _ = self._api.send_request(
            method=HttpMethods.GET,
            endpoint=Endpoints.ORGANIZATION_MEMBERS_ENDPOINT,
            params={"page": page, "pageSize": page_size},
        )
        return MembersHttpResponse(**data).members

    def update_organization_member_role(
        self, user_id: str, role_id: str
    ) -> Member:
        body = UpdateMemberRoleRequest(role_id=role_id).model_dump(
            by_alias=True, exclude_none=True
        )
        data, _ = self._api.send_request(
            method=HttpMethods.PUT,
            endpoint=Endpoints.ORGANIZATION_MEMBER_ENDPOINT,
            url_params={"userId": user_id},
            body=body,
        )
        return MemberHttpResponse(**data).member

    def remove_organization_member(self, user_id: str) -> None:
        self._api.send_request(
            method=HttpMethods.DELETE,
            endpoint=Endpoints.ORGANIZATION_MEMBER_ENDPOINT,
            url_params={"userId": user_id},
        )

    def list_project_members(
        self, project_id: str, page: int = 1, page_size: int = 25
    ) -> List[Member]:
        data, _ = self._api.send_request(
            method=HttpMethods.GET,
            endpoint=Endpoints.PROJECT_MEMBERS_ENDPOINT,
            url_params={"projectId": project_id},
            params={"page": page, "pageSize": page_size},
        )
        return MembersHttpResponse(**data).members

    def update_project_member_role(
        self, project_id: str, user_id: str, role_id: str
    ) -> Member:
        body = UpdateMemberRoleRequest(role_id=role_id).model_dump(
            by_alias=True, exclude_none=True
        )
        data, _ = self._api.send_request(
            method=HttpMethods.PUT,
            endpoint=Endpoints.PROJECT_MEMBER_ENDPOINT,
            url_params={"projectId": project_id, "userId": user_id},
            body=body,
        )
        return MemberHttpResponse(**data).member

    def remove_project_member(self, project_id: str, user_id: str) -> None:
        self._api.send_request(
            method=HttpMethods.DELETE,
            endpoint=Endpoints.PROJECT_MEMBER_ENDPOINT,
            url_params={"projectId": project_id, "userId": user_id},
        )

    ############################################
    ### Invitations
    ############################################

    def list_organization_invitations(self) -> List[Invitation]:
        data, _ = self._api.send_request(
            method=HttpMethods.GET,
            endpoint=Endpoints.ORGANIZATION_INVITATIONS_ENDPOINT,
        )
        return InvitationsHttpResponse(**data).invitations

    def create_organization_invitations(
        self, emails: List[str], role_id: Optional[str] = None
    ) -> List[Invitation]:
        body = CreateOrganizationInvitationsRequest(
            emails=emails, role_id=role_id
        ).model_dump(by_alias=True, exclude_none=True)
        data, _ = self._api.send_request(
            method=HttpMethods.POST,
            endpoint=Endpoints.ORGANIZATION_INVITATIONS_ENDPOINT,
            body=body,
        )
        return InvitationsHttpResponse(**data).invitations

    def resend_organization_invitation(self, invitation_id: int) -> Invitation:
        data, _ = self._api.send_request(
            method=HttpMethods.PUT,
            endpoint=Endpoints.ORGANIZATION_INVITATION_ENDPOINT,
            url_params={"invitationId": invitation_id},
        )
        return InvitationHttpResponse(**data).invitation

    def delete_organization_invitation(self, invitation_id: int) -> None:
        self._api.send_request(
            method=HttpMethods.DELETE,
            endpoint=Endpoints.ORGANIZATION_INVITATION_ENDPOINT,
            url_params={"invitationId": invitation_id},
        )

    def list_project_invitations(self, project_id: str) -> List[Invitation]:
        data, _ = self._api.send_request(
            method=HttpMethods.GET,
            endpoint=Endpoints.PROJECT_INVITATIONS_ENDPOINT,
            url_params={"projectId": project_id},
        )
        return InvitationsHttpResponse(**data).invitations

    def create_project_invitations(
        self,
        project_id: str,
        emails: List[str],
        role_id: Optional[str] = None,
    ) -> List[Invitation]:
        body = CreateProjectInvitationsRequest(
            emails=emails, role_id=role_id
        ).model_dump(by_alias=True, exclude_none=True)
        data, _ = self._api.send_request(
            method=HttpMethods.POST,
            endpoint=Endpoints.PROJECT_INVITATIONS_ENDPOINT,
            url_params={"projectId": project_id},
            body=body,
        )
        return InvitationsHttpResponse(**data).invitations

    def resend_project_invitation(
        self, project_id: str, invitation_id: int
    ) -> Invitation:
        data, _ = self._api.send_request(
            method=HttpMethods.PUT,
            endpoint=Endpoints.PROJECT_INVITATION_ENDPOINT,
            url_params={
                "projectId": project_id,
                "invitationId": invitation_id,
            },
        )
        return InvitationHttpResponse(**data).invitation

    def delete_project_invitation(
        self, project_id: str, invitation_id: int
    ) -> None:
        self._api.send_request(
            method=HttpMethods.DELETE,
            endpoint=Endpoints.PROJECT_INVITATION_ENDPOINT,
            url_params={
                "projectId": project_id,
                "invitationId": invitation_id,
            },
        )

    ############################################
    ### Roles
    ############################################

    def list_organization_roles(self) -> List[Role]:
        data, _ = self._api.send_request(
            method=HttpMethods.GET,
            endpoint=Endpoints.ORGANIZATION_ROLES_ENDPOINT,
        )
        return RolesHttpResponse(**data).roles

    def create_organization_role(
        self,
        name: str,
        policy_ids: List[str],
        description: Optional[str] = None,
    ) -> Role:
        body = RoleRequest(
            name=name, policy_ids=policy_ids, description=description
        ).model_dump(by_alias=True, exclude_none=True)
        data, _ = self._api.send_request(
            method=HttpMethods.POST,
            endpoint=Endpoints.ORGANIZATION_ROLES_ENDPOINT,
            body=body,
        )
        return RoleHttpResponse(**data).role

    def update_organization_role(
        self,
        role_id: str,
        name: str,
        policy_ids: List[str],
        description: Optional[str] = None,
    ) -> Role:
        body = RoleRequest(
            name=name, policy_ids=policy_ids, description=description
        ).model_dump(by_alias=True, exclude_none=True)
        data, _ = self._api.send_request(
            method=HttpMethods.PUT,
            endpoint=Endpoints.ORGANIZATION_ROLE_ENDPOINT,
            url_params={"roleId": role_id},
            body=body,
        )
        return RoleHttpResponse(**data).role

    def delete_organization_role(self, role_id: str) -> None:
        self._api.send_request(
            method=HttpMethods.DELETE,
            endpoint=Endpoints.ORGANIZATION_ROLE_ENDPOINT,
            url_params={"roleId": role_id},
        )

    def list_project_roles(self, project_id: str) -> List[Role]:
        data, _ = self._api.send_request(
            method=HttpMethods.GET,
            endpoint=Endpoints.PROJECT_ROLES_ENDPOINT,
            url_params={"projectId": project_id},
        )
        return RolesHttpResponse(**data).roles

    def create_project_role(
        self,
        project_id: str,
        name: str,
        policy_ids: List[str],
        description: Optional[str] = None,
    ) -> Role:
        body = RoleRequest(
            name=name, policy_ids=policy_ids, description=description
        ).model_dump(by_alias=True, exclude_none=True)
        data, _ = self._api.send_request(
            method=HttpMethods.POST,
            endpoint=Endpoints.PROJECT_ROLES_ENDPOINT,
            url_params={"projectId": project_id},
            body=body,
        )
        return RoleHttpResponse(**data).role

    def update_project_role(
        self,
        project_id: str,
        role_id: str,
        name: str,
        policy_ids: List[str],
        description: Optional[str] = None,
    ) -> Role:
        body = RoleRequest(
            name=name, policy_ids=policy_ids, description=description
        ).model_dump(by_alias=True, exclude_none=True)
        data, _ = self._api.send_request(
            method=HttpMethods.PUT,
            endpoint=Endpoints.PROJECT_ROLE_ENDPOINT,
            url_params={"projectId": project_id, "roleId": role_id},
            body=body,
        )
        return RoleHttpResponse(**data).role

    def delete_project_role(self, project_id: str, role_id: str) -> None:
        self._api.send_request(
            method=HttpMethods.DELETE,
            endpoint=Endpoints.PROJECT_ROLE_ENDPOINT,
            url_params={"projectId": project_id, "roleId": role_id},
        )

    ############################################
    ### Policies
    ############################################

    def list_organization_policies(self) -> List[Policy]:
        data, _ = self._api.send_request(
            method=HttpMethods.GET,
            endpoint=Endpoints.ORGANIZATION_POLICIES_ENDPOINT,
        )
        return PoliciesHttpResponse(**data).policies

    def create_organization_policy(
        self,
        name: str,
        permission_ids: List[str],
        description: Optional[str] = None,
    ) -> Policy:
        body = PolicyRequest(
            name=name,
            permission_ids=permission_ids,
            description=description,
        ).model_dump(by_alias=True, exclude_none=True)
        data, _ = self._api.send_request(
            method=HttpMethods.POST,
            endpoint=Endpoints.ORGANIZATION_POLICIES_ENDPOINT,
            body=body,
        )
        return PolicyHttpResponse(**data).policy

    def update_organization_policy(
        self,
        policy_id: str,
        name: str,
        permission_ids: List[str],
        description: Optional[str] = None,
    ) -> Policy:
        body = PolicyRequest(
            name=name,
            permission_ids=permission_ids,
            description=description,
        ).model_dump(by_alias=True, exclude_none=True)
        data, _ = self._api.send_request(
            method=HttpMethods.PUT,
            endpoint=Endpoints.ORGANIZATION_POLICY_ENDPOINT,
            url_params={"policyId": policy_id},
            body=body,
        )
        return PolicyHttpResponse(**data).policy

    def delete_organization_policy(self, policy_id: str) -> None:
        self._api.send_request(
            method=HttpMethods.DELETE,
            endpoint=Endpoints.ORGANIZATION_POLICY_ENDPOINT,
            url_params={"policyId": policy_id},
        )

    def list_project_policies(self, project_id: str) -> List[Policy]:
        data, _ = self._api.send_request(
            method=HttpMethods.GET,
            endpoint=Endpoints.PROJECT_POLICIES_ENDPOINT,
            url_params={"projectId": project_id},
        )
        return PoliciesHttpResponse(**data).policies

    def create_project_policy(
        self,
        project_id: str,
        name: str,
        permission_ids: List[str],
        description: Optional[str] = None,
    ) -> Policy:
        body = PolicyRequest(
            name=name,
            permission_ids=permission_ids,
            description=description,
        ).model_dump(by_alias=True, exclude_none=True)
        data, _ = self._api.send_request(
            method=HttpMethods.POST,
            endpoint=Endpoints.PROJECT_POLICIES_ENDPOINT,
            url_params={"projectId": project_id},
            body=body,
        )
        return PolicyHttpResponse(**data).policy

    def update_project_policy(
        self,
        project_id: str,
        policy_id: str,
        name: str,
        permission_ids: List[str],
        description: Optional[str] = None,
    ) -> Policy:
        body = PolicyRequest(
            name=name,
            permission_ids=permission_ids,
            description=description,
        ).model_dump(by_alias=True, exclude_none=True)
        data, _ = self._api.send_request(
            method=HttpMethods.PUT,
            endpoint=Endpoints.PROJECT_POLICY_ENDPOINT,
            url_params={"projectId": project_id, "policyId": policy_id},
            body=body,
        )
        return PolicyHttpResponse(**data).policy

    def delete_project_policy(self, project_id: str, policy_id: str) -> None:
        self._api.send_request(
            method=HttpMethods.DELETE,
            endpoint=Endpoints.PROJECT_POLICY_ENDPOINT,
            url_params={"projectId": project_id, "policyId": policy_id},
        )

    ############################################
    ### Permissions
    ############################################

    def list_organization_permissions(self) -> List[Permission]:
        data, _ = self._api.send_request(
            method=HttpMethods.GET,
            endpoint=Endpoints.ORGANIZATION_PERMISSIONS_ENDPOINT,
        )
        return PermissionsHttpResponse(**data).permissions

    def list_project_permissions(self, project_id: str) -> List[Permission]:
        data, _ = self._api.send_request(
            method=HttpMethods.GET,
            endpoint=Endpoints.PROJECT_PERMISSIONS_ENDPOINT,
            url_params={"projectId": project_id},
        )
        return PermissionsHttpResponse(**data).permissions

    ############################################
    ### API Keys
    ############################################

    def list_organization_api_keys(self) -> List[ApiKey]:
        data, _ = self._api.send_request(
            method=HttpMethods.GET,
            endpoint=Endpoints.ORGANIZATION_API_KEYS_ENDPOINT,
        )
        return ApiKeysHttpResponse(**data).api_keys

    def get_organization_api_key(self, api_key_id: int) -> ApiKey:
        data, _ = self._api.send_request(
            method=HttpMethods.GET,
            endpoint=Endpoints.ORGANIZATION_API_KEY_ENDPOINT,
            url_params={"apiKeyId": api_key_id},
        )
        return ApiKeyHttpResponse(**data).api_key

    def create_organization_api_key(self, name: str) -> ApiKey:
        body = CreateApiKeyRequest(name=name).model_dump(
            by_alias=True, exclude_none=True
        )
        data, _ = self._api.send_request(
            method=HttpMethods.POST,
            endpoint=Endpoints.ORGANIZATION_API_KEYS_ENDPOINT,
            body=body,
        )
        return ApiKeyHttpResponse(**data).api_key

    def update_organization_api_key(
        self, api_key_id: int, valid: bool
    ) -> ApiKey:
        body = UpdateApiKeyRequest(valid=valid).model_dump(
            by_alias=True, exclude_none=True
        )
        data, _ = self._api.send_request(
            method=HttpMethods.PUT,
            endpoint=Endpoints.ORGANIZATION_API_KEY_ENDPOINT,
            url_params={"apiKeyId": api_key_id},
            body=body,
        )
        return ApiKeyHttpResponse(**data).api_key

    def delete_organization_api_key(self, api_key_id: int) -> None:
        self._api.send_request(
            method=HttpMethods.DELETE,
            endpoint=Endpoints.ORGANIZATION_API_KEY_ENDPOINT,
            url_params={"apiKeyId": api_key_id},
        )

    def list_project_api_keys(self, project_id: str) -> List[ApiKey]:
        data, _ = self._api.send_request(
            method=HttpMethods.GET,
            endpoint=Endpoints.PROJECT_API_KEYS_ENDPOINT,
            url_params={"projectId": project_id},
        )
        return ApiKeysHttpResponse(**data).api_keys

    def get_project_api_key(self, project_id: str, api_key_id: int) -> ApiKey:
        data, _ = self._api.send_request(
            method=HttpMethods.GET,
            endpoint=Endpoints.PROJECT_API_KEY_ENDPOINT,
            url_params={"projectId": project_id, "apiKeyId": api_key_id},
        )
        return ApiKeyHttpResponse(**data).api_key

    def create_project_api_key(self, project_id: str, name: str) -> ApiKey:
        body = CreateApiKeyRequest(name=name).model_dump(
            by_alias=True, exclude_none=True
        )
        data, _ = self._api.send_request(
            method=HttpMethods.POST,
            endpoint=Endpoints.PROJECT_API_KEYS_ENDPOINT,
            url_params={"projectId": project_id},
            body=body,
        )
        return ApiKeyHttpResponse(**data).api_key

    def update_project_api_key(
        self, project_id: str, api_key_id: int, valid: bool
    ) -> ApiKey:
        body = UpdateApiKeyRequest(valid=valid).model_dump(
            by_alias=True, exclude_none=True
        )
        data, _ = self._api.send_request(
            method=HttpMethods.PUT,
            endpoint=Endpoints.PROJECT_API_KEY_ENDPOINT,
            url_params={"projectId": project_id, "apiKeyId": api_key_id},
            body=body,
        )
        return ApiKeyHttpResponse(**data).api_key

    def delete_project_api_key(self, project_id: str, api_key_id: int) -> None:
        self._api.send_request(
            method=HttpMethods.DELETE,
            endpoint=Endpoints.PROJECT_API_KEY_ENDPOINT,
            url_params={"projectId": project_id, "apiKeyId": api_key_id},
        )
