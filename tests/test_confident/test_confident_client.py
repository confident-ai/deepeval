from typing import Any, Callable, Dict, List, Optional

import pytest
from pydantic import SecretStr

from deepeval.confident import api as confident_api
from deepeval.confident import (
    ApiKey,
    ConfidentClient,
    Invitation,
    Member,
    Organization,
    Permission,
    Policy,
    Project,
    Role,
)


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


def _install_fake_http(
    monkeypatch: pytest.MonkeyPatch,
    handler: Callable[[str, str, Optional[Dict], Optional[Dict]], Dict],
) -> List[Dict[str, Any]]:
    """Patch settings + Api._http_request and record every outgoing call."""

    class DummySettings:
        CONFIDENT_BASE_URL = None
        CONFIDENT_API_KEY = SecretStr("confident_us_test")
        API_KEY = None
        DEEPEVAL_DEFAULT_SAVE = None

    monkeypatch.setattr(confident_api, "get_settings", lambda: DummySettings())
    monkeypatch.setattr(
        confident_api.KEY_FILE_HANDLER,
        "fetch_data",
        lambda *args, **kwargs: None,
    )

    calls: List[Dict[str, Any]] = []

    def fake_http_request(
        method: str, url: str, headers=None, json=None, params=None
    ):
        calls.append(
            {
                "method": method,
                "url": url,
                "json": json,
                "params": params,
            }
        )
        payload = handler(method, url, json, params)
        return _FakeResponse(200, payload)

    monkeypatch.setattr(
        confident_api.Api, "_http_request", staticmethod(fake_http_request)
    )
    return calls


def _ok(data: Any) -> Dict[str, Any]:
    return {"success": True, "data": data, "deprecated": False}


def test_client_requires_api_key_when_settings_empty(monkeypatch):
    class DummySettings:
        CONFIDENT_BASE_URL = None
        CONFIDENT_API_KEY = None
        API_KEY = None
        DEEPEVAL_DEFAULT_SAVE = None

    monkeypatch.setattr(confident_api, "get_settings", lambda: DummySettings())

    with pytest.raises(ValueError):
        ConfidentClient()


def test_client_accepts_explicit_api_key(monkeypatch):
    class DummySettings:
        CONFIDENT_BASE_URL = None
        CONFIDENT_API_KEY = None
        API_KEY = None
        DEEPEVAL_DEFAULT_SAVE = None

    monkeypatch.setattr(confident_api, "get_settings", lambda: DummySettings())
    monkeypatch.setattr(
        confident_api.KEY_FILE_HANDLER,
        "fetch_data",
        lambda *args, **kwargs: None,
    )

    client = ConfidentClient(api_key="confident_us_explicit")
    assert client._api.api_key == "confident_us_explicit"


def test_get_organization_parses_response(monkeypatch):
    def handler(method, url, json, params):
        assert method == "GET"
        assert url.endswith("/v1/organization")
        return _ok({"organization": {"id": "org_1", "name": "Acme"}})

    _install_fake_http(monkeypatch, handler)

    org = ConfidentClient().get_organization()
    assert isinstance(org, Organization)
    assert org.id == "org_1"
    assert org.name == "Acme"


def test_update_organization_sends_body(monkeypatch):
    def handler(method, url, json, params):
        assert method == "PUT"
        assert json == {"name": "Acme 2"}
        return _ok({"organization": {"id": "org_1", "name": "Acme 2"}})

    _install_fake_http(monkeypatch, handler)

    org = ConfidentClient().update_organization(name="Acme 2")
    assert org.name == "Acme 2"


def test_list_projects_returns_list_of_models(monkeypatch):
    def handler(method, url, json, params):
        assert method == "GET"
        assert url.endswith("/v1/projects")
        return _ok(
            {
                "projects": [
                    {"id": "p1", "name": "Alpha", "organizationId": "org_1"},
                    {"id": "p2", "name": "Beta"},
                ]
            }
        )

    _install_fake_http(monkeypatch, handler)

    projects = ConfidentClient().list_projects()
    assert [p.id for p in projects] == ["p1", "p2"]
    assert all(isinstance(p, Project) for p in projects)
    assert projects[0].organization_id == "org_1"


def test_create_project_strips_none_fields(monkeypatch):
    def handler(method, url, json, params):
        assert method == "POST"
        assert json == {"name": "Alpha"}
        return _ok({"project": {"id": "p1", "name": "Alpha"}})

    _install_fake_http(monkeypatch, handler)

    project = ConfidentClient().create_project(name="Alpha")
    assert project.id == "p1"


def test_create_project_includes_description_when_provided(monkeypatch):
    def handler(method, url, json, params):
        assert json == {"name": "Alpha", "description": "first"}
        return _ok(
            {"project": {"id": "p1", "name": "Alpha", "description": "first"}}
        )

    _install_fake_http(monkeypatch, handler)

    project = ConfidentClient().create_project(
        name="Alpha", description="first"
    )
    assert project.description == "first"


def test_get_project_substitutes_url_params(monkeypatch):
    def handler(method, url, json, params):
        assert method == "GET"
        assert url.endswith("/v1/projects/p_42")
        return _ok({"project": {"id": "p_42", "name": "Forty Two"}})

    _install_fake_http(monkeypatch, handler)

    project = ConfidentClient().get_project(project_id="p_42")
    assert project.id == "p_42"


def test_delete_project_does_not_return(monkeypatch):
    def handler(method, url, json, params):
        assert method == "DELETE"
        assert url.endswith("/v1/projects/p_42")
        return _ok(None)

    _install_fake_http(monkeypatch, handler)

    assert ConfidentClient().delete_project(project_id="p_42") is None


def test_list_organization_members_sends_pagination(monkeypatch):
    def handler(method, url, json, params):
        assert method == "GET"
        assert params == {"page": 2, "pageSize": 50}
        return _ok({"members": [{"id": "u1", "email": "u1@example.com"}]})

    _install_fake_http(monkeypatch, handler)

    members = ConfidentClient().list_organization_members(page=2, page_size=50)
    assert isinstance(members[0], Member)
    assert members[0].id == "u1"


def test_member_response_uses_aliased_fields(monkeypatch):
    def handler(method, url, json, params):
        return _ok(
            {
                "members": [
                    {
                        "id": "u1",
                        "email": "u1@example.com",
                        "organizationRole": {"id": "r1", "name": "admin"},
                    }
                ]
            }
        )

    _install_fake_http(monkeypatch, handler)

    members = ConfidentClient().list_organization_members()
    assert members[0].organization_role is not None
    assert members[0].organization_role.id == "r1"


def test_create_organization_invitations_sends_role_alias(monkeypatch):
    def handler(method, url, json, params):
        assert method == "POST"
        assert json == {
            "emails": ["a@example.com", "b@example.com"],
            "organizationRoleId": "role_1",
        }
        return _ok(
            {
                "invitations": [
                    {"id": 1, "email": "a@example.com"},
                    {"id": 2, "email": "b@example.com"},
                ]
            }
        )

    _install_fake_http(monkeypatch, handler)

    invitations = ConfidentClient().create_organization_invitations(
        emails=["a@example.com", "b@example.com"], role_id="role_1"
    )
    assert [i.id for i in invitations] == [1, 2]
    assert all(isinstance(i, Invitation) for i in invitations)


def test_create_organization_invitations_omits_role_when_none(monkeypatch):
    def handler(method, url, json, params):
        assert json == {"emails": ["a@example.com"]}
        return _ok({"invitations": [{"id": 1, "email": "a@example.com"}]})

    _install_fake_http(monkeypatch, handler)

    ConfidentClient().create_organization_invitations(emails=["a@example.com"])


def test_create_organization_role_sends_policy_ids_alias(monkeypatch):
    def handler(method, url, json, params):
        assert method == "POST"
        assert json == {"name": "Editor", "policyIds": ["pol_1", "pol_2"]}
        return _ok({"role": {"id": "r1", "name": "Editor"}})

    _install_fake_http(monkeypatch, handler)

    role = ConfidentClient().create_organization_role(
        name="Editor", policy_ids=["pol_1", "pol_2"]
    )
    assert isinstance(role, Role)


def test_create_organization_policy_sends_permission_ids_alias(monkeypatch):
    def handler(method, url, json, params):
        assert json == {
            "name": "Read",
            "permissionIds": ["perm_1"],
        }
        return _ok({"policy": {"id": "pol_1", "name": "Read"}})

    _install_fake_http(monkeypatch, handler)

    policy = ConfidentClient().create_organization_policy(
        name="Read", permission_ids=["perm_1"]
    )
    assert isinstance(policy, Policy)


def test_list_project_permissions_substitutes_project_id(monkeypatch):
    def handler(method, url, json, params):
        assert url.endswith("/v1/projects/p_42/permissions")
        return _ok({"permissions": [{"id": "perm_1", "name": "read"}]})

    _install_fake_http(monkeypatch, handler)

    permissions = ConfidentClient().list_project_permissions(project_id="p_42")
    assert isinstance(permissions[0], Permission)


def test_update_project_api_key_combines_url_params_and_body(monkeypatch):
    def handler(method, url, json, params):
        assert method == "PUT"
        assert url.endswith("/v1/projects/p_42/api-keys/7")
        assert json == {"valid": False}
        return _ok(
            {"apiKey": {"id": 7, "name": "ci", "value": "***", "valid": False}}
        )

    _install_fake_http(monkeypatch, handler)

    api_key = ConfidentClient().update_project_api_key(
        project_id="p_42", api_key_id=7, valid=False
    )
    assert isinstance(api_key, ApiKey)
    assert api_key.valid is False


def test_resend_project_invitation_substitutes_both_url_params(monkeypatch):
    def handler(method, url, json, params):
        assert method == "PUT"
        assert url.endswith("/v1/projects/p_42/invitations/9")
        return _ok({"invitation": {"id": 9, "email": "a@example.com"}})

    _install_fake_http(monkeypatch, handler)

    invitation = ConfidentClient().resend_project_invitation(
        project_id="p_42", invitation_id=9
    )
    assert invitation.id == 9
