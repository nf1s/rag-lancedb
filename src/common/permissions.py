"""Role-based access control shared by the store and the API."""

from __future__ import annotations

from enum import StrEnum


class Role(StrEnum):
    VIEWER = "viewer"
    EDITOR = "editor"
    ADMIN = "admin"
    OWNER = "owner"


class Action(StrEnum):
    READ = "read"
    WRITE = "write"
    REINDEX = "reindex"
    MANAGE_MEMBERS = "manage_members"
    DELETE_WORKSPACE = "delete_workspace"


ROLE_RANK: dict[str, int] = {
    Role.VIEWER: 1,
    Role.EDITOR: 2,
    Role.ADMIN: 3,
    Role.OWNER: 4,
}

REQUIRED_ROLE: dict[str, Role] = {
    Action.READ: Role.VIEWER,
    Action.WRITE: Role.EDITOR,
    Action.REINDEX: Role.ADMIN,
    Action.MANAGE_MEMBERS: Role.ADMIN,
    Action.DELETE_WORKSPACE: Role.OWNER,
}

ASSIGNABLE_ROLES: tuple[str, ...] = tuple(role.value for role in Role)


def role_allows(role: str | None, action: str | Action) -> bool:
    """True when ``role`` grants ``action``. Unknown roles never grant access."""
    required = REQUIRED_ROLE.get(str(action))
    if required is None:
        raise ValueError(f"unknown action: {action}")
    if role is None:
        return False
    return ROLE_RANK.get(role, 0) >= ROLE_RANK[required]
