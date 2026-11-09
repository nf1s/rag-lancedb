import pytest

from common.permissions import Action, Role, role_allows


@pytest.mark.parametrize(
    ("role", "action", "expected"),
    [
        ("viewer", Action.READ, True),
        ("viewer", Action.WRITE, False),
        ("viewer", Action.REINDEX, False),
        ("viewer", Action.MANAGE_MEMBERS, False),
        ("viewer", Action.DELETE_WORKSPACE, False),
        ("editor", Action.READ, True),
        ("editor", Action.WRITE, True),
        ("editor", Action.REINDEX, False),
        ("editor", Action.MANAGE_MEMBERS, False),
        ("editor", Action.DELETE_WORKSPACE, False),
        ("admin", Action.READ, True),
        ("admin", Action.WRITE, True),
        ("admin", Action.REINDEX, True),
        ("admin", Action.MANAGE_MEMBERS, True),
        ("admin", Action.DELETE_WORKSPACE, False),
        ("owner", Action.READ, True),
        ("owner", Action.WRITE, True),
        ("owner", Action.REINDEX, True),
        ("owner", Action.MANAGE_MEMBERS, True),
        ("owner", Action.DELETE_WORKSPACE, True),
        (None, Action.READ, False),
        ("unknown", Action.READ, False),
    ],
)
def test_role_action_matrix(role, action, expected):
    assert role_allows(role, action) is expected


def test_unknown_action_raises():
    with pytest.raises(ValueError):
        role_allows(Role.OWNER, "fly")
