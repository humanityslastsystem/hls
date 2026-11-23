from __future__ import annotations

import pytest

from hls.substrate.ucounts import (INOTIFY_FLOOR, INOTIFY_INSIDE, INOTIFY_OUTSIDE, NAMESPACE_FLOOR, NAMESPACE_OUTSIDE,
                                   SEAL_UCOUNTS_VAR, UCOUNTS, WATCHES_FLOOR, cap_problem, commands_limits, node_limits,
                                   ServedUcounts, node_shortfall, served_limits, ucount_limits, ucounts_env)

CAPS = {"sandbox_namespaces": 1024, "sandbox_inotify_instances": 8, "sandbox_inotify_watches": 65_536}
#: A site that sets no cap: the commands get their role's share less the harness's.
NO_CAPS = {"sandbox_namespaces": None, "sandbox_inotify_instances": None, "sandbox_inotify_watches": None}
#: A node's own limits.
NODE_LIMITS = {u.name: {"max_inotify_instances": 128, "max_inotify_watches": 1_048_576}.get(u.name, 4_124_579)
              for u in UCOUNTS}


def test_each_role_gets_its_evaluations_share_and_the_commands_the_sites_cap_within_it():
    made = {c.name: c for c in ucount_limits(NODE_LIMITS, CAPS, 8)}
    assert set(made) == {u.name for u in UCOUNTS}
    # 128 instances shared by 8 evaluations: 16 for each role, of which the commands get the cap,
    # 8, and the sandbox's own harness keeps the rest.
    assert (made["max_inotify_instances"].role, made["max_inotify_instances"].commands) == (16, 8)
    assert (made["max_inotify_watches"].role, made["max_inotify_watches"].commands) == (131_072, 65_536)
    assert {c.commands for name, c in made.items() if name.endswith("_namespaces")} == {1024}
    assert all(c.problem is None for c in made.values())


def test_without_a_cap_the_commands_get_their_roles_share_less_what_the_sandboxs_harness_keeps():
    made = {c.name: c for c in ucount_limits(NODE_LIMITS, NO_CAPS, 8)}
    assert (made["max_inotify_instances"].role, made["max_inotify_instances"].commands) == (16, 15)
    assert (made["max_inotify_watches"].role, made["max_inotify_watches"].commands) == (131_072, 130_048)
    assert made["max_user_namespaces"].commands == (4_124_579 - 8 * 16) // 8 - 16
    assert all(c.cap is None and c.problem is None for c in made.values())
    assert cap_problem("sandbox_inotify_watches", None) is None


def test_the_roles_and_their_harnesses_stay_within_the_nodes_limit_and_the_commands_within_their_role():
    for evaluations, caps in ((1, CAPS), (8, CAPS), (16, CAPS), (32, CAPS), (8, NO_CAPS), (32, NO_CAPS)):
        for c in ucount_limits(NODE_LIMITS, caps, evaluations):
            assert c.evaluations * (c.role + c.outside) <= c.node, c
            assert c.commands + c.inside <= c.role, c


def test_the_share_binds_where_it_leaves_the_commands_less_than_the_cap():
    inotify = {c.name: c for c in ucount_limits(NODE_LIMITS, CAPS, 32)}["max_inotify_instances"]
    assert inotify.role == (128 - 32 * INOTIFY_OUTSIDE) // 32 == 4
    assert inotify.commands == 4 - INOTIFY_INSIDE == 3 and inotify.problem is None


def test_commands_under_their_floor_name_why_no_sandbox_can_start():
    made = {c.name: c for c in ucount_limits(NODE_LIMITS, CAPS, 64)}
    inotify = made["max_inotify_instances"]
    assert (inotify.role, inotify.commands) == (2, 1)
    assert inotify.problem == ("the node's max_inotify_instances of 128, shared by 64 evaluations that each keep 0 "
                               "for their harness, leaves each sealed role 2 and a sandbox's commands 1 once its own "
                               "harness keeps 1, under the floor of 2; raise the node's limit, or run fewer "
                               "evaluations on it")
    assert made["max_user_namespaces"].problem is None


def test_the_commands_limits_go_to_the_spawner_and_the_roles_to_the_seal():
    made = ucount_limits({"max_user_namespaces": 1000, "max_inotify_instances": 128}, CAPS, 2)
    assert commands_limits(made) == {"max_user_namespaces": 468, "max_inotify_instances": 8}
    assert ucounts_env({c.name: c.role for c in made}) == {
        SEAL_UCOUNTS_VAR: "max_user_namespaces=484 max_inotify_instances=64"}


def test_a_count_the_nodes_kernel_has_no_limit_for_is_not_limited(tmp_path):
    (tmp_path / "max_user_namespaces").write_text("100\n")
    (tmp_path / "max_inotify_instances").write_text("128\n")
    (tmp_path / "max_fanotify_groups").write_text("128\n")  # a count the roles are not limited in
    assert node_limits(tmp_path) == {"max_user_namespaces": 100, "max_inotify_instances": 128}
    (tmp_path / "max_pid_namespaces").write_text("x\n")
    with pytest.raises(ValueError):
        node_limits(tmp_path)


def test_this_nodes_own_limits_are_read():
    node = node_limits()
    assert node["max_inotify_instances"] > 0 and node["max_inotify_watches"] > 0


def test_a_sites_cap_under_its_floor_is_refused_naming_the_setting():
    assert cap_problem("sandbox_namespaces", NAMESPACE_FLOOR) is None
    assert cap_problem("sandbox_namespaces", NAMESPACE_FLOOR - 1) == (
        "site.sandbox_namespaces of 15 is under the floor of 16; raise site.sandbox_namespaces to at least 16")
    assert cap_problem("sandbox_inotify_instances", INOTIFY_FLOOR - 1) is not None
    assert cap_problem("sandbox_inotify_watches", WATCHES_FLOOR) is None
    assert cap_problem("sandbox_inotify_watches", WATCHES_FLOOR - 1) is not None


#: The grading node limits a site declares (``site.grading_node_*``), here a node's own.
DECLARED = {"grading_node_namespaces": 4_124_579, "grading_node_inotify_instances": 128,
            "grading_node_inotify_watches": 1_048_576}


def test_a_graded_servers_share_comes_from_the_sites_declaration_for_every_count():
    made = {c.name: c for c in served_limits(DECLARED, 8)}
    # Every count, whatever any node's kernel has: the declaration, not a node, says what it is.
    assert set(made) == {u.name for u in UCOUNTS}
    assert made["max_time_namespaces"].role == made["max_user_namespaces"].role == (4_124_579 - 8 * 16) // 8
    assert (made["max_inotify_instances"].role, made["max_inotify_watches"].role) == (16, 131_072)
    assert (made["max_user_namespaces"].setting, made["max_user_namespaces"].declared) == (
        "grading_node_namespaces", 4_124_579)
    assert all(c.evaluations == 8 and c.problem is None for c in made.values())
    # A server of twice the size shares a node with half as many.
    assert {c.name: c.role for c in served_limits(DECLARED, 4)}["max_inotify_instances"] == 32


def test_a_graded_servers_share_under_the_floor_names_the_setting_to_raise():
    inotify = {c.name: c for c in served_limits(DECLARED, 128)}["max_inotify_instances"]
    assert inotify.role == 1 and inotify.problem == (
        "site.grading_node_inotify_instances of 128, shared by 128 graded servers that each keep 0 for their "
        "harness, leaves each 1 max_inotify_instances, under the floor of 2; raise "
        "site.grading_node_inotify_instances if the site's grading nodes hold more, or grade fewer on a node")


def test_a_node_whose_own_limits_cannot_hold_its_graded_servers_is_named():
    limits = served_limits(DECLARED, 8)
    node = {u.name: 4_124_579 for u in UCOUNTS if u.name.endswith("_namespaces")} | {
        "max_inotify_instances": 128, "max_inotify_watches": 1_048_576}
    assert node_shortfall(limits, node, 8) is None
    # A kernel without a count can make none of it: it needs no room there.
    assert node_shortfall(limits, {k: v for k, v in node.items() if k != "max_time_namespaces"}, 8) is None
    # Exactly what eight roles and their harnesses take is enough; one less is not.
    role = (4_124_579 - 8 * NAMESPACE_OUTSIDE) // 8
    assert node_shortfall(limits, node | {"max_pid_namespaces": 8 * (role + NAMESPACE_OUTSIDE)}, 8) is None
    assert node_shortfall(limits, node | {"max_pid_namespaces": 8 * (role + NAMESPACE_OUTSIDE) - 1}, 8) == (
        f"this node's max_pid_namespaces of {8 * (role + NAMESPACE_OUTSIDE) - 1} cannot hold 8 graded servers of "
        f"{role} each beside the {NAMESPACE_OUTSIDE} each harness keeps")
    assert node_shortfall(limits, node | {"max_inotify_instances": 127}, 8) is not None
    # The grading node's own count of servers decides, whatever the build's was: a node that
    # holds twice as many of the server's size needs room for twice as many shares.
    assert node_shortfall(limits, node, 16) is not None and node_shortfall(limits, node, 4) is None


def test_the_figures_a_build_states_are_recorded_whole():
    record = ServedUcounts(limits=served_limits(DECLARED, 8))
    assert ServedUcounts.model_validate_json(record.model_dump_json()) == record
    assert record.figures() == {c.name: c.role for c in served_limits(DECLARED, 8)}
