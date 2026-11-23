"""``[sandbox_stores]``: the directory of each store a task's sandboxes start from, by name,
checked when the file loads, and left out of the resolved settings when the site names none. Every
sandbox a drive starts can reach the stores, so each stands apart from the store root, the secrets
file's directory, the checkout and every other store: as written when the file loads, and with
links followed where the paths are read (``sandbox_store_problems``)."""

from __future__ import annotations

import os
import re
import textwrap
from pathlib import Path

import pytest

from hls.pipeline.commands import split_overrides
from hls.settings import SettingsError, load_settings, resolved, sandbox_store_problems, settings_sha256

SITE = """
[site]
partition = "gpu"
store_root = "/shared/hls/store"
node_scratch = "/tmp"
secrets_path = "/shared/secrets/hls.toml"
"""
#: The reasons a store is refused, by what it meets.
STORE_ROOT = r"is, holds or lies in the store root \(site\.store_root\)"
SECRETS_DIR = r"is, holds or lies in the secrets file's directory \(site\.secrets_path\)"
CHECKOUT = "is, holds or lies in the checkout"


def _load(tmp_path: Path, text: str):
    path = tmp_path / "hls.toml"
    path.write_text(textwrap.dedent(text))
    return load_settings(path)


def test_a_site_names_each_store_by_its_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", "/home/operator")
    settings = _load(tmp_path, SITE + """
[sandbox_stores]
agentic-coding-images = "/shared/agentic-coding/images"
agentic-coding-overlays = "~/agentic-coding/overlays"
""")
    assert settings.sandbox_stores == {"agentic-coding-images": Path("/shared/agentic-coding/images"),
                                       "agentic-coding-overlays": Path("/home/operator/agentic-coding/overlays")}
    assert resolved(settings)["sandbox_stores"] == {"agentic-coding-images": "/shared/agentic-coding/images",
                                                    "agentic-coding-overlays": "/home/operator/agentic-coding/overlays"}


def test_a_site_that_names_none_resolves_and_digests_as_a_file_without_the_table(tmp_path):
    bare = _load(tmp_path, SITE)
    empty = _load(tmp_path, SITE + "\n[sandbox_stores]\n")
    assert bare.sandbox_stores == empty.sandbox_stores == {}
    assert "sandbox_stores" not in resolved(bare) and resolved(bare) == resolved(empty)
    assert settings_sha256(bare) == settings_sha256(empty)
    named = _load(tmp_path, SITE + '\n[sandbox_stores]\nimages = "/shared/images"\n')
    assert settings_sha256(named) != settings_sha256(bare)


@pytest.mark.parametrize("key, why", [
    ("Images", "lowercase letters, digits and hyphens"),
    ("1images", "lowercase letters, digits and hyphens"),
    ("agentic_coding", "lowercase letters, digits and hyphens"),
    ('"a b"', "lowercase letters, digits and hyphens"),
    ("device", "names a device node"),
])
def test_a_store_name_that_could_not_be_a_tool_runners_is_refused(tmp_path, key, why):
    with pytest.raises(SettingsError, match=why):
        _load(tmp_path, SITE + f'\n[sandbox_stores]\n{key} = "/shared/x"\n')


@pytest.mark.parametrize("value, why", [
    ("shared/images", "absolute path in normal form"),
    ("/shared/../images", "absolute path in normal form"),
    ("//shared/images", "absolute path in normal form"),
    ("/", "cannot be the root directory"),
])
def test_a_store_that_is_no_absolute_directory_path_is_refused_naming_it(tmp_path, value, why):
    with pytest.raises(SettingsError, match=rf"sandbox_stores\.images: .*{why}"):
        _load(tmp_path, SITE + f'\n[sandbox_stores]\nimages = "{value}"\n')


@pytest.mark.parametrize("value", ["/shared/hls/store/sandboxes", "/shared/hls/store", "/shared/hls", "/shared"])
def test_a_store_in_the_store_root_or_holding_it_is_refused(tmp_path, value):
    with pytest.raises(SettingsError, match=rf"sandbox_stores\.images {STORE_ROOT}"):
        _load(tmp_path, SITE + f'\n[sandbox_stores]\nimages = "{value}"\n')


@pytest.mark.parametrize("value", ["/shared/secrets", "/shared/secrets/runtime", "/shared", "~", "~/.config"])
def test_a_store_that_is_in_or_holds_the_secrets_files_directory_is_refused_naming_no_path(tmp_path, monkeypatch,
                                                                                         value):
    # The runtime store is bound whole into every sandbox, and an online row keeps a network: a
    # store that held the secrets file would hand it to a sandbox's commands.
    monkeypatch.setenv("HOME", "/home/operator")
    site = SITE.replace("/shared/secrets/hls.toml", "/shared/secrets/hls.toml" if value.startswith("/")
                        else "~/.config/hls/secrets.toml")
    with pytest.raises(SettingsError, match=rf"sandbox_stores\.agentic-coding-runtime {SECRETS_DIR}") as e:
        _load(tmp_path, site + f'\n[sandbox_stores]\nagentic-coding-runtime = "{value}"\n')
    assert "/home/operator" not in str(e.value) and "/shared/secrets" not in str(e.value)


@pytest.mark.parametrize("other", ["/shared/sandboxes/images", "/shared/sandboxes/images/runtime", "/shared/sandboxes"])
def test_two_stores_that_overlap_are_refused_naming_both(tmp_path, other):
    with pytest.raises(SettingsError, match=r"sandbox_stores\.images and sandbox_stores\.runtime overlap"):
        _load(tmp_path, SITE + f'\n[sandbox_stores]\nimages = "/shared/sandboxes/images"\nruntime = "{other}"\n')


def test_a_store_beside_the_store_root_the_secrets_and_the_other_stores_is_accepted(tmp_path):
    settings = _load(tmp_path, SITE + '\n[sandbox_stores]\nimages = "/shared/hls/store-sandboxes"\n'
                                      'runtime = "/shared/secrets-runtime"\nprefixes = "/shared/hls/store-sandboxes2"\n')
    assert settings.sandbox_stores["images"] == Path("/shared/hls/store-sandboxes")


def test_the_stores_are_set_in_the_file_only(tmp_path):
    with pytest.raises(ValueError, match="sandbox_stores.images"):
        split_overrides({"sandbox_stores.images": "/shared/images"})


@pytest.fixture
def site(tmp_path) -> dict[str, Path]:
    """A store root, a secrets file's directory, a checkout and a store, each a directory, and
    beside them a directory of aliases."""
    dirs = {name: tmp_path / name for name in ("store", "secrets", "checkout", "images", "aliases")}
    for d in dirs.values():
        d.mkdir()
    return dirs


@pytest.mark.parametrize("target, why", [("store", STORE_ROOT), ("secrets", SECRETS_DIR), ("checkout", CHECKOUT),
                                         ("images", r"sandbox_stores\.images and sandbox_stores\.runtime overlap")])
@pytest.mark.parametrize("where", ["at", "in"])
def test_a_store_reached_through_a_link_is_judged_by_where_it_leads(site, target, why, where):
    alias = site["aliases"] / target
    os.symlink(site[target], alias)
    store = alias if where == "at" else alias / "runtime"
    stores = {"images": site["images"], "runtime": store}
    args = {"store_root": site["store"], "secrets_path": site["secrets"] / "secrets.toml", "checkout": site["checkout"]}
    # As written, the alias stands apart from everything: the file's own check passes it.
    assert sandbox_store_problems(stores, **args) == []
    [problem] = sandbox_store_problems(stores, **args, resolve=True)
    assert re.search(why, problem) and "runtime" in problem


def test_with_links_followed_a_store_apart_from_everything_passes(site):
    os.symlink(site["images"], site["aliases"] / "images")
    stores = {"images": site["aliases"] / "images", "runtime": site["aliases"] / "runtime"}
    assert sandbox_store_problems(stores, store_root=site["store"], secrets_path=site["secrets"] / "secrets.toml",
                                  checkout=site["checkout"], resolve=True) == []
