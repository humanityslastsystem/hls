"""The container interface stages start roles through.

A stage describes a container as a ``ContainerSpec``: the image, the command inside it, the
host paths it may see and where, and whether it gets the step's GPUs. It hands the spec to the
launcher as a ``ContainerRole`` (``RoleSpec.container``), which starts it the substrate's way,
as one step like any other role. The runtime decides every isolation flag; a stage decides only
what to mount, which device nodes the container may open, whether the container's network is
sealed, and, for the agent build's sandbox alone, the confinement its spawner needs
(``Confinement``). Variables, secrets among them,
never appear in the spec: they travel in an env file that the launcher hands to the runtime.

The image is a ``PinnedImage``: an image file pinned on the node, which the container runs from
read-only. The substrate's ``ImageRuntime`` says how a harness role pins one; the stage context
runs that role and keeps the result (``StageContext.node_image``, ``pin_file``).

Apptainer implements this interface (``hls.substrate.apptainer``). Another substrate
implements the same, and no stage changes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from typing import TYPE_CHECKING, Literal

from hls.egress.policy import SealPolicy
from hls.store.digest import check_sha256_hex
from hls.substrate.seal import SealFiles

if TYPE_CHECKING:
    from hls.recipes.model import ImageDecl, RegistryImage, UvPin


@dataclass(frozen=True)
class Mount:
    """One host path visible inside a container."""

    #: Absolute host path.
    source: Path
    #: Absolute path inside the container.
    target: str
    #: False lets the container write to *source*.
    read_only: bool = True


@dataclass(frozen=True)
class SealPlan:
    """A container whose network is sealed: every TCP connection it makes goes to an
    interceptor outside it, which lets through only what the policy allows. A seal with an index
    is a build's: its policy must carry a package-deny rule that refuses every name of the
    producer's own declared deny list (``deny_list``), so no build installs what its producer
    denies through it. Most producers declare the harness list (``hls.egress.denied_packages``);
    one keeps a list of its own."""

    #: What the interceptor enforces: the network policy and the hosts the seal vouches for, handed
    #: to it as JSON (``SealPolicy.model_dump_json()``), which it reads back through the model.
    policy: SealPolicy
    #: True also serves a pull-through package index on the container's loopback, filtered by
    #: the policy's package rule: the seal of a step that installs packages.
    index: bool
    #: The port a served container listens on inside the seal, the one port the seal forwards,
    #: over TCP from a free host port; None for a container nothing connects to, whose seal
    #: forwards none.
    ingress_port: int | None = None
    #: The package names the producer declares denied, which a seal with an index must refuse;
    #: empty for a seal without one.
    deny_list: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.policy, SealPolicy):
            raise TypeError(f"a seal's policy is a SealPolicy, not {type(self.policy).__name__}")
        if not self.index:
            if self.deny_list:
                raise ValueError("a seal without an index has no package rule to hold a deny list")
        elif not self.deny_list:
            raise ValueError("a seal with an index needs the producer's own deny list")
        else:
            rule = self.policy.network.packages
            if rule is None or rule.mode != "deny":
                raise ValueError("a seal with an index needs a package-deny rule in its policy")
            permitted = [name for name in self.deny_list if rule.permits(name, normalize=True)]
            if permitted:
                raise ValueError(f"a seal with an index must refuse the producer's deny list; its package rule "
                                 f"permits {permitted[:3]} and {max(len(permitted) - 3, 0)} more")
        if self.ingress_port is not None and not 0 < self.ingress_port < 65536:
            raise ValueError(f"ingress_port must be a TCP port, got {self.ingress_port}")


@dataclass(frozen=True)
class PinnedImage:
    """An image pinned on the node to the image file it came from, from which containers run
    read-only. Only the stage context that pinned it starts a container from it."""

    #: What the runtime runs a container from: Apptainer's extracted directory, the container's
    #: root; Docker's image ID in the node's daemon.
    ref: str
    #: The bare hex sha256 of the image file it was pinned from (``ImageRuntime.image_file``).
    file_sha256: str
    #: The bare hex sha256 a reuse of it is checked against: Apptainer's manifest written when it
    #: was extracted (``hls.substrate.sandbox``); Docker's manifest digest.
    check_sha256: str
    #: True for an image that holds a contestant's code, such as a submission image, on which
    #: only the contestant's own containers run; never for a harness-built image.
    contestant: bool = False

    def __post_init__(self) -> None:
        if not self.ref:
            raise ValueError("a pinned image needs a reference to run containers from")
        check_sha256_hex(self.file_sha256)
        check_sha256_hex(self.check_sha256)


#: What a container asks of its runtime's confinement. ``default``: the runtime's own, which every
#: container but one gets. ``spawner``: the agent build's sandbox (``hls.stages.agent_build``) alone,
#: whose spawner makes the commands' user, mount and PID namespaces and mounts a ``/proc`` of the
#: new PID namespace (``hls.agent.spawner``). Apptainer confines no container against that, so the
#: Slurm launcher runs a ``spawner`` container as any other; the Docker launcher widens its seccomp
#: profile, AppArmor and masked paths for it alone (``hls.substrate.docker.launcher``).
Confinement = Literal["default", "spawner"]


@dataclass(frozen=True)
class ContainerSpec:
    """One container to run as a role."""

    #: The image, extracted on this node.
    image: PinnedImage
    #: The command run inside the container.
    command: tuple[str, ...]
    #: Node-local host directory backing the container's /tmp, /var/tmp and $HOME.
    workdir: Path
    #: Host paths the container may see; nothing else from the host is mounted.
    mounts: tuple[Mount, ...] = ()
    #: True gives the container the step's GPUs.
    gpus: bool = False
    #: Directory inside the container the command starts in; None for the runtime's default.
    cwd: str | None = None
    #: The network seal; None for a container on the step's own network.
    seal: SealPlan | None = None
    #: True for a container that holds a contestant's payload and nothing of the harness: a served
    #: artifact, the seal probe that stands in for it, a package step. Sealed, it holds no
    #: capability at all. False for one the harness also runs in, such as the build's sandbox with
    #: its exec server: sealed, it keeps what that harness needs, less what the runtime drops from
    #: every sealed container. Unsealed, a container runs as the node's own user and holds none.
    contestant_payload: bool = False
    #: Device nodes under /dev the container may open, each the node's own at the same path and
    #: writable, since a device bound read-only cannot be opened for a mount: for a harness role
    #: that starts containers of its own, the one a rootless runtime mounts an image through. Only
    #: an unsealed container that holds no contestant payload may have one. A runtime that cannot
    #: give one exactly refuses the spec.
    devices: tuple[str, ...] = ()
    #: The confinement it asks of the runtime (``Confinement``): ``spawner`` for the agent build's
    #: sandbox, ``default`` for every other container.
    confinement: Confinement = "default"

    def __post_init__(self) -> None:
        if not isinstance(self.image, PinnedImage):
            raise TypeError(f"a container's image is a PinnedImage, not {type(self.image).__name__}")
        if self.devices and (self.seal is not None or self.contestant_payload):
            raise ValueError(f"only an unsealed harness container is given a device node; this one is "
                             f"{'sealed' if self.seal is not None else 'a contestant payload'} and names "
                             f"{', '.join(self.devices)}")
        if self.confinement not in ("default", "spawner"):
            raise ValueError(f"a container's confinement is default or spawner, not {self.confinement!r}")


class ContainerRuntime(ABC):
    """What a stage reads of the runtime its launcher starts containers with."""

    #: The line every container prints on stderr just before its command starts. A role that
    #: exits nonzero without printing it failed in the runtime, before its command ran.
    start_marker: str
    #: The directories the runtime makes in a container's workdir for its writable paths. A
    #: stage that makes them first starts the container without new space on the node's disk.
    workdir_dirs: tuple[str, ...]
    #: The line a sealed container's seal prints on stderr before each attempt to start it; None
    #: for a seal that makes one attempt.
    attempt_line: str | None
    #: GiB a sealed container's seal holds beside the role's own memory, outside the role's limit:
    #: 0 for a seal that runs inside the role's own step.
    seal_memory_gib: int

    @abstractmethod
    def check(self, spec: ContainerSpec) -> None:
        """Raise ValueError for a spec the runtime cannot express exactly."""


@dataclass(frozen=True)
class ContainerRole:
    """A container role as a stage hands it to the launcher (``RoleSpec.container``), which
    starts it the substrate's way."""

    #: The container.
    spec: ContainerSpec
    #: The node-local env file (mode 0600) of the container's variables, which may hold none.
    env_file: Path
    #: The host files of the container's seal; set exactly when ``spec.seal`` is.
    seal_files: SealFiles | None
    #: The limits on the uid's per-namespace kernel counts the seal writes before the payload
    #: starts, by their ``/proc/sys/user`` file names; empty for an unsealed container. A runtime
    #: that makes no user namespace around the payload records them as not applied.
    ucounts: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.spec, ContainerSpec):
            raise TypeError(f"a container role's spec is a ContainerSpec, not {type(self.spec).__name__}")
        if (self.seal_files is None) != (self.spec.seal is None):
            raise ValueError("a container role has seal files exactly when its container is sealed")
        if self.spec.seal is None and self.ucounts:
            raise ValueError("only a sealed container's seal writes kernel-count limits")


#: Exit status of a ``tree_role`` refusing a path its image does not hold, or holds as no
#: directory, having printed a ``refused:`` line.
TREE_ABSENT_EXIT = 66


class ImageRuntime(ABC):
    """How a substrate keeps images on a node: the file an images cache entry holds one in, the
    harness role that pins one for containers to run from, and the role that builds one. The
    stage context runs the roles and keeps what they pinned (``StageContext.node_image``,
    ``pin_file``, ``release``)."""

    #: The file an images cache entry, and a build's directory, hold the image in.
    image_file: str
    #: The program a submission image runs, as a path inside the image.
    runscript: str
    #: The file a built image's recipe is in, in its build's directory and its images cache entry.
    recipe_file: str
    #: The file in a build's directory that a layer's or a cap's parent image file is copied to.
    parent_file: str

    @staticmethod
    @abstractmethod
    def recipe_key(decl: ImageDecl, *, parent_key: str | None, locks_dir: Path, uv: UvPin, tasks_dir: Path,
                   registry: RegistryImage | None = None) -> str:
        """The images cache key of *decl* built by this runtime's recipe, given its parent's key
        (None for a base or a registry image), the lock files under *locks_dir*, the task
        directories under *tasks_dir*, which hold a task's own layer's locks and files, and
        *registry*, the registry image its chain starts at (``model.registry_of``), None for a
        chain that starts at a base. Raise ValueError or OSError, as the recipe's renderer does,
        for a declaration that cannot be keyed."""

    @abstractmethod
    def write_recipe(self, decl: ImageDecl, work: Path, *, locks_dir: Path, uv: UvPin, tasks_dir: Path,
                     registry: RegistryImage | None = None) -> str:
        """Write the recipe of *decl* into *work*, its build's directory, as ``recipe_file``, with
        whatever else its build reads there, from the lock files under *locks_dir* and the task
        directories under *tasks_dir*, with *registry* as ``recipe_key`` takes it; return the
        recipe's text, which the cache entry keeps. A layer's or a cap's recipe builds on the
        parent's copy at ``parent_file``."""

    @abstractmethod
    def build_record(self, work: Path, stderr: str) -> dict[str, object]:
        """What a build that succeeded in *work*, with *stderr*, records in its cache entry's
        ``meta.json``, by field. Raise ValueError when the build left no record it must have
        left."""

    @abstractmethod
    def left(self, sha256: str, *, contestant: bool) -> PinnedImage | str | None:
        """The image of the file whose bare hex sha256 is *sha256* that an earlier stage of the
        job pinned on this node and could not remove: the image, checked, when it can be reused
        (marked *contestant*); a line saying why it cannot, which ``remove_left`` then removes;
        None when nothing was left. Raise ValueError for a *sha256* that is not bare hex."""

    @abstractmethod
    def remove_left(self, sha256: str) -> None:
        """Remove what an earlier stage left of the image of *sha256* (``left``). Raise OSError
        when it cannot be removed."""

    @abstractmethod
    def pin_role(self, image_file: Path, *, sha256: str, own: Path) -> tuple[list[str], dict[str, str]]:
        """The command line and declared environment of the harness role that pins the image
        file *image_file* on this node, run in its own directory *own*. It reads the file once,
        into *own*, checks that the copy's sha256 is *sha256* and pins only the copy. A copy with
        another digest is refused: the role prints a line starting ``refused:`` on stderr and
        exits ``DIGEST_MISMATCH_EXIT`` before it pins anything. Make the directories the role
        needs. Raise ValueError for a path the command line cannot express exactly, or a
        *sha256* that is not bare hex."""

    @abstractmethod
    def pinned(self, own: Path, *, sha256: str, contestant: bool) -> PinnedImage:
        """The image the pin role that ran in *own* pinned, for the file of *sha256*, once that
        role succeeded, put in place and marked *contestant*. Raise ValueError, naming what the
        role left, when it pinned nothing, once that is removed."""

    @abstractmethod
    def discard(self, own: Path) -> None:
        """Remove what the pin role that ran in *own* left, once that role failed."""

    @abstractmethod
    def remove(self, image: PinnedImage) -> None:
        """Remove *image*, which this runtime pinned, from the node. Raise OSError when it cannot
        be removed."""

    @abstractmethod
    def tree_path(self, image: PinnedImage, path: str) -> Path | None:
        """Where *image* holds the absolute *path* as a tree on the host, read in place; None when
        the runtime holds no image as a host tree, and a role of ``tree_role`` puts a copy on the
        host instead. The caller refuses a path that is no directory there. Raise ValueError for a
        path that is not absolute."""

    @abstractmethod
    def tree_role(self, image: PinnedImage, path: str, dest: Path, *, own: Path) -> tuple[list[str], dict[str, str]]:
        """For a runtime whose ``tree_path`` is None: the command line and declared environment of
        the harness role, run in its own directory *own* and started with ``daemon=True``, that puts
        what *image* holds at the absolute *path* into the empty host directory *dest*, as files of
        the host user. *path* is one directory, never the image's root: an image is not copied whole
        (``reader_args``). A path the image does not hold, or holds as no directory, is refused: the
        role exits ``TREE_ABSENT_EXIT`` with a ``refused:`` line. An empty directory gives an empty
        *dest*. Raise ValueError for a path that is not absolute or is the root, and on a runtime
        whose ``tree_path`` is a path."""

    @abstractmethod
    def reader_args(self, flag: str, image: PinnedImage, file: Path, *, listing: Path | None = None,
                    listing_out: Path | None = None) -> list[str]:
        """The arguments that hand *image*, pinned from the image file *file*, to a harness program
        that reads it whole (the package's scan and integrity check) as its ``--<flag>`` image: on a
        runtime that extracts its images, ``--<flag> TREE``, the tree it is pinned as, read in
        place; on one that does not, ``--<flag>-archive FILE --<flag>-manifest HEX``, the archive
        and the bare hex manifest digest it was pinned by, which the program reads layer by layer
        (``hls.package.layers``) with no daemon and no container. On such a runtime, *listing_out*
        is where the program leaves the image's tree once it has read it, and *listing* where a
        later program reads that tree instead of the layers, refusing one of another manifest
        (``--<flag>-listing-out``, ``--<flag>-listing``); a runtime whose images are trees on the
        host ignores both. Raise ValueError for a path the command line cannot express exactly."""

    @abstractmethod
    def submission_role(self, out: Path, *, digest: Path, base_file: tuple[Path, str], base: PinnedImage, tree: Path,
                        wheels: Path, project: str, server: str, nonce: str,
                        constraints: str | None) -> tuple[list[str], dict[str, str]]:
        """The command line and declared environment of the harness role that builds a submission
        image into the image file *out*, in the role's own directory ``out.parent``, and then writes
        the file's sha256 to *digest*: from the submission base, whose image file *base_file* is (its
        path in the store and the sha256 its entry records; a role that reads it reads it once and
        refuses a copy of another digest) and which the stage pinned as *base*, the checkout's *tree*
        and the built *wheels*, installing *project* offline with the
        install script of ``hls.package.offline`` (*server*, *nonce* and *constraints* as there), so
        its output carries the script's status lines. The install runs the contestant's code: the
        runtime runs it confined as a role of the stage is, never on a privileged builder. Make the
        directories the role needs. Raise ValueError for a path or digest the role cannot take."""

    @abstractmethod
    def build_role(self, recipe: Path, out: Path, *, digest: Path, parent: tuple[Path, str, Path] | None,
                   login: bool, contestant: bool) -> tuple[list[str], dict[str, str]]:
        """The command line and declared environment of the harness role that builds the recipe
        *recipe* into the image file *out*, in the role's own directory, ``out.parent``, and then
        writes the file's sha256 to *digest*. *parent* is, for a layer or a cap, the parent's
        image file in the store, the sha256 its entry records and the path in the role's own
        directory it is copied to, which the recipe names: the copy is refused unless it has
        that digest. With *login*, the role logs in to the registry from ``REGISTRY_USER_VAR``
        in its declared environment and ``REGISTRY_TOKEN_GRANT`` in its env file. *contestant*
        says whether the recipe's input holds a contestant's files, as a submission image's does;
        a runtime whose builder may not build such input raises ValueError for it. Make the
        directories the role needs. Raise ValueError for a path that is not absolute or a
        digest that is not bare hex."""
