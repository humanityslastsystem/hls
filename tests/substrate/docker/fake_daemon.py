"""A fake Docker daemon: an HTTP/1.1 server on a Unix socket, in a thread of the test process,
that speaks the few Engine API endpoints ``hls.substrate.docker.api`` calls, with the status codes
and body forms the Engine API documents. A test can make any endpoint fail (``fail``), hang
(``hang``) or answer something that is not JSON (``garble``)."""

from __future__ import annotations

import io
import json
import re
import socketserver
import tarfile
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

VERSION = {
    "Version": "29.5.2", "ApiVersion": "1.54", "MinAPIVersion": "1.40", "Os": "linux", "Arch": "amd64",
    "KernelVersion": "7.0.0", "Components": [{"Name": "Engine", "Version": "29.5.2"},
                                             {"Name": "containerd", "Version": "2.2.0"}],
}
INFO = {
    "ID": "db28a0d1-0000-4000-8000-000000000000", "Name": "fakehost", "ServerVersion": "29.5.2",
    "OperatingSystem": "Fake Linux", "KernelVersion": "7.0.0", "CgroupDriver": "systemd",
    "CgroupVersion": "2", "LiveRestoreEnabled": False,
    "SecurityOptions": ["name=apparmor", "name=seccomp,profile=builtin", "name=cgroupns"],
    "NCPU": 8, "MemTotal": 26764296192, "DockerRootDir": "/var/lib/docker", "Driver": "overlayfs",
    "DriverStatus": [["driver-type", "io.containerd.snapshotter.v1"]],
    "Runtimes": {"runc": {"path": "runc"}, "nvidia": {"path": "nvidia-container-runtime"}},
    "DefaultRuntime": "runc", "Warnings": None,
}
ZERO_TIME = "0001-01-01T00:00:00Z"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + ".123456789Z"


class FakeDocker:
    """The daemon's state and its endpoints. ``requests`` records each request as
    ``(method, path, query, body)``; ``create_bodies`` each container create body."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.version: dict[str, Any] = json.loads(json.dumps(VERSION))
        self.info: dict[str, Any] = json.loads(json.dumps(INFO))
        self.containers: dict[str, dict[str, Any]] = {}
        self.images: dict[str, dict[str, Any]] = {}
        #: Volume names, each to the IDs of the containers that use it.
        self.volumes: dict[str, set[str]] = {}
        self.events: list[dict[str, Any]] = []
        self.requests: list[tuple[str, str, dict[str, list[str]], bytes]] = []
        self.create_bodies: list[dict[str, Any]] = []
        self.create_warnings: list[str] = []
        self.load_error: str | None = None
        #: How a load names the image of an unnamed archive: None gives it an ID of the fake's own;
        #: "containerd" the digest of the manifest its index.json names, as containerd's image store
        #: does; "classic" the digest of the config its manifest.json names, as the classic store
        #: does, which refuses an archive without manifest.json.
        self.image_store: str | None = None
        #: Called with a container's ID and record when it starts, as its process would run.
        self.on_start = None
        #: Each commit as ``(container, query, config body)``.
        self.commits: list[tuple[str, dict[str, list[str]], dict[str, Any]]] = []
        #: What ``GET /images/{id}/get`` answers for an image a commit made, by image ID.
        self.saved: dict[str, bytes] = {}
        #: Per image ID, what ``GET /containers/{id}/archive`` finds in a container of it: absolute
        #: path to a directory's {relative file name: bytes}, or to one file's bytes.
        self.trees: dict[str, dict[str, dict[str, bytes] | bytes]] = {}
        #: Whether that archive names its members from the root, with a leading slash, as one image
        #: store does.
        self.archive_from_root = False
        self.loaded_bodies: list[bytes] = []
        self._faults: list[tuple[str, re.Pattern[str], str, Any]] = []
        self._changed = threading.Condition()
        self._stopping = False
        self._server: socketserver.ThreadingUnixStreamServer | None = None

    # -- test controls ---------------------------------------------------------------------

    def fail(self, method: str, path: str, status: int, message: str,
             when: Callable[[bytes], bool] | None = None) -> None:
        """Answer requests whose path matches the regex *path*, and whose body *when* holds true of
        when given, with *status* and *message*."""
        self._faults.append((method, re.compile(path), "fail", (status, message, when)))

    def hang(self, method: str, path: str, seconds: float) -> None:
        self._faults.append((method, re.compile(path), "hang", seconds))

    def garble(self, method: str, path: str, body: bytes = b"<html>not json</html>") -> None:
        self._faults.append((method, re.compile(path), "garble", body))

    def add_image(self, ref: str, labels: dict[str, str] | None = None) -> str:
        image_id = "sha256:" + uuid.uuid4().hex * 2
        self.images[image_id] = {"Id": image_id, "RepoTags": [ref], "RepoDigests": [],
                                 "Config": {"Labels": labels or {}}, "Architecture": "amd64",
                                 "Os": "linux", "Size": 1234}
        return image_id

    def exit(self, container_id: str, code: int, *, oom: bool = False) -> None:
        """End a running container by itself, as its process exiting would."""
        with self._changed:
            c = self.containers[container_id]
            c["State"].update(Status="exited", Running=False, Pid=0, ExitCode=code, OOMKilled=oom,
                              FinishedAt=_now())
            self._event("container", "die", container_id, {"exitCode": str(code)})
            self._changed.notify_all()

    # -- serving ---------------------------------------------------------------------------

    @contextmanager
    def serving(self) -> Iterator[FakeDocker]:
        fake = self

        class Handler(_Handler):
            daemon = fake

        server = socketserver.ThreadingUnixStreamServer(str(self.path), Handler)
        server.daemon_threads = True
        self._server = server
        thread = threading.Thread(target=server.serve_forever, args=(0.05,), daemon=True)
        thread.start()
        try:
            yield self
        finally:
            with self._changed:
                self._stopping = True
                self._changed.notify_all()
            server.shutdown()
            server.server_close()
            thread.join()

    def _event(self, kind: str, action: str, actor: str, attributes: dict[str, str]) -> None:
        labels = (self.containers.get(actor) or {}).get("Config", {}).get("Labels", {})
        ns = time.time_ns()
        self.events.append({"Type": kind, "Action": action, "Actor": {"ID": actor,
                            "Attributes": {**labels, **attributes}}, "scope": "local",
                            "time": ns // 1_000_000_000, "timeNano": ns})
        self._changed.notify_all()

    def _find(self, name: str) -> str | None:
        for cid, c in self.containers.items():
            if cid == name or cid.startswith(name) or c["Name"] == "/" + name:
                return cid
        return None

    def _find_image(self, ref: str) -> str | None:
        for iid, image in self.images.items():
            if iid == ref or ref in image["RepoTags"] or iid == "sha256:" + ref:
                return iid
        return None


def _matches(event: dict[str, Any], filters: dict[str, list[str]], since: float, until: float | None) -> bool:
    t = event["timeNano"] / 1e9
    if t < since or (until is not None and t > until):
        return False
    if filters.get("container") and event["Actor"]["ID"] not in filters["container"]:
        return False
    if filters.get("type") and event["Type"] not in filters["type"]:
        return False
    if filters.get("event") and event["Action"] not in filters["event"]:
        return False
    return True


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    daemon: FakeDocker

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - the base class's name
        pass

    def address_string(self) -> str:
        return "unix"

    # -- framing ---------------------------------------------------------------------------

    def _reply(self, status: int, body: Any = None, *, raw: bytes | None = None) -> None:
        data = raw if raw is not None else (b"" if body is None else json.dumps(body).encode())
        self.send_response(status)
        if data or status not in (204, 304):
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if data:
            self.wfile.write(data)

    def _error(self, status: int, message: str) -> None:
        self._reply(status, {"message": message})

    def _start_chunked(self, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        self.wfile.flush()

    def _chunk(self, data: bytes) -> None:
        self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
        self.wfile.flush()

    def _end_chunked(self) -> None:
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def _body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(length) if length else b""

    # -- dispatch --------------------------------------------------------------------------

    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_POST(self) -> None:
        self._dispatch("POST")

    def do_DELETE(self) -> None:
        self._dispatch("DELETE")

    def _dispatch(self, method: str) -> None:
        d = self.daemon
        url = urlsplit(self.path)
        query = parse_qs(url.query)
        body = self._body()
        d.requests.append((method, url.path, query, body))
        path = url.path
        for m, pattern, kind, arg in d._faults:
            if m == method and pattern.fullmatch(path):
                if kind == "fail":
                    status, message, when = arg
                    if when is not None and not when(body):
                        continue
                    return self._error(status, message)
                if kind == "hang":
                    time.sleep(arg)
                    return self._error(500, "hung")
                return self._reply(200, raw=arg)
        if path == "/version":
            return self._reply(200, d.version)
        prefix = "/v1.44"
        if not path.startswith(prefix + "/"):
            return self._error(400, f"unexpected API version in {path}")
        path = path[len(prefix):]
        for pattern, handler in _ROUTES:
            found = re.fullmatch(pattern, path)
            if found and handler.__name__.startswith(method.lower() + "_"):
                return handler(self, query, body, *found.groups())
        return self._error(404, f"page not found: {method} {path}")

    # -- endpoints -------------------------------------------------------------------------

    def get_version(self, query, body) -> None:
        self._reply(200, self.daemon.version)

    def get_info(self, query, body) -> None:
        self._reply(200, self.daemon.info)

    def post_create(self, query, body) -> None:
        d = self.daemon
        config = json.loads(body)
        d.create_bodies.append(config)
        image = d._find_image(config["Image"])
        if image is None:
            return self._error(404, f"No such image: {config['Image']}")
        name = (query.get("name") or [""])[0]
        if name and d._find(name):
            return self._error(409, f'Conflict. The container name "/{name}" is already in use')
        cid = uuid.uuid4().hex * 2
        ports = {key: [{"HostIp": b["HostIp"], "HostPort": b["HostPort"] or "49153"} for b in bound]
                 for key, bound in config["HostConfig"].get("PortBindings", {}).items()}
        with d._changed:
            d.containers[cid] = {
                "Id": cid, "Name": "/" + (name or "fake_" + cid[:6]), "Image": image,
                "Config": {"Labels": config.get("Labels") or {}, "Env": config.get("Env")},
                "HostConfig": config["HostConfig"],
                "State": {"Status": "created", "Running": False, "Pid": 0, "ExitCode": 0, "OOMKilled": False,
                          "Error": "", "StartedAt": ZERO_TIME, "FinishedAt": ZERO_TIME},
                "NetworkSettings": {"Ports": {}}, "_ports": ports,
            }
            d._event("container", "create", cid, {})
        self._reply(201, {"Id": cid, "Warnings": list(d.create_warnings)})

    def post_start(self, query, body, name) -> None:
        d = self.daemon
        cid = d._find(name)
        if cid is None:
            return self._error(404, f"No such container: {name}")
        c = d.containers[cid]
        if c["State"]["Running"]:
            return self._reply(304)
        with d._changed:
            c["State"].update(Status="running", Running=True, Pid=4242, StartedAt=_now())
            c["NetworkSettings"]["Ports"] = c["_ports"]
            d._event("container", "start", cid, {})
        if d.on_start is not None:
            d.on_start(cid, c)
        self._reply(204)

    def get_inspect(self, query, body, name) -> None:
        d = self.daemon
        cid = d._find(name)
        if cid is None:
            return self._error(404, f"No such container: {name}")
        self._reply(200, {k: v for k, v in d.containers[cid].items() if not k.startswith("_")})

    def post_wait(self, query, body, name) -> None:
        d = self.daemon
        cid = d._find(name)
        if cid is None:
            return self._error(404, f"No such container: {name}")
        # The daemon sends the headers at once and the body when the container stops, even when
        # another client removes the stopped container before this wait reads how it ended.
        self._start_chunked()
        with d._changed:
            state = d.containers[cid]["State"]
            while state["Running"] and cid in d.containers and not d._stopping:
                d._changed.wait(0.1)
        if state["Running"]:
            # The container went while it ran, as one the daemon loses: the wait ends with its
            # connection, and no answer.
            self.close_connection = True
            return
        self._chunk(json.dumps({"StatusCode": state["ExitCode"], "Error": None}).encode())
        self._end_chunked()

    def _stop(self, cid: str, signum: int, actions: tuple[str, ...]) -> None:
        d = self.daemon
        with d._changed:
            c = d.containers[cid]
            c["State"].update(Status="exited", Running=False, Pid=0, ExitCode=128 + signum, FinishedAt=_now())
            for action in actions:
                d._event("container", action, cid, {"signal": str(signum)} if action == "kill" else {})

    def post_kill(self, query, body, name) -> None:
        d = self.daemon
        cid = d._find(name)
        if cid is None:
            return self._error(404, f"No such container: {name}")
        if not d.containers[cid]["State"]["Running"]:
            return self._error(409, f"Cannot kill container: {name}: container {cid} is not running")
        signals = {"SIGTERM": 15, "SIGKILL": 9, "SIGINT": 2}
        self._stop(cid, signals[query["signal"][0]], ("kill", "die"))
        self._reply(204)

    def post_stop(self, query, body, name) -> None:
        d = self.daemon
        cid = d._find(name)
        if cid is None:
            return self._error(404, f"No such container: {name}")
        if not d.containers[cid]["State"]["Running"]:
            return self._reply(304)
        self._stop(cid, 15, ("kill", "die", "stop"))
        self._reply(204)

    def delete_container(self, query, body, name) -> None:
        d = self.daemon
        cid = d._find(name)
        if cid is None:
            return self._error(404, f"No such container: {name}")
        if d.containers[cid]["State"]["Running"] and query.get("force") != ["1"]:
            return self._error(409, "You cannot remove a running container. Stop the container first")
        with d._changed:
            del d.containers[cid]
            d._event("container", "destroy", cid, {})
        self._reply(204)

    def get_list(self, query, body) -> None:
        d = self.daemon
        filters = json.loads(query["filters"][0])
        wanted, names = filters.get("label", []), filters.get("name", [])
        rows = []
        for cid, c in d.containers.items():
            labels = c["Config"]["Labels"]
            if (all(labels.get(w.partition("=")[0]) == w.partition("=")[2] for w in wanted)
                    and all(n in c["Name"] for n in names)):
                rows.append({"Id": cid, "Names": [c["Name"]], "Image": c["Image"], "Labels": labels,
                             "State": c["State"]["Status"]})
        self._reply(200, rows)

    def get_events(self, query, body) -> None:
        d = self.daemon
        since = float(query["since"][0])
        until = float(query["until"][0]) if "until" in query else None
        filters = json.loads(query.get("filters", ["{}"])[0])
        self._start_chunked()
        sent = 0
        while True:
            with d._changed:
                fresh = d.events[sent:]
                sent = len(d.events)
            for event in fresh:
                if _matches(event, filters, since, until):
                    self._chunk(json.dumps(event).encode() + b"\n")
            if until is not None and time.time() >= until:
                break
            with d._changed:
                if d._stopping:
                    break
                if len(d.events) == sent:
                    d._changed.wait(0.05)
        self._end_chunked()

    def post_load(self, query, body) -> None:
        d = self.daemon
        d.loaded_bodies.append(body)
        names = []
        with tarfile.open(fileobj=io.BytesIO(body)) as tar:
            index = json.load(tar.extractfile("index.json"))
            members = tar.getnames()
            docker_manifest = json.load(tar.extractfile("manifest.json")) if "manifest.json" in members else None
        if d.image_store is not None:
            if d.image_store == "classic" and docker_manifest is None:
                return self._error(500, "invalid archive: does not contain a manifest.json")
            image_id = (index["manifests"][0]["digest"] if d.image_store == "containerd"
                        else "sha256:" + docker_manifest[0]["Config"].rsplit("/", 1)[1])
            d.images.setdefault(image_id, {"Id": image_id, "RepoTags": [], "RepoDigests": [],
                                           "Config": {"Labels": {}}, "Architecture": "amd64", "Os": "linux",
                                           "Size": len(body)})
            self._start_chunked()
            self._chunk(json.dumps({"stream": f"Loaded image ID: {image_id}\n"}).encode() + b"\r\n")
            return self._end_chunked()
        for m in index["manifests"]:
            name = (m.get("annotations") or {}).get("io.containerd.image.name")
            image_id = d.add_image(name or "", {})
            if not name:
                d.images[image_id]["RepoTags"] = []
            names.append(f"Loaded image: {name}\n" if name else f"Loaded image ID: {image_id}\n")
        self._start_chunked()
        if d.load_error:
            self._chunk(json.dumps({"errorDetail": {"message": d.load_error}, "error": d.load_error}).encode())
        else:
            for line in names:
                self._chunk(json.dumps({"stream": line}).encode() + b"\r\n")
        self._end_chunked()

    def get_images(self, query, body) -> None:
        d = self.daemon
        wanted = json.loads(query.get("filters", ["{}"])[0]).get("reference", [])
        rows = []
        for iid, image in d.images.items():
            # As containerd's image store does, only the names the reference filter matches are listed.
            tags = [t for t in image["RepoTags"] if all(t.rpartition(":")[0] == w or t == w for w in wanted)]
            if tags:
                rows.append({"Id": iid, "RepoTags": tags})
        self._reply(200, rows)

    def get_volumes(self, query, body) -> None:
        d = self.daemon
        wanted = json.loads(query.get("filters", ["{}"])[0]).get("name", [])
        self._reply(200, {"Volumes": [{"Name": name, "Driver": "local"} for name in d.volumes
                                      if all(w in name for w in wanted)], "Warnings": None})

    def delete_volume(self, query, body, name) -> None:
        d = self.daemon
        if name not in d.volumes:
            return self._error(404, f"get {name}: no such volume")
        if d.volumes[name] & set(d.containers):
            return self._error(409, f"remove {name}: volume is in use")
        del d.volumes[name]
        self._reply(204)

    def post_commit(self, query, body) -> None:
        from tests.substrate.docker.archives import write_archive

        d = self.daemon
        cid = d._find(query["container"][0])
        if cid is None:
            return self._error(404, f"No such container: {query['container'][0]}")
        config = json.loads(body) if body else {}
        d.commits.append((cid, query, config))
        made = write_archive(d.path.with_name(f"commit-{len(d.commits)}.tar"))
        image_id = made.config_digest if d.image_store == "classic" else made.manifest_digest
        d.saved[image_id] = made.path.read_bytes()
        d.images[image_id] = {"Id": image_id, "RepoTags": [f"{query['repo'][0]}:{query['tag'][0]}"],
                              "RepoDigests": [], "Config": config, "Architecture": "amd64", "Os": "linux",
                              "Size": len(d.saved[image_id])}
        self._reply(201, {"Id": image_id})

    def get_saved(self, query, body, ref) -> None:
        d = self.daemon
        iid = d._find_image(ref)
        if iid is None or iid not in d.saved:
            return self._error(404, f"No such image: {ref}")
        self._reply(200, raw=d.saved[iid])

    def get_archive(self, query, body, name) -> None:
        d = self.daemon
        cid = d._find(name)
        if cid is None:
            return self._error(404, f"No such container: {name}")
        path = query["path"][0]
        files = d.trees.get(d.containers[cid]["Image"], {}).get(path)
        if files is None:
            return self._error(404, f"Could not find the file {path} in container {name}")
        top = path.rstrip("/").rsplit("/", 1)[-1] or "."
        if d.archive_from_root:
            top = "/" + top
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            if isinstance(files, bytes):
                info = tarfile.TarInfo(top)
                info.size, info.mode = len(files), 0o644
                tar.addfile(info, io.BytesIO(files))
                files = {}
            else:
                info = tarfile.TarInfo(top)
                info.type, info.mode = tarfile.DIRTYPE, 0o755
                tar.addfile(info)
            for rel, data in sorted(files.items()):
                info = tarfile.TarInfo(f"{top}/{rel}")
                info.size, info.mode = len(data), 0o644
                tar.addfile(info, io.BytesIO(data))
        self._reply(200, raw=buf.getvalue())

    def get_image(self, query, body, ref) -> None:
        d = self.daemon
        iid = d._find_image(ref)
        if iid is None:
            return self._error(404, f"No such image: {ref}")
        self._reply(200, d.images[iid])

    def post_tag(self, query, body, ref) -> None:
        d = self.daemon
        iid = d._find_image(ref)
        if iid is None:
            return self._error(404, f"No such image: {ref}")
        name = f"{query['repo'][0]}:{query['tag'][0]}"
        for image in d.images.values():
            if name in image["RepoTags"]:
                image["RepoTags"].remove(name)
        d.images[iid]["RepoTags"].append(name)
        self._reply(201)

    def delete_image(self, query, body, ref) -> None:
        d = self.daemon
        iid = d._find_image(ref)
        if iid is None:
            return self._error(404, f"No such image: {ref}")
        image = d.images[iid]
        if ref in image["RepoTags"]:
            image["RepoTags"].remove(ref)
            out = [{"Untagged": ref}]
        else:
            image["RepoTags"] = []
            out = []
        if not image["RepoTags"]:
            del d.images[iid]
            out.append({"Deleted": iid})
        self._reply(200, out)


_ROUTES = [
    (r"/version", _Handler.get_version),
    (r"/info", _Handler.get_info),
    (r"/containers/create", _Handler.post_create),
    (r"/containers/json", _Handler.get_list),
    (r"/volumes", _Handler.get_volumes),
    (r"/volumes/([^/]+)", _Handler.delete_volume),
    (r"/commit", _Handler.post_commit),
    (r"/containers/([^/]+)/archive", _Handler.get_archive),
    (r"/containers/([^/]+)/start", _Handler.post_start),
    (r"/containers/([^/]+)/json", _Handler.get_inspect),
    (r"/containers/([^/]+)/wait", _Handler.post_wait),
    (r"/containers/([^/]+)/kill", _Handler.post_kill),
    (r"/containers/([^/]+)/stop", _Handler.post_stop),
    (r"/containers/([^/]+)", _Handler.delete_container),
    (r"/events", _Handler.get_events),
    (r"/images/json", _Handler.get_images),
    (r"/images/load", _Handler.post_load),
    (r"/images/(.+)/json", _Handler.get_image),
    (r"/images/(.+)/get", _Handler.get_saved),
    (r"/images/(.+)/tag", _Handler.post_tag),
    (r"/images/(.+)", _Handler.delete_image),
]
