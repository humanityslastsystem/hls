"""Records when a stage context copies cache entries, pins images and starts roles, so a test can
check that a stage copies and pins everything it needs before it starts its own roles: a copy or
an extraction is a role that runs alone."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from hls.pipeline.stage import StageContext

#: The context's methods that copy a cache entry to the node, read-only or as a writable seed, or
#: pin an image.
STAGING = ("materialize", "seed", "node_image", "pin_file")
#: The context's methods that start a role.
STARTS = ("start_role", "start_container")


class CallOrder:
    """Wraps *ctx*'s copying, pinning and starting methods on the instance. It records, in
    order, each copy or pin the stage asks for as it begins, with the roles then running, and as
    it returns, and each role the stage starts itself. What a copy or a pin does on its own
    behalf, a role it starts or a copy it makes, belongs to it and is not recorded."""

    def __init__(self, ctx: StageContext) -> None:
        #: The wrapped context.
        self.ctx = ctx
        #: ``("begin", method, running roles)``, ``("end", method, ())`` and ``("role", name, ())``,
        #: in the order they happened.
        self.calls: list[tuple[str, str, tuple[str, ...]]] = []
        self._depth = 0
        for name in STAGING:
            setattr(ctx, name, self._staging(name, getattr(ctx, name)))
        for name in STARTS:
            setattr(ctx, name, self._start(getattr(ctx, name)))

    def _staging(self, method: str, call: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            outer = self._depth == 0
            if outer:
                self.calls.append(("begin", method, tuple(self.ctx.running_roles())))
            self._depth += 1
            try:
                return call(*args, **kwargs)
            finally:
                self._depth -= 1
                if outer:
                    self.calls.append(("end", method, ()))
        return wrapper

    def _start(self, call: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(name: str, *args: Any, **kwargs: Any) -> Any:
            if self._depth == 0:
                self.calls.append(("role", name, ()))
            return call(name, *args, **kwargs)
        return wrapper

    def staged(self) -> list[str]:
        """The copies and pins the stage made, by method, in order."""
        return [method for kind, method, _ in self.calls if kind == "begin"]

    def while_running(self) -> list[tuple[str, tuple[str, ...]]]:
        """Each copy or pin that began while roles of the stage ran, with those roles."""
        return [(method, running) for kind, method, running in self.calls if kind == "begin" and running]

    def roles_before_staged(self) -> list[str]:
        """The roles the stage started before its last copy or pin returned."""
        last = max((i for i, (kind, _, _) in enumerate(self.calls) if kind == "end"), default=-1)
        return [name for kind, name, _ in self.calls[:last] if kind == "role"]
