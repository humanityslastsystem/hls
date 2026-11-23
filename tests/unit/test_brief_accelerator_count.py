"""The weights bullet must name every device the task declares, not always `cuda:0`.

"The accelerator is `cuda:0`" is false for a task that declares several devices, such as
multimodal_omni with two H200s: it reads as a one-device budget while the contestant is
measured on a two-device pin. The Hardware Envelope section says "2xH200", so the bullet
must agree with it.
"""

from __future__ import annotations

from tests.unit.brief_task import render_brief


def _task_with_devices(n: int):
    """The sample task, restated with ``n`` accelerators.

    Built by copy rather than by reaching for a real two-device task, so the test pins the
    template's behaviour at an arbitrary count instead of one task's current declaration.
    """
    from tasks.sample_task.task import TASK

    accelerator = TASK.hardware.accelerator.model_copy(update={"count": n})
    return TASK.model_copy(
        update={"hardware": TASK.hardware.model_copy(update={"accelerator": accelerator})})


def test_one_device_reads_exactly_as_it_always_has():
    """The single-device wording is unchanged, so no existing brief shifts under this."""
    text, _ = render_brief(_task_with_devices(1))

    assert "The accelerator is `cuda:0`." in text
    assert "accelerators are" not in text


def test_two_devices_are_both_named():
    text, _ = render_brief(_task_with_devices(2))

    assert "The accelerators are `cuda:0` through `cuda:1`; all 2 are yours." in text
    assert "The accelerator is `cuda:0`." not in text


def test_eight_devices_name_the_last_one_correctly():
    """The range is ``count - 1``; an off-by-one here understates or invents a device."""
    text, _ = render_brief(_task_with_devices(8))

    assert "`cuda:0` through `cuda:7`; all 8 are yours." in text


def test_the_bullet_still_ends_in_a_line_break():
    """The conditional closes with ``+%}`` for this reason.

    ``trim_blocks`` eats the newline after a block tag that ends a line, which runs the
    next bullet onto this one -- every assertion above still passes when it does, which is
    why the boundary is asserted across the newline rather than on the bullet alone.
    """
    for n in (1, 2):
        text, _ = render_brief(_task_with_devices(n))
        assert "\n- Data: a development split is on disk" in text, (
            f"the weights bullet ran into the next one at accelerator_count={n}")
