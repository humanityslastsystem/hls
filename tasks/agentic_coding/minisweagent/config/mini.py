"""The templates from mini-SWE-agent's ``src/minisweagent/config/mini.yaml``.

Upstream: https://github.com/SWE-agent/mini-swe-agent at
``04d809ceab9df28f9adaed044884180159172930``. MIT licensed, Copyright (c) 2025
Kilian A. Lieret and Carlos E. Jimenez; see ``LICENSE`` beside this package.

Python strings rather than the YAML file, so nothing needs a YAML loader, and
``str.format`` placeholders rather than jinja2, so nothing needs jinja2 either. The
name mirrors ``mini.yaml`` so the pair lines up under ``diff -r``.

Divergences:

1. **No jinja2.** Upstream renders through ``Template(..., undefined=StrictUndefined)``.
   No placeholder here needs more than ``str.format``. Upstream's two templates that
   branch in jinja -- the observation's short/long split and the format error's
   truncated/unreadable split -- are two strings each here, chosen in Python.
   ``StrictUndefined`` catches a typo'd placeholder; ``test_mini_swe_loop`` renders
   every template instead, which costs nothing at run time.
2. The task is a Terminal-Bench task in a prepared container, not a SWE-bench issue in
   a checkout, so the workflow steps name looking around and matching an output format
   instead of reproducing and fixing a bug. The surrounding structure, the command
   execution rules and the submit sentinel are upstream's.
3. Upstream asks for reasoning text outside the tool call, twice. Dropped; see the
   comment on ``INSTANCE_TEMPLATE``.
4. Upstream's ``<system_information>`` block renders the environment's platform.
   Dropped: it describes the machine the loop runs on, and our commands run in a
   separate Apptainer container, so it would be telling the model about a machine it
   cannot see. The macOS ``sed -i ''`` warning goes with it, the container being Linux.
5. The observation's elision threshold is a parameter rather than upstream's
   hard-coded 10,000 characters; see ``OBSERVATION_LONG``.

Braces that are part of the text are doubled, because these are ``str.format``
strings. ``test_every_template_renders`` is what keeps that honest.
"""
from __future__ import annotations

import json

#: Upstream's ``agent.system_template``, verbatim, and it is one sentence on purpose.
#: Everything a run needs to know rides on the instance template instead, which is
#: where upstream puts it.
SYSTEM_TEMPLATE = "You are a helpful assistant that can interact with a computer.\n"

#: Upstream's ``agent.instance_template``, restated for a Terminal-Bench task.
#:
#: What is deliberately NOT carried over: upstream's "Your response SHOULD include
#: reasoning text explaining what you're doing" and the "**Reasoning text** where you
#: explain your analysis and plan" list item. Both ask for prose reasoning outside the
#: tool call. This checkpoint's chat template ends its generation prompt with
#: ``<|assistant|><think>``, so a reasoning channel is already open before the model
#: writes a token, and a prompt that also asks for written reasoning gets both: the
#: model reasons in the channel, closes it, writes the prose it was asked for, and
#: closes a second time. Asking a thinking model to think twice is not free on a task
#: that scores how long a turn takes.
INSTANCE_TEMPLATE = """\
Please solve the following task.

<task>
{task}
</task>

Your shell session has started in {work_dir}, inside a container prepared for this
task. You can run any bash command and edit any file. The task statement names the
paths it expects you to read and write.

## Recommended Workflow

This workflow should be done step-by-step so that you can iterate on your work and any
possible problems.

1. Look around the working directory to see what you have been given
2. Read the inputs the task names
3. Do the work, checking intermediate results as you go
4. Verify your output matches what the task asked for, including its exact format
5. Finish your work by issuing the following command: `echo {submit_sentinel}`.
   Do not combine it with any other command. <important>After this command, you cannot
   continue working on this task.</important>

## Command Execution Rules

You are operating in an environment where

1. You issue at least one command
2. The system executes the command(s) in a subshell
3. You see the result(s)
4. You write your next command(s)

**CRITICAL REQUIREMENTS:**

- Your response MUST include AT LEAST ONE bash tool call
- Directory or environment variable changes are not persistent. Every action is \
executed in a new subshell.
- However, you can prefix any action with `MY_ENV_VAR=MY_VALUE cd /path/to/working/dir \
&& ...` or write/load environment variables from files
- Finish your work by issuing the following command: `echo {submit_sentinel}`.
  Do not combine it with any other command. <important>After this command, you cannot
  continue working on this task.</important>

Example of a CORRECT response:
<example_response>
[Makes bash tool call with {{"command": "ls -la"}} as arguments]
</example_response>

## Useful command examples

### Create a new file:

```bash
cat <<'EOF' > newfile.py
import numpy as np
hello = "world"
print(hello)
EOF
```

### Edit files with sed:

```bash
# Replace all occurrences
sed -i 's/old_string/new_string/g' filename.py

# Replace only first occurrence
sed -i 's/old_string/new_string/' filename.py

# Replace first occurrence on line 1
sed -i '1s/old_string/new_string/' filename.py

# Replace all occurrences in lines 1-10
sed -i '1,10s/old_string/new_string/g' filename.py
```

### View file content:

```bash
# View specific lines with numbers
nl -ba filename.py | sed -n '10,20p'
```

### Any other command you want to run

```bash
anything
```
"""

#: Upstream's ``model.observation_template``, short branch: a JSON object carrying the
#: whole output.
OBSERVATION_SHORT = """\
{{
  "returncode": {returncode},
  "output": {output}{exception_info}
}}"""

#: Upstream's long branch: head and tail kept, the middle counted rather than shown,
#: because a failure's cause is usually at one end.
#:
#: Upstream elides at a hard-coded 10,000 characters. It is a parameter here because
#: on this task it decides how fast the declared context window fills.
#:
#: PER COMMAND, not per turn. Each output is rendered through this template on its
#: own, and a reply on this checkpoint usually carries two calls. That is the easy
#: thing to misread, it is not visible from where the call sits, and it doubles a
#: turn's worst case. Command output tokenizes at roughly 4.4 characters per token, so
#: against the 1M window:
#:
#:     cap       1 cmd/turn   2 cmd/turn   1M window reached in
#:     32,000    7,300 tok    14,500 tok   144 / 72 turns
#:     10,000    2,300 tok     4,500 tok   461 / 231 turns
#:
#: 32,000 is chosen off that table. This is a long-context task, so the window has to
#: be reachable by an ordinary trajectory and not only by the most verbose one
#: possible: at 10,000 it takes 231 of the 250 turns a trajectory is allowed, with
#: nothing to spare, while at 32,000 it takes 72.
OBSERVATION_LONG = """\
{{
  "returncode": {returncode},
  "output_head": {output_head},
  "output_tail": {output_tail},
  "elided_chars": {elided_chars},
  "warning": "Output too long."{exception_info}
}}"""


def render_observation(output: dict, *, output_limit: int) -> str:
    """One command's result as the model sees it.

    Upstream's jinja conditional, in Python. Values go through ``json.dumps`` for the
    same reason upstream pipes them through ``tojson``: command output is arbitrary
    bytes and the result has to stay parseable as the JSON object it looks like."""
    text = output.get("output", "") or ""
    exception_info = output.get("exception_info")
    suffix = (f', "exception_info": {json.dumps(exception_info)}'
              if exception_info else "")
    returncode = json.dumps(output.get("returncode"))
    if len(text) < output_limit:
        return OBSERVATION_SHORT.format(returncode=returncode,
                                        output=json.dumps(text),
                                        exception_info=suffix)
    half = output_limit // 2
    return OBSERVATION_LONG.format(returncode=returncode,
                                   output_head=json.dumps(text[:half]),
                                   output_tail=json.dumps(text[-half:]),
                                   elided_chars=len(text) - output_limit,
                                   exception_info=suffix)


#: Upstream's ``model.format_error_template``, truncated branch. Upstream selects it
#: in jinja on ``finish_reason``; the caller selects it here, from the token count
#: rather than from the label, because the label is a string the server chooses.
FORMAT_ERROR_TRUNCATED = """\
Your previous response reached the output token limit before you produced a tool \
call, so it was cut off. Respond more concisely and finish with exactly one bash tool \
call. If you need to think more, do so briefly.
"""

#: Upstream's other branch: the reply was finished and could not be read.
FORMAT_ERROR_UNREADABLE = """\
Tool call error:

<error>
{error}
</error>

Here is general guidance on how to submit correct toolcalls:

Every response needs to use the 'bash' tool at least once to execute commands.

Call the bash tool with your command as the argument:
- Tool: bash
- Arguments: {{"command": "your_command_here"}}

If you want to end the task, please issue the following command: `echo {submit_sentinel}`
without any other command.
"""

__all__ = ["FORMAT_ERROR_TRUNCATED", "FORMAT_ERROR_UNREADABLE", "INSTANCE_TEMPLATE",
           "OBSERVATION_LONG", "OBSERVATION_SHORT", "SYSTEM_TEMPLATE",
           "render_observation"]
