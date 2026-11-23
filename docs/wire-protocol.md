# Wire protocol

A submission is an HTTP server. HLS starts it, waits until it is ready, sends it the task's
requests and reads its replies. Each task declares a contract: the endpoint the server must
serve and the shape of the requests and replies. HLS renders the same declaration into the
endpoint's section of the brief (the instructions the agent reads), so the agent is told
exactly what HLS sends.

```python
from hls.spec.contracts import RequestResponseContract
from hls.spec.wire import OpenAIChatProtocol

CONTRACT = RequestResponseContract(
    path="/v1/chat/completions",
    input_modalities=frozenset({"text"}),
    output_modalities=frozenset({"text"}),
    transport="http-sse",
    protocol=OpenAIChatProtocol(return_logprobs=True, top_logprobs=21),
)
```

| Field | Meaning |
|---|---|
| `path` | The inference endpoint. |
| `input_modalities`, `output_modalities` | Any of `text`, `image`, `audio`. Every verifier, the task's correctness checks, must accept the output modalities. |
| `transport` | `"http"` for one whole JSON reply (the default), `"http-sse"` for a reply streamed as server-sent events. |
| `protocol` | The request and reply shapes. Defaults to `OpenAIChatProtocol()`. |
| `port` | The port the server listens on. Defaults to 8000. |

## Starting the server and checking that it is ready

HLS installs the submission's repository with `pip install .` and runs it as
`python server.py --host 0.0.0.0 --port <P>`, where `<P>` is the contract's `port`; any extra
arguments the task declares follow. Unless the task sets `TaskSpec.network`, the server has no
network access. It reads the model's weights from `$HLS_WEIGHTS_DIR`, mounted read-only.

HLS then makes two readiness checks before it sends traffic:

1. `GET /health` once a second until it answers 200. The server must answer 200 only when the
   model is loaded and it can serve a request.
2. A `POST` of a minimal request to `path` every 2 seconds, until the status is below 400 or is
   exactly 400. Any other 4xx (404, 405 and 422 among them), a 5xx or no answer means not ready.
   The probe has the protocol's own request shape: one short prompt with `max_tokens` 1, an
   empty `audio_b64` for transcription, an empty `input` for speech.

Each health probe waits up to 5 seconds for an answer, and each inference probe up to 10
seconds. Each of the two checks may take up to 45 minutes (2,700 seconds). A server that exits
before it is ready, or is still not ready when a check's time runs out, scores zero.

## Protocols HLS provides

| Protocol | Request | Reply | Token ids | Streams |
|---|---|---|---|---|
| `OpenAIChatProtocol` | OpenAI chat completions: `messages` | `choices[0].message.content` | required | yes |
| `OpenAICompletionProtocol` | OpenAI completions: a raw `prompt` string | `choices[0].text` | required | yes |
| `OmniChatProtocol` | Chat completions whose `content` is a list of `text`, `image_url` and `input_audio` parts | `choices[0].message.content` | not asked for | yes |
| `AudioTranscriptionProtocol` | `audio_b64` (base64 PCM16 mono WAV), `sample_rate`, `language` | `text`, and `token_ids` when the server sends them | optional | no |
| `TTSStreamProtocol` | `input` (the text), `voice` | Audio frames, then one frame of codec codes when asked | optional (the codec ids) | yes |

All five are in `hls.spec.wire`. Every request carries `model`, which is `served-model` unless
the task sets another name. Every request except a speech (`TTSStreamProtocol`) request carries
the workload's output cap as `max_tokens`, and the server must honor it as given.

## OpenAI chat and completion requests

A request:

```json
{
  "model": "served-model",
  "messages": [{"role": "user", "content": "<prompt>"}],
  "max_tokens": 256,
  "return_token_ids": true
}
```

`messages` holds the whole conversation of a multi-turn session. When the task drives an agent
loop, `tools` holds the turn's tool definitions. A protocol declared with `sends_tools=True`
also sends `tool_choice: "none"`, so the model's tool call stays in `content` in the
checkpoint's own syntax. A turn that asks the model to finish its last assistant message carries
`continue_final_message: true` and `add_generation_prompt: false`.

The protocol's options add fields to the request and to what the reply must carry:

| Option | Request adds | Reply must carry |
|---|---|---|
| (always) | `return_token_ids: true` | `choices[0].token_ids` |
| `return_logprobs=True` | `logprobs: true` | `choices[0].logprobs.content[]`, one entry per token |
| `top_logprobs=N` (0 to 32, needs `return_logprobs`) | `top_logprobs: N` | `top_logprobs` on every entry |
| `stop_token_ids=(...)` | `stop_token_ids: [...]` | The stop token as the last of `token_ids` when the reply ended on one |
| `transport="http-sse"` | `stream: true`, `stream_options: {"include_usage": true}` | An SSE stream |

A reply:

```json
{
  "choices": [{
    "message": {"role": "assistant", "content": "<answer>"},
    "finish_reason": "stop",
    "token_ids": [1234, 5678]
  }],
  "usage": {"completion_tokens": 2}
}
```

HLS holds each reply to these rules:

- `choices[0]` is an object with a `finish_reason` string. The values `error`, `failed`,
  `aborted`, `cancelled` and `timeout`, in any of their spellings, mean the reply did not
  finish, and they fail the request. Any other value is accepted.
- `token_ids` is the list of integer vocabulary ids of the tokens the server emitted, in order.
  It is never a re-tokenization of the text. HLS counts, times and verifies these ids. With
  `stop_token_ids` set, an empty `token_ids` fails the request.
- With logprobs, every entry of `logprobs.content[]` carries `token`, an integer `token_id` and
  `logprob`, and every alternative in its `top_logprobs` carries `token_id` and `logprob`.
- `usage` is optional. HLS records `completion_tokens`, `prompt_tokens` and `total_tokens` when
  they are present.
- A reply may also carry an `hls_context` object (serving settings such as `max_context_len`
  and `max_num_seqs`) and a `gpu` object (a utilization and memory snapshot). HLS records their
  plain values for reports and scores nothing from them.

`OpenAICompletionProtocol` differs in two places: the request carries `prompt` in place of
`messages`, and the text is in `choices[0].text`. The prompt is the model input verbatim, so the
server applies no chat template. This protocol takes no logprob options.

`OmniChatProtocol` sends one user message whose `content` is a list of parts. It pins greedy
sampling on the request (`temperature` 0, `top_p` 1, `top_k` -1, no penalties) and does not ask
for `token_ids`.

## Streamed replies (server-sent events)

With `transport="http-sse"` the reply is an OpenAI event stream: lines of `data: <json>`, ended
by `data: [DONE]`. HLS skips blank lines, comments and `event:` lines.

- Each chunk is a JSON object. A chat chunk carries its text in `choices[0].delta.content`, and
  a completion chunk in `choices[0].text`.
- When the protocol requires token ids, every chunk that carries text also carries the
  `token_ids` of that text. A chunk may carry `token_ids` with no text.
- The reply's last chunk carries `finish_reason`. A stream that ends without one fails the
  request.
- Logprob entries arrive in `choices[0].logprobs.content[]` of the chunk that carries their
  tokens.
- A chunk with no `choices` may carry `usage`, or an `error`. HLS records that `error` as the
  cause when the stream then ends without a `finish_reason`.

HLS takes its timings from the stream. The time of the first chunk with `token_ids` is the time
to first token. Every token id is stamped with the arrival time of its chunk, so a chunk of
eight tokens counts as eight tokens arriving together. For a protocol that carries no token
ids, each chunk of text counts as one arrival.

## Speech-to-text and text-to-speech requests

`AudioTranscriptionProtocol` sends and receives plain JSON. The request carries `language` only
when the task sets one:

```json
{"model": "served-model", "audio_b64": "<base64 WAV>", "sample_rate": 16000, "language": "en", "max_tokens": 440}
```

```json
{"text": "<transcript>", "token_ids": [1234, 5678]}
```

`TTSStreamProtocol` sends `model`, `input` (the text), `voice`, `return_token_ids` and
`stream: true`. The reply is an event stream of frames, each `{"audio": "<base64 WAV>"}` and
optionally the frame's codec ids in `token_ids`, ended by `data: [DONE]`. Each frame is a
complete WAV file. HLS times the request to the last audio frame.

With `return_codec_codes=True`, the request asks for the codes, and the reply sends one more
frame after the audio. That frame holds `codec_codes`, one row of integer codes per codec frame.
With `return_logprobs=True` as well, it also holds the `codec_report` of log-probabilities the
codes were sampled from. The brief of `multimodal_tts`,
`tests/instances/golden/multimodal_tts.brief.md`, has the full shape.

## What makes a request fail

A request fails when the server answers with an HTTP error status, drops the connection, sends
a body that is not JSON, or breaks one of the reply rules. One failed request fails the run.

HLS records each failure in the trace of the drive. A drive is one pass of the task's requests
through the server, and its trace is the record of every request and reply. Each failure gets
one of these reasons:

| Reason | What happened |
|---|---|
| `server_error` | The server answered with an HTTP error status. |
| `transport` | The connection was refused, reset or dropped. |
| `protocol` | The reply arrived but broke the protocol's rules. |
| `stalled` | The stream delivered tokens, then none for longer than the task's stall limit. Only some tasks set one. |
| `deadline` | The request was still open when the drive's time ran out. |
| `circuit_breaker` | The drive stopped after too many consecutive failures. |

HLS asks for uncompressed replies, and it fails a reply sent in another content coding. A reply
may carry at most 256 MiB.

## Checks HLS makes when it loads a task

HLS checks a task's contract against its verifiers and its SLO (its latency budget) when the
task module is imported. It refuses a task that breaks one of these rules:

- A task declares at least one verifier.
- Every verifier accepts every output modality of the contract.
- A streaming latency SLO, which budgets the time to first token (TTFT) and the time per output
  token (TPOT), needs `transport="http-sse"` and a protocol that requires token ids.
- A verifier that scores the served tokens needs a protocol that requires token ids.
- A verifier that grades audio codec codes needs a protocol that asks for them
  (`return_codec_codes=True`).
- A verifier that scores reported distributions needs `return_logprobs=True`, `top_logprobs` of
  at least its report width (21 by default) and a reference that samples.
- A verifier that compares answers with the reference's needs a reference that decodes
  greedily.

## Writing a new protocol

A task whose endpoint has another shape subclasses `hls.spec.wire.WireProtocol`. The subclass
implements four methods:

| Method | Returns |
|---|---|
| `encode_request(env, *, stream=False)` | The JSON request body for one request. |
| `decode_response(data, request_id)` | The reply as an `Envelope`. Raises `ArtifactProtocolError` on a reply that breaks the shape. |
| `body_schema()` | An example request and reply. |
| `brief_section(*, path, streaming=False)` | The endpoint's section of the brief. |

It overrides these when the defaults do not fit:

| Method | Returns | Default |
|---|---|---|
| `probe_request()` | The readiness probe's body. | An OpenAI chat request with `max_tokens` 1. |
| `decode_stream(timed_chunks, request_id)` | For a streaming protocol: the reply rebuilt from `(time, chunk)` pairs, with the first-packet time and one timestamp per packet. | Refuses to stream. |
| `stream_token_count(chunk)` | For a streaming protocol: how many output tokens one chunk carries, which the stall limit counts. | 0. |
| `decode_partial(timed_chunks)` | What an incomplete stream produced, for the failure's record. | Nothing. |
| `trace_growth(*, stream)`, `stream_packet_bytes(frame_bytes)` | For a streaming protocol: the bounds the drive uses to size the trace. | Bounds for a whole reply only. |

Set the class attribute `emits_token_ids = True` when every reply must carry token ids, and
`conditioning = "raw"` when the prompt is the model input verbatim.
