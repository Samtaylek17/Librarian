from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class LLMLoggingProxy:

    def __init__(
        self,
        target_url: str,
        agent_id: str = "openhands_0",
        agent_role: str = "openhands",
        tokenizer_name: str = "",
    ) -> None:
        clean = target_url.rstrip("/")
        if clean.endswith("/v1"):
            clean = clean[:-3]
        self.target_url = clean
        self.agent_id = agent_id
        self.agent_role = agent_role
        self._calls: list[dict[str, Any]] = []
        self._server: asyncio.Server | None = None
        self._client: httpx.AsyncClient | None = None

        self._tokenizer = None
        self._tokenizer_name = tokenizer_name
        self._prev_token_ids: list[int] = []
        self._block_size = self._get_block_size(tokenizer_name)

    MODEL_BLOCK_SIZES: dict[str, int] = {
        "Qwen/Qwen3.6-35B-A3B-FP8": 1056,
        "Qwen/Qwen3.5-35B-A3B-FP8": 1056,
    }
    DEFAULT_BLOCK_SIZE = 16

    @classmethod
    def _get_block_size(cls, model_name: str) -> int:
        for pattern, size in cls.MODEL_BLOCK_SIZES.items():
            if pattern in model_name:
                return size
        return cls.DEFAULT_BLOCK_SIZE

    def _ensure_tokenizer(self, model_name: str) -> None:
        if self._tokenizer is not None:
            return
        name = self._tokenizer_name or model_name
        if not name:
            return
        try:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                name, trust_remote_code=True,
            )
            logger.info("Loaded tokenizer: %s", name)
        except Exception as e:
            logger.warning("Failed to load tokenizer %s: %s", name, e)

    def _compute_cached_tokens(self, messages: list[dict]) -> tuple[int, int]:
        if self._tokenizer is None:
            return 0, 0
        try:
            result = self._tokenizer.apply_chat_template(
                messages, add_generation_prompt=True,
            )
            if hasattr(result, 'input_ids'):
                token_ids = result.input_ids
                if hasattr(token_ids, 'tolist'):
                    token_ids = token_ids.tolist()
            elif isinstance(result, list) and result and isinstance(result[0], int):
                token_ids = result
            else:
                token_ids = list(result)
            total = len(token_ids)

            common = 0
            for a, b in zip(self._prev_token_ids, token_ids):
                if a != b:
                    break
                common += 1
            cached = (common // self._block_size) * self._block_size

            self._prev_token_ids = token_ids
            return total, cached
        except Exception as e:
            logger.debug("Tokenization failed: %s", e)
            return 0, 0

    def get_calls(self) -> list[dict[str, Any]]:
        return list(self._calls)

    async def _snapshot_energy(self) -> dict[str, Any] | None:
        if self._client is None:
            return None
        try:
            r = await self._client.get(f"{self.target_url}/energy", timeout=2.0)
            if r.status_code != 200:
                return None
            return r.json()
        except Exception:
            return None

    @staticmethod
    def _energy_diff_joules(e0: dict | None, e1: dict | None) -> tuple[float, list[float]]:
        if not e0 or not e1:
            return 0.0, []
        a = e0.get("gpus_mJ") or []
        b = e1.get("gpus_mJ") or []
        per_gpu_J: list[float] = []
        for x, y in zip(a, b):
            if x is None or y is None:
                per_gpu_J.append(0.0)
                continue
            d_mJ = y - x
            per_gpu_J.append(max(0.0, d_mJ / 1000.0))
        return sum(per_gpu_J), per_gpu_J

    async def start(self, port: int = 0) -> int:
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(600.0))
        self._server = await asyncio.start_server(self._handle, "0.0.0.0", port)
        actual_port = self._server.sockets[0].getsockname()[1]
        logger.info("LLM proxy on port %d -> %s", actual_port, self.target_url)
        return actual_port

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        if self._client:
            await self._client.aclose()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while True:
                request_line = await reader.readline()
                if not request_line:
                    break
                request_line = request_line.decode().strip()
                if not request_line:
                    break

                parts = request_line.split(" ", 2)
                if len(parts) < 3:
                    break
                method, path, _ = parts

                headers: dict[str, str] = {}
                content_length = 0
                while True:
                    line = await reader.readline()
                    if not line or line == b"\r\n":
                        break
                    decoded = line.decode().strip()
                    if ":" in decoded:
                        key, val = decoded.split(":", 1)
                        headers[key.strip().lower()] = val.strip()
                        if key.strip().lower() == "content-length":
                            content_length = int(val.strip())

                body = b""
                if content_length > 0:
                    body = await reader.readexactly(content_length)

                is_chat = "chat/completions" in path
                request_data = None
                if is_chat and body:
                    try:
                        request_data = json.loads(body)
                    except json.JSONDecodeError:
                        pass

                is_streaming = request_data and request_data.get("stream")

                if is_streaming:
                    await self._handle_streaming(method, path, headers, body, request_data, writer)
                else:
                    resp_status, resp_headers, resp_body = await self._forward(
                        method, path, headers, body, request_data,
                    )
                    writer.write(f"HTTP/1.1 {resp_status}\r\n".encode())
                    for k, v in resp_headers.items():
                        writer.write(f"{k}: {v}\r\n".encode())
                    writer.write(f"Content-Length: {len(resp_body)}\r\n".encode())
                    writer.write(b"Connection: keep-alive\r\n\r\n")
                    writer.write(resp_body)
                    await writer.drain()

        except (asyncio.IncompleteReadError, ConnectionResetError, BrokenPipeError):
            pass
        except Exception as e:
            logger.debug("Proxy error: %s", e)
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    async def _forward(self, method, path, headers, body, request_data):
        target = f"{self.target_url}{path}"
        fwd_headers = {k: v for k, v in headers.items() if k not in ("host", "content-length", "transfer-encoding", "connection")}
        is_chat = "chat/completions" in path and request_data is not None
        energy_before = await self._snapshot_energy() if is_chat else None
        started_at = time.time()
        start = time.monotonic()
        try:
            resp = await self._client.request(method, target, content=body, headers=fwd_headers)
        except Exception as e:
            err_body = json.dumps({"error": str(e)}).encode()
            return "502 Bad Gateway", {"Content-Type": "application/json"}, err_body

        duration = time.monotonic() - start
        ended_at = time.time()
        if is_chat and resp.status_code == 200:
            energy_after = await self._snapshot_energy()
            try:
                self._log_call(
                    request_data, json.loads(resp.content), duration,
                    started_at=started_at, ended_at=ended_at,
                    energy_before=energy_before, energy_after=energy_after,
                )
            except json.JSONDecodeError:
                pass

        return f"{resp.status_code} OK", {"Content-Type": resp.headers.get("content-type", "application/json")}, resp.content

    async def _handle_streaming(self, method, path, headers, body, request_data, writer):
        target = f"{self.target_url}{path}"
        if "stream_options" not in request_data or not request_data["stream_options"]:
            request_data["stream_options"] = {}
        request_data["stream_options"]["include_usage"] = True
        body = json.dumps(request_data).encode()

        fwd_headers = {k: v for k, v in headers.items() if k not in ("host", "content-length", "transfer-encoding", "connection")}
        energy_before = await self._snapshot_energy()
        started_at = time.time()
        start = time.monotonic()
        buffer = bytearray()

        try:
            async with self._client.stream(method, target, content=body, headers=fwd_headers) as resp:
                ct = resp.headers.get("content-type", "text/event-stream")
                writer.write(f"HTTP/1.1 {resp.status_code} OK\r\nContent-Type: {ct}\r\nConnection: close\r\n\r\n".encode())
                await writer.drain()
                async for chunk in resp.aiter_raw():
                    buffer.extend(chunk)
                    writer.write(chunk)
                    await writer.drain()
        except Exception as e:
            logger.error("Streaming failed: %s", e)
            return

        duration = time.monotonic() - start
        ended_at = time.time()
        if resp.status_code == 200:
            energy_after = await self._snapshot_energy()
            self._log_streaming_call(
                request_data, bytes(buffer), duration,
                started_at=started_at, ended_at=ended_at,
                energy_before=energy_before, energy_after=energy_after,
            )

    def _log_streaming_call(self, request_data, buffer, duration, *,
                            started_at=None, ended_at=None,
                            energy_before=None, energy_after=None):
        text = buffer.decode(errors="replace")
        last_data_json = None
        response_parts = []
        for line in text.split("\n"):
            line = line.strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:].strip()
            if payload == "[DONE]":
                continue
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                if delta.get("content"):
                    response_parts.append(delta["content"])
            if chunk.get("usage"):
                last_data_json = chunk

        if not last_data_json:
            return

        usage = last_data_json.get("usage", {})
        self._record_call(
            request_data, usage, "".join(response_parts), duration,
            started_at=started_at, ended_at=ended_at,
            energy_before=energy_before, energy_after=energy_after,
        )

    def _log_call(self, request_data, response_data, duration, *,
                  started_at=None, ended_at=None,
                  energy_before=None, energy_after=None):
        usage = response_data.get("usage", {})
        response_text = ""
        choices = response_data.get("choices", [])
        if choices:
            response_text = choices[0].get("message", {}).get("content", "") or ""

        self._record_call(
            request_data, usage, response_text, duration,
            started_at=started_at, ended_at=ended_at,
            energy_before=energy_before, energy_after=energy_after,
        )

    def _record_call(self, request_data, usage, response_text, duration, *,
                     started_at=None, ended_at=None,
                     energy_before=None, energy_after=None):
        server_cached = 0
        prompt_details = usage.get("prompt_tokens_details", {})
        if isinstance(prompt_details, dict):
            server_cached = prompt_details.get("cached_tokens", 0) or 0

        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        model_name = request_data.get("model", "")
        messages = request_data.get("messages", [])
        self._ensure_tokenizer(model_name)
        client_total, client_cached = self._compute_cached_tokens(messages)

        joules_total, joules_per_gpu = self._energy_diff_joules(energy_before, energy_after)

        self._calls.append({
            "agent_id": self.agent_id,
            "agent_role": self.agent_role,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cached_tokens": server_cached,
            "theoretical_cached_tokens": client_cached,
            "prefill_tokens": input_tokens - (client_cached or server_cached),
            "duration_seconds": round(duration, 3),
            "started_at": started_at,
            "ended_at": ended_at,
            "joules": round(joules_total, 3),
            "joules_per_gpu": [round(j, 3) for j in joules_per_gpu],
            "input": messages,
            "action": "",
            "response": response_text,
            "observation": "",
            "model": model_name,
            "attempt": 1,
            "error": None,
        })
