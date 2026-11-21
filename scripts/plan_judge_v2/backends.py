"""Judge backends. Each loads its model once and answers batches of chat requests.

- vllm (default, Colab GPU): continuous batching with JSON-schema guided decoding.
- transformers (fallback): padded batched greedy generation without guided decoding.
- fake (tests): a deterministic in-process responder; never imports model libraries.

identity() is recorded in manifest.json and must match on resume; runtime()
(library versions, GPU) is recorded per session. Credentials are read by the
Hugging Face libraries from HF_TOKEN in the environment and never recorded.
"""

from dataclasses import dataclass
import json
import os

from scripts.plan_judge_v2.protocol import DECODING


@dataclass
class Request:
    messages: list
    schema: dict


@dataclass
class Result:
    text: str | None
    finish_reason: str | None = None
    error: dict | None = None


SECRET_ENV = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN")


def redact(text):
    """Remove Hugging Face credentials from text that will be saved."""
    for name in SECRET_ENV:
        secret = os.environ.get(name)
        if secret and len(secret) >= 8:
            text = text.replace(secret, "[REDACTED]")
    return text


def _error(exc):
    return {"type": type(exc).__name__, "message": redact(str(exc))[:2000]}


def resolve_revision(model_id, revision=None):
    """Pin a branch/tag to the exact Hub commit so the manifest identifies the weights."""
    from huggingface_hub import HfApi

    return HfApi().model_info(model_id, revision=revision).sha


class FakeBackend:
    """responder(messages, schema) -> reply text, or an Exception to simulate a failed request."""

    name = "fake"

    def __init__(self, responder, *, model_id="fake/judge", revision="fake-revision"):
        self.responder = responder
        self.model_id, self.revision = model_id, revision
        self.batches = []
        self.loaded = False

    def identity(self):
        return {"backend": self.name, "model_id": self.model_id, "revision_requested": self.revision,
                "revision_resolved": self.revision, "dtype": None, "max_model_len": None,
                "guided_decoding": False, "chat_template_kwargs": {}, "engine_args": {}}

    def runtime(self):
        return {"gpu": None, "libraries": {}}

    def load(self):
        self.loaded = True

    def generate(self, requests):
        self.batches.append(len(requests))
        results = []
        for request in requests:
            try:
                reply = self.responder(request.messages, request.schema)
            except Exception as exc:  # noqa: BLE001 - recorded as an error row
                results.append(Result(None, None, _error(exc)))
                continue
            if isinstance(reply, Result):
                results.append(reply)
            else:
                results.append(Result(reply, "stop"))
        return results


class _HubBackend:
    """Shared configuration for backends that load Hugging Face Hub weights."""

    def __init__(self, model_id, *, revision=None, dtype="auto", max_model_len=4096,
                 chat_template_kwargs=None, engine_args=None, resolver=resolve_revision):
        self.model_id = model_id
        self.revision_requested = revision
        self.revision_resolved = resolver(model_id, revision)
        self.dtype, self.max_model_len = dtype, max_model_len
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self.engine_args = dict(engine_args or {})

    def identity(self):
        return {"backend": self.name, "model_id": self.model_id,
                "revision_requested": self.revision_requested, "revision_resolved": self.revision_resolved,
                "dtype": self.dtype, "max_model_len": self.max_model_len,
                "guided_decoding": self.guided_decoding,
                "chat_template_kwargs": self.chat_template_kwargs, "engine_args": self.engine_args}

    @staticmethod
    def _gpu():
        try:
            import torch
            if torch.cuda.is_available():
                return {"name": torch.cuda.get_device_name(0),
                        "memory_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1),
                        "torch": torch.__version__, "cuda": torch.version.cuda}
        except Exception:  # noqa: BLE001 - runtime metadata only
            pass
        return None

    def _too_long(self, tokenizer, messages):
        ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                            **self.chat_template_kwargs)
        return len(ids) + DECODING["max_new_tokens"] > self.max_model_len, len(ids)


class VllmBackend(_HubBackend):
    name = "vllm"
    guided_decoding = True

    def __init__(self, model_id, *, gpu_memory_utilization=0.9, **kwargs):
        super().__init__(model_id, **kwargs)
        self.gpu_memory_utilization = gpu_memory_utilization
        self.llm = None

    def runtime(self):
        import vllm
        return {"gpu": self._gpu(), "libraries": {"vllm": vllm.__version__},
                "gpu_memory_utilization": self.gpu_memory_utilization}

    def load(self):
        from vllm import LLM
        self.llm = LLM(model=self.model_id, revision=self.revision_resolved,
                       tokenizer_revision=self.revision_resolved, dtype=self.dtype,
                       max_model_len=self.max_model_len, gpu_memory_utilization=self.gpu_memory_utilization,
                       seed=0, **self.engine_args)

    @staticmethod
    def sampling_params(schema):
        from vllm import SamplingParams
        common = {"temperature": DECODING["temperature"], "max_tokens": DECODING["max_new_tokens"]}
        try:  # vLLM >= 0.10.2
            from vllm.sampling_params import StructuredOutputsParams
        except ImportError:
            from vllm.sampling_params import GuidedDecodingParams
            return SamplingParams(**common, guided_decoding=GuidedDecodingParams(json=schema))
        return SamplingParams(**common, structured_outputs=StructuredOutputsParams(json=schema))

    def generate(self, requests):
        tokenizer = self.llm.get_tokenizer()
        results, runnable = [None] * len(requests), []
        for index, request in enumerate(requests):
            too_long, length = self._too_long(tokenizer, request.messages)
            if too_long:
                results[index] = Result(None, None, {"type": "PromptTooLong", "message":
                                                     f"{length} prompt tokens exceed max_model_len "
                                                     f"{self.max_model_len} minus the output budget"})
            else:
                runnable.append(index)
        if runnable:
            try:
                outputs = self.llm.chat([requests[i].messages for i in runnable],
                                        sampling_params=[self.sampling_params(requests[i].schema) for i in runnable],
                                        use_tqdm=False, chat_template_kwargs=self.chat_template_kwargs or None)
            except Exception as exc:  # noqa: BLE001 - every request in the batch gets the error
                for index in runnable:
                    results[index] = Result(None, None, _error(exc))
            else:
                for index, output in zip(runnable, outputs):
                    completion = output.outputs[0]
                    results[index] = Result(completion.text, completion.finish_reason)
        return results


class TransformersBackend(_HubBackend):
    name = "transformers"
    guided_decoding = False

    def __init__(self, model_id, **kwargs):
        super().__init__(model_id, **kwargs)
        self.model = self.tokenizer = None

    def runtime(self):
        import transformers
        return {"gpu": self._gpu(), "libraries": {"transformers": transformers.__version__}}

    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        dtype = self.dtype if self.dtype == "auto" else getattr(torch, self.dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision_resolved,
                                                       padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        options = {"revision": self.revision_resolved, "torch_dtype": dtype,
                   "device_map": "cuda" if torch.cuda.is_available() else None, **self.engine_args}
        try:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **options)
        except ValueError:
            # Multimodal checkpoints such as MedGemma 4B-it register an image-text-to-text class.
            from transformers import AutoModelForImageTextToText
            self.model = AutoModelForImageTextToText.from_pretrained(self.model_id, **options)
        self.model.eval()

    def generate(self, requests):
        import torch
        results, runnable = [None] * len(requests), []
        for index, request in enumerate(requests):
            too_long, length = self._too_long(self.tokenizer, request.messages)
            if too_long:
                results[index] = Result(None, None, {"type": "PromptTooLong", "message":
                                                     f"{length} prompt tokens exceed max_model_len "
                                                     f"{self.max_model_len} minus the output budget"})
            else:
                runnable.append(index)
        if not runnable:
            return results
        try:
            prompts = [self.tokenizer.apply_chat_template(requests[i].messages, add_generation_prompt=True,
                                                          tokenize=False, **self.chat_template_kwargs)
                       for i in runnable]
            batch = self.tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
            batch = batch.to(self.model.device)
            with torch.inference_mode():
                output = self.model.generate(**batch, do_sample=False, max_new_tokens=DECODING["max_new_tokens"],
                                             pad_token_id=self.tokenizer.pad_token_id)
            new_tokens = output[:, batch["input_ids"].shape[1]:]
        except Exception as exc:  # noqa: BLE001
            for index in runnable:
                results[index] = Result(None, None, _error(exc))
            return results
        # Chat models may stop on several ids (Gemma: <end_of_turn>); padding follows a stop.
        stops = self.model.generation_config.eos_token_id
        stops = set(stops if isinstance(stops, list) else [stops]) | {self.tokenizer.pad_token_id}
        for index, tokens in zip(runnable, new_tokens):
            finished = any(token in stops for token in tokens.tolist())
            text = self.tokenizer.decode(tokens, skip_special_tokens=True)
            results[index] = Result(text, "stop" if finished else "length")
        return results


BACKENDS = {"vllm": VllmBackend, "transformers": TransformersBackend}


def build_backend(name, model_id, **kwargs):
    if name not in BACKENDS:
        raise ValueError(f"Unknown backend {name}; choose one of {', '.join(BACKENDS)}")
    return BACKENDS[name](model_id, **kwargs)


def parse_engine_args(pairs):
    """KEY=JSON pairs, e.g. limit_mm_per_prompt={"image": 0} for text-only use of a multimodal model."""
    parsed = {}
    for pair in pairs or ():
        key, separator, value = pair.partition("=")
        if not separator or not key:
            raise ValueError(f"engine arg must be KEY=JSON: {pair}")
        try:
            parsed[key] = json.loads(value)
        except json.JSONDecodeError:
            parsed[key] = value
    return parsed
