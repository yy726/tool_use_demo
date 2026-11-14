"""GPU-side helpers for the notebooks: loading, generation, saving, pushing.

Everything here needs a GPU and the Unsloth / TRL stack, so it is exercised by
the notebooks' SMOKE mode rather than by the unit tests. The calls follow the
two Unsloth Qwen3 notebooks (Tool_Calling and GRPO-Tool_Use).
"""

from __future__ import annotations

import gc
import importlib.util
import inspect
import os

from .persist import push_with_retry

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def has_vllm() -> bool:
    """vLLM is installed on non-T4 GPUs by the Unsloth install cell."""
    return importlib.util.find_spec("vllm") is not None


def load_model(config, fast_inference: bool = False, gpu_memory_utilization: float = 0.6):
    """Load the base model with a fresh LoRA. `fast_inference=True` adds Unsloth's vLLM engine (GRPO)."""
    from unsloth import FastLanguageModel

    kwargs = dict(model_name=config.model_name, max_seq_length=config.max_seq_length, load_in_4bit=False)
    if fast_inference:
        kwargs.update(fast_inference=True, max_lora_rank=config.lora_rank, gpu_memory_utilization=gpu_memory_utilization)
    model, tokenizer = FastLanguageModel.from_pretrained(**kwargs)
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_rank,
        target_modules=LORA_TARGETS,
        lora_alpha=config.lora_rank * 2,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=config.seed,
    )
    return model, tokenizer


def load_lora_weights(model, adapter_dir: str) -> None:
    """Load saved adapter weights into the model's (same-shaped) LoRA, e.g. SFT -> GRPO."""
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    state = load_file(os.path.join(adapter_dir, "adapter_model.safetensors"))
    result = set_peft_model_state_dict(model, state)
    unexpected = list(getattr(result, "unexpected_keys", []) or [])
    if unexpected:
        raise RuntimeError(f"adapter in {adapter_dir} does not match the model's LoRA ({len(unexpected)} unexpected keys, e.g. {unexpected[:2]})")
    print(f"Loaded {len(state)} LoRA tensors from {adapter_dir}")


def filter_kwargs(cls, kwargs: dict) -> dict:
    """Keep only the arguments `cls` accepts, so one notebook works across TRL versions."""
    accepted = set(inspect.signature(cls.__init__).parameters)
    dropped = sorted(k for k in kwargs if k not in accepted)
    if dropped:
        print(f"{cls.__name__} does not accept {dropped}: skipped")
    return {k: v for k, v in kwargs.items() if k in accepted}


def tokenizer_kwarg(trainer_cls, tokenizer) -> dict:
    """`processing_class=` on current TRL, `tokenizer=` on older versions / Unsloth's wrapper."""
    name = "processing_class" if "processing_class" in inspect.signature(trainer_cls.__init__).parameters else "tokenizer"
    return {name: tokenizer}


# --------------------------------------------------------------- generation
def _clean(text: str, tokenizer) -> str:
    for token in ("<|im_end|>", "<|endoftext|>"):
        text = text.split(token)[0]
    return text.replace(tokenizer.pad_token or "", "") if tokenizer.pad_token else text


def gpu_report(tag: str = "") -> None:
    """Print where GPU memory is: PyTorch's allocator vs. everything else (cuda context, other GPU libraries, other processes)."""
    import subprocess

    import torch

    if not torch.cuda.is_available():
        return
    free, total = torch.cuda.mem_get_info()
    gib = lambda n: f"{n / 2**30:.2f}"
    reserved = torch.cuda.memory_reserved()
    print(
        f"[gpu {tag}] total {gib(total)} GiB | free {gib(free)} | torch allocated {gib(torch.cuda.memory_allocated())}"
        f" | torch reserved {gib(reserved)} | used OUTSIDE torch's allocator {gib(total - free - reserved)}"
    )
    if total - free - reserved > 4 * 2**30:
        print(f"[gpu {tag}] several GiB are held outside PyTorch: another GPU library in this process (JAX and TensorFlow pre-allocate) or another process")
    try:
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip()
        print(f"[gpu {tag}] processes on this GPU (pid, memory): {apps or 'none reported'}")
    except Exception:
        pass


def hf_generate_fn(model, tokenizer, max_new_tokens: int = 384, batch_size: int = 32):
    """Greedy batched generation with transformers. Call `FastLanguageModel.for_inference(model)` first.

    On a CUDA out-of-memory error the failed batch is retried in halves (down to one prompt) instead of losing the
    whole benchmark run.

    Two deliberate details, both from a full run that died with "CUDA error: an illegal memory access" inside
    Unsloth's decode path:

    - `torch.cuda.empty_cache()` is NOT called between batches (only after an out-of-memory error). Unsloth's decode
      step launches its own GPU kernels on raw pointers; as long as freed blocks stay in PyTorch's pool, a stale
      pointer reads stale data, but once the pool hands them back to the driver the same read is a hard fault that
      kills the CUDA context for the whole session.
    - Each batch ends with `torch.cuda.synchronize()`, so an asynchronous GPU error is raised for the batch that
      caused it, with that batch's shapes in the message, instead of somewhere in a later call.
    """
    import torch

    def free_memory():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def run(prompts: list[str]) -> list[str]:
        enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        try:
            with torch.inference_mode():
                out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError:
            raise
        except RuntimeError as error:
            if "CUDA error" in str(error):
                shape = tuple(enc["input_ids"].shape)
                print(
                    f"\nGPU fault during generation: batch of {shape[0]} prompts, padded to {shape[1]} tokens, max_new_tokens={max_new_tokens}.\n"
                    "The CUDA context of this session is now unusable: choose Runtime > Restart session, then Run all\n"
                    "(finished stages are on Drive and are skipped). To get an exact stack trace, set DEBUG_CUDA = True\n"
                    "in the first cell before re-running; generation is then slower but errors are reported where they happen."
                )
            raise
        texts = tokenizer.batch_decode(out[:, enc["input_ids"].shape[1] :], skip_special_tokens=False)
        return [_clean(text, tokenizer) for text in texts]

    def run_safe(prompts: list[str]) -> list[str]:
        try:
            return run(prompts)
        except torch.cuda.OutOfMemoryError:
            gpu_report(f"OOM with {len(prompts)} prompts")
            free_memory()
            if len(prompts) == 1:
                raise
            half = len(prompts) // 2
            print(f"CUDA out of memory with {len(prompts)} prompts: retrying as {half} + {len(prompts) - half}")
            return run_safe(prompts[:half]) + run_safe(prompts[half:])

    def generate(prompts: list[str]) -> list[str]:
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))  # similar lengths -> less padding
        outputs = [""] * len(prompts)
        side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        try:
            for start in range(0, len(order), batch_size):
                index = order[start : start + batch_size]
                for i, text in zip(index, run_safe([prompts[i] for i in index])):
                    outputs[i] = text
        finally:
            tokenizer.padding_side = side
        return outputs

    return generate


def vllm_generate_fn(model, tokenizer, lora_dir: str | None = None, max_new_tokens: int = 384):
    """Greedy generation with Unsloth's vLLM engine, optionally through a saved LoRA."""
    from vllm import SamplingParams

    params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens, skip_special_tokens=False)
    lora_request = model.load_lora(lora_dir) if lora_dir else None

    def generate(prompts: list[str]) -> list[str]:
        results = model.fast_generate(list(prompts), sampling_params=params, lora_request=lora_request)
        return [_clean(r.outputs[0].text, tokenizer) for r in results]

    return generate


# ------------------------------------------------------- saving and pushing
def save_adapter(model, tokenizer, chat_format, directory: str) -> str:
    """Save the LoRA adapter and a tokenizer that carries the serving chat template."""
    os.makedirs(directory, exist_ok=True)
    if hasattr(model, "save_lora"):  # Unsloth's vLLM engine reads adapters saved this way
        model.save_lora(directory)
    else:
        model.save_pretrained(directory)
    stock = tokenizer.chat_template
    tokenizer.chat_template = chat_format.serving
    try:
        tokenizer.save_pretrained(directory)
    finally:
        tokenizer.chat_template = stock
    return directory


def export_model(
    model,
    tokenizer,
    chat_format,
    config,
    stage: str,
    hf_user: str | None,
    adapter_dir: str,
    merged_dir: str,
    card: str = "",
    push: bool = True,
    private: bool = False,
    gguf: bool = True,
    gguf_quantizations=("q8_0", "q4_k_m"),
) -> dict:
    """Write merged weights locally (BFCL needs them) and push adapter / merged / GGUF to the Hub.

    Returns which of the pushes succeeded. Nothing here raises on a failed
    upload: the adapter is already safe on Drive and the cell can be re-run.
    """
    token = os.environ.get("HF_TOKEN")
    stock = tokenizer.chat_template
    tokenizer.chat_template = chat_format.serving  # what vLLM / llama.cpp will use
    status = {}
    try:
        if not os.path.exists(os.path.join(merged_dir, "config.json")):
            model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
        if card:
            for directory in (adapter_dir, merged_dir):
                with open(os.path.join(directory, "README.md"), "w", encoding="utf-8") as f:
                    f.write(card)
        if push and hf_user and token:
            from huggingface_hub import HfApi

            api = HfApi(token=token)

            def upload(repo: str, folder: str):
                def run():
                    api.create_repo(repo, private=private, exist_ok=True)
                    api.upload_folder(repo_id=repo, folder_path=folder, commit_message=f"{stage} checkpoint")

                return run

            status["adapter"] = push_with_retry(upload(config.repo(hf_user, stage, "lora"), adapter_dir), f"{stage} adapter")
            status["merged"] = push_with_retry(upload(config.repo(hf_user, stage), merged_dir), f"{stage} merged model")
            if gguf:
                repo = config.repo(hf_user, stage, "GGUF")
                extra = {"private": True} if private else {}
                status["gguf"] = push_with_retry(
                    lambda: model.push_to_hub_gguf(repo, tokenizer, quantization_method=list(gguf_quantizations), token=token, **extra),
                    f"{stage} GGUF",
                    retries=1,
                )
        else:
            print("Hub push skipped (PUSH_TO_HUB is off or HF_TOKEN is missing). Merged weights are in", merged_dir)
    finally:
        tokenizer.chat_template = stock
    return status


def resolve_merged(config, stage: str, merged_dir: str, hf_user: str | None) -> str | None:
    """Merged weights for a stage: the local copy, else downloaded from the Hub, else None."""
    if os.path.exists(os.path.join(merged_dir, "config.json")):
        return merged_dir
    if hf_user and os.environ.get("HF_TOKEN"):
        try:
            from huggingface_hub import snapshot_download

            return snapshot_download(config.repo(hf_user, stage), local_dir=merged_dir, token=os.environ["HF_TOKEN"])
        except Exception as exc:  # noqa: BLE001 - repo missing or network error
            print(f"Could not download merged {stage} model: {exc}")
    return None


# ------------------------------------------------------------------- GPU
def gpu_free_fraction() -> float:
    import torch

    if not torch.cuda.is_available():
        return 0.0
    free, total = torch.cuda.mem_get_info()
    return free / total


def release_gpu(*objects) -> float:
    """Best-effort cleanup; returns the free fraction afterwards (a restart is the reliable way)."""
    import torch

    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return gpu_free_fraction()
