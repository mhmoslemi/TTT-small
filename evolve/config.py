"""
EVOLVE — configuration system.

Every tunable in the framework lives in exactly one place: the dataclass schema
below. Nothing downstream is allowed to hardcode a hyperparameter; components
receive their own config section and read from it.

------------------------------------------------------------------------------
RESOLUTION ORDER (highest priority wins)
------------------------------------------------------------------------------

    1. CLI flags              run.sh / command line      "cli"
    2. FRAMEWORK_OVERRIDES    this file, below           "config.py:override"
    3. example YAML           examples/<name>/config.yaml  "yaml:example"
    4. base YAML              configs/base.yaml          "yaml:base"
    5. dataclass defaults     this file, the schema      "config.py:default"

Layers 2 and 5 are both "config.py" but play different roles, and the split is
deliberate. The dataclass defaults are the *floor*: they define the schema and
supply a value when nobody else does. FRAMEWORK_OVERRIDES is the *ceiling below
the CLI*: an explicit dict of pins that outrank every YAML file. Without that
split, putting config.py above YAML would make every example's config.yaml
unreachable, since a dataclass field always has some value.

So: leave FRAMEWORK_OVERRIDES empty and YAML behaves normally. Add a key to it
and that value is locked for all examples until the CLI overrides it.

------------------------------------------------------------------------------
ADDRESSING
------------------------------------------------------------------------------

Every leaf has a dotted path: `search.alpha`, `model.lora_rank`,
`example.params.num_circles`. Three ways to set one, all equivalent:

    YAML, nested                YAML, dotted              CLI
    --------------              --------------            ---
    search:                     search.alpha: 0.5         --alpha 0.5
      alpha: 0.5                                          --set search.alpha=0.5

`example.params` is an open namespace: arbitrary keys are accepted there and
passed through to the example untouched. Every other path must exist in the
schema, so a typo raises instead of silently doing nothing.
"""

import argparse
import difflib
import json
import os
import sys
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, get_args, get_origin

import yaml

FRAMEWORK_ROOT = Path(__file__).resolve().parent
CONFIGS_DIR = FRAMEWORK_ROOT / "configs"
EXAMPLES_DIR = FRAMEWORK_ROOT / "examples"
BASE_YAML = CONFIGS_DIR / "base.yaml"


# ======================================================================
# Layer 2 — framework-level Python overrides (outrank every YAML file)
# ======================================================================
# Keys are dotted paths. Anything pinned here wins over base.yaml and over any
# example's config.yaml, and loses only to the CLI. Empty by default.
#
# Example:
#     FRAMEWORK_OVERRIDES = {
#         "elo.scale": 1.0,          # force the paper's un-scaled Elo everywhere
#         "rl.enabled": False,       # search-only ablation across all examples
#     }
FRAMEWORK_OVERRIDES: Dict[str, Any] = {}


# ======================================================================
# Schema
# ======================================================================
@dataclass
class ExampleConfig:
    """Which use-case to run. The framework knows nothing about its contents."""
    name: str = "circle_packing"
    # Dotted import path to the module exposing `build(cfg) -> Example`.
    # Empty -> examples.<name>.env
    module: str = ""
    # Free-form namespace handed to the example verbatim (num_circles, target,
    # budget_s, ...). Open: unknown keys are allowed here and only here.
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    """The single LLM backbone. Generator, Elo judge and memory maker share it."""
    name: str = "Qwen/Qwen3-8B"
    backend: str = "auto"                 # auto | unsloth | hf
    max_seq_length: int = 32000
    load_in_4bit: bool = False
    # LoRA — the only trainable parameters (paper §2.3).
    lora_rank: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    # Qwen3-family chat templates accept enable_thinking and default it to True,
    # so the model opens a <think> block and can spend the entire token budget
    # in it without ever emitting an answer. The reference implementation passes
    # False, and so do we: the prompt already asks for a bounded <strategy>
    # block, and two layers of reasoning is what exhausts the budget. Ignored by
    # templates that do not accept the argument.
    enable_thinking: bool = False
    target_modules: Tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )


@dataclass
class GenerationConfig:
    """Rollout sampling. B_t in [n, n*k] responses per step."""
    max_new_tokens: int = 4000
    temperature: float = 1.0
    top_p: float = 1.0
    # Multi-GPU generation pool. 1 -> in-process generation, no workers.
    num_gpus: int = 1
    gpu_ids: str = ""                     # "" -> 0..num_gpus-1
    # Rollouts per generate() call. Decoding re-reads the whole weight matrix
    # each step regardless of batch size, so a larger batch is close to free
    # throughput until the KV cache fills the card. 0 = the whole step's B_t in
    # one call. On OOM the batch is halved and retried, so this is a hint.
    batch_size: int = 0
    # --- thinking budget -------------------------------------------------
    # A reasoning model will happily spend the entire budget inside its think
    # block and never emit an answer, which scores zero however good the
    # reasoning was. With think_budget > 0 generation runs in two phases:
    # think for at most this many tokens, then force the block closed and spend
    # what is left writing the answer. Reasoning is capped, not removed.
    # 0 disables it. A good starting point is ~60% of max_new_tokens.
    think_budget: int = 0
    think_close_tag: str = "</think>"
    # Stop a sequence as soon as it has produced a complete ```python block.
    # Without it a model that finishes the program keeps going -- reopening
    # <strategy>, writing a second worse program, and burning the rest of the
    # budget. Costs one decode per check, hence stop_check_every.
    stop_on_code_block: bool = True
    stop_check_every: int = 16
    # Injected when the budget runs out, to hand the model a running start on
    # the answer rather than dropping it at the closing tag.
    think_force_text: str = (
        "\n</think>\n\nI have analysed enough. Final program:\n")


@dataclass
class SearchConfig:
    """§2.1 — D-PUCT over the node dataset."""
    n_select: int = 8                     # n   top-n actions per step
    k_children: int = 8                   # k   children per leaf expansion
    c_puct: float = 1.0                   # c   exploration strength, Eq. 6
    alpha: float = 0.5                    # α   rank vs Elo mix, Eq. 3
    lambda_virtual: float = 1.0           # λ   virtual-child optimism, Eq. 4
    tau: float = 1.0                      # τ   prior softmax temperature, Eq. 5
    # V(p,a) = W_m(a) is in raw reward units while the prior is a probability,
    # so c does not transfer across problems. Rescaling W_m by the archive
    # spread makes c problem-independent. Set false for the literal Eq. 6.
    normalize_exploitation: bool = True
    # "node": one generation target per node -- a leaf expands into k children,
    #   a node that already has children offers its virtual action for 1 child.
    #   This is the literal "select top-n nodes by Eq. 6", and an internal node
    #   is never itself a target, which removes the paper's undefined case.
    # "action_descend": score (parent, action) pairs and walk down through a
    #   chosen internal child until reaching a leaf or a virtual action.
    selection_mode: str = "node"          # node | action_descend
    # V(p, s-hat) for the virtual action. The paper leaves it undefined and says
    # the score is "driven by the prior and the exploration bonus" -> zero.
    virtual_value_mode: str = "zero"      # zero | parent_mean
    max_archive_size: int = 1000          # cap on |D|; 0 = unbounded
    num_seed_nodes: int = 1               # root(s) at step 0


@dataclass
class EloConfig:
    """§2.1 — Elo debate signal feeding the global node logit."""
    enabled: bool = True
    k_factor: float = 24.0                # K   update rate
    initial_rating: float = 0.0           # E_0 (identical for all nodes)
    # Logistic scale in p_ij = 1 / (1 + 10^((E_j - E_i) / scale)).
    # The paper writes 10^(E_j - E_i), i.e. scale = 1.0. Classic Elo is 400.0.
    # Kept configurable because it changes K's meaning by ~400x.
    scale: float = 400.0
    # round_robin: every pair among the candidates
    # random:      up to num_matches distinct pairs
    # neighbors:   sort candidates by W_m and compare adjacent pairs (k-1 matches)
    pairing_mode: str = "round_robin"
    num_matches: int = 60                 # cap for pairing_mode=random
    candidate_top_k: int = 16             # only rate the top-k nodes of D
    rounds_per_step: int = 1              # repeat the pairing schedule this often
    allow_ties: bool = True
    judge_max_tokens: int = 1024
    judge_temperature: float = 0.7
    judge_batch_size: int = 8


@dataclass
class MemoryConfig:
    """§2.2 — experience memory: 2L lessons per step, top-m retrieved."""
    enabled: bool = True
    lessons_per_group: int = 3            # L  -> 2L lessons per step
    top_m: int = 5                        # m  retrieved per parent, Eq. 7
    embedding_backend: str = "hash"       # hash | sentence_transformers | backbone
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 512              # used by the dependency-free hash backend
    max_bank_size: int = 0                # 0 = unbounded (paper: purely additive)
    extractor_max_tokens: int = 2048
    extractor_temperature: float = 0.7
    # How much of each rollout to show the extractor, in characters.
    max_chars_per_example: int = 4000


@dataclass
class RLConfig:
    """§2.3 — test-time RL on LoRA only."""
    enabled: bool = True
    beta: float = 1.0                     # β    reward tilt, Eq. 8 (0 => zero advantage)
    lambda_feedback: float = 0.5          # λ_f  weight on the failure signal
    clip_epsilon: float = 0.2             # ε    Eq. 11
    kl_coef: float = 0.1                  # η_KL regularization toward θ_0
    learning_rate: float = 1e-6
    grad_clip: float = 1.0
    updates_per_step: int = 1             # paper: one update per evolution step
    microbatch_size: int = 1
    # Eq. 9 needs two extra forward passes per failed rollout. Disable to train
    # on the group-relative reward signal alone.
    use_feedback_signal: bool = True
    # Guard rail, not part of the paper: A_fb is a log-prob difference and is
    # unbounded below, so one token the feedback-conditioned teacher considers
    # near-impossible can dominate the batch. 0 disables the clamp.
    advantage_clip: float = 10.0
    # Skip the update when every response in the batch scored identically --
    # Eq. 8 yields all-zero advantages there, so the step would be a no-op that
    # still costs a full backward pass.
    skip_degenerate_batches: bool = True


@dataclass
class VerifierConfig:
    """Environment verification — the reward r and textual feedback f."""
    timeout_s: float = 100.0
    max_cpus: int = 1
    # Feedback f is truncated to this many characters before entering a prompt.
    feedback_max_chars: int = 2000
    fail_reward: float = 0.0              # invalid candidates receive zero reward


@dataclass
class RunConfig:
    max_steps: int = 50                   # T_max
    seed: int = 42
    output_root: str = "runs"
    run_name: str = ""                    # "" -> auto-generated from the config
    save_rollouts: bool = True
    print_responses: int = 0              # how many rollouts to echo per step
    resume_from: str = ""
    # Per-phase progress bars. A step is otherwise silent through one blocking
    # generate() per target and a serial loop of sandbox subprocesses, which
    # looks identical to being hung.
    progress: bool = True


@dataclass
class Config:
    example: ExampleConfig = field(default_factory=ExampleConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    elo: EloConfig = field(default_factory=EloConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    verifier: VerifierConfig = field(default_factory=VerifierConfig)
    run: RunConfig = field(default_factory=RunConfig)

    def to_dict(self) -> Dict[str, Any]:
        return _to_plain(self)

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, default=str))


# ======================================================================
# CLI surface — declarative, so adding a flag is one line
# ======================================================================
def _optbool(s: str) -> bool:
    v = str(s).strip().lower()
    if v in ("1", "true", "t", "yes", "y", "on"):
        return True
    if v in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {s!r}")


# (flag, dotted path, type, help)
_CLI_SPEC: List[Tuple[str, str, Any, str]] = [
    # -- model / backend --
    ("--model-name",        "model.name",              str,      "HF model id for the backbone"),
    ("--backend",           "model.backend",           str,      "auto | unsloth | hf"),
    ("--max-seq-length",    "model.max_seq_length",    int,      "context length"),
    ("--load-in-4bit",      "model.load_in_4bit",      _optbool, "quantize the backbone"),
    ("--lora-rank",         "model.lora_rank",         int,      "LoRA rank"),
    ("--lora-alpha",        "model.lora_alpha",        int,      "LoRA alpha"),
    ("--lora-dropout",      "model.lora_dropout",      float,    "LoRA dropout"),
    # -- generation --
    ("--max-new-tokens",    "generation.max_new_tokens", int,    "tokens per rollout"),
    ("--temperature",       "generation.temperature",  float,    "sampling temperature"),
    ("--top-p",             "generation.top_p",        float,    "nucleus sampling"),
    ("--num-gpus",          "generation.num_gpus",     int,      "generation workers (1 = in-process)"),
    ("--gpu-ids",           "generation.gpu_ids",      str,      "e.g. '6,7'"),
    ("--gen-batch-size",    "generation.batch_size",   int,      "rollouts per generate() call; 0 = all of B_t"),
    ("--think-budget",      "generation.think_budget", int,      "cap tokens spent inside <think>; 0 = uncapped"),
    ("--enable-thinking",   "model.enable_thinking",   _optbool, "Qwen-style native <think> mode"),
    ("--stop-on-code",      "generation.stop_on_code_block", _optbool, "stop once a complete ```python block exists"),
    # -- D-PUCT (§2.1) --
    ("--n-select",          "search.n_select",         int,      "n: top-n actions per step"),
    ("--k-children",        "search.k_children",       int,      "k: children per leaf expansion"),
    ("--c-puct",            "search.c_puct",           float,    "c: exploration strength (Eq. 6)"),
    ("--alpha",             "search.alpha",            float,    "alpha: rank vs Elo mix (Eq. 3)"),
    ("--lambda-virtual",    "search.lambda_virtual",   float,    "lambda: virtual-child optimism (Eq. 4)"),
    ("--tau",               "search.tau",              float,    "tau: prior softmax temperature (Eq. 5)"),
    ("--max-archive-size",  "search.max_archive_size", int,      "cap on |D|; 0 = unbounded"),
    # -- Elo debate --
    ("--elo-enabled",       "elo.enabled",             _optbool, "enable the Elo debate signal"),
    ("--elo-k",             "elo.k_factor",            float,    "K: Elo update rate"),
    ("--elo-scale",         "elo.scale",               float,    "logistic scale (paper: 1.0, classic: 400.0)"),
    ("--elo-matches",       "elo.num_matches",         int,      "matches per step when pairing=random"),
    ("--elo-top-k",         "elo.candidate_top_k",     int,      "how many nodes of D to rate"),
    # -- memory (§2.2) --
    ("--memory-enabled",    "memory.enabled",          _optbool, "enable the memory module"),
    ("--lessons-per-group", "memory.lessons_per_group", int,     "L: lessons per group (2L per step)"),
    ("--top-m",             "memory.top_m",            int,      "m: lessons retrieved per parent"),
    # -- test-time RL (§2.3) --
    ("--rl-enabled",        "rl.enabled",              _optbool, "enable test-time RL"),
    ("--beta",              "rl.beta",                 float,    "beta: reward tilt (Eq. 8)"),
    ("--lambda-feedback",   "rl.lambda_feedback",      float,    "lambda_f: failure-signal weight"),
    ("--clip-epsilon",      "rl.clip_epsilon",         float,    "epsilon: PPO clip (Eq. 11)"),
    ("--kl-coef",           "rl.kl_coef",              float,    "eta_KL: KL toward theta_0"),
    ("--lr",                "rl.learning_rate",        float,    "LoRA learning rate"),
    ("--grad-clip",         "rl.grad_clip",            float,    "gradient clipping"),
    # -- verifier --
    ("--verifier-timeout",  "verifier.timeout_s",      float,    "sandbox timeout in seconds"),
    # -- run --
    ("--steps",             "run.max_steps",           int,      "T_max: evolution steps"),
    ("--seed",              "run.seed",                int,      "global RNG seed"),
    ("--output-root",       "run.output_root",         str,      "root directory for runs/"),
    ("--run-name",          "run.run_name",            str,      "explicit run directory name"),
    ("--print-responses",   "run.print_responses",     int,      "rollouts to echo per step"),
    ("--progress",          "run.progress",            _optbool, "per-phase progress bars"),
    ("--resume-from",       "run.resume_from",         str,      "run directory to resume"),
]


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="evolve",
        description="EVOLVE — memory-augmented D-PUCT search with test-time RL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Structural: these choose *which* YAML layers load, they are not overrides.
    p.add_argument("--example", default=None,
                   help="Example name; loads examples/<name>/config.yaml.")
    p.add_argument("--config", action="append", default=None, metavar="PATH",
                   help="Extra YAML file, applied at the example layer. Repeatable; "
                        "later files win.")
    p.add_argument("--no-base-yaml", action="store_true",
                   help="Skip configs/base.yaml.")

    # Generic escape hatch — reaches any leaf, including example.params.*
    p.add_argument("--set", action="append", default=None, metavar="PATH=VALUE",
                   help="Override any dotted path, e.g. --set example.params.num_circles=32. "
                        "Repeatable; applied after the named flags.")

    # Introspection
    p.add_argument("--print-config", action="store_true",
                   help="Print the resolved config with per-key provenance, then exit.")
    p.add_argument("--print-config-json", action="store_true",
                   help="Print the resolved config as JSON, then exit.")
    p.add_argument("--list-examples", action="store_true",
                   help="List available examples, then exit.")

    for flag, path, typ, helptext in _CLI_SPEC:
        p.add_argument(flag, dest=_flag_to_dest(flag), type=typ, default=None,
                       metavar=path.split(".")[-1].upper(), help=f"[{path}] {helptext}")
    return p


def _flag_to_dest(flag: str) -> str:
    return flag.lstrip("-").replace("-", "_")


# ======================================================================
# Schema introspection
# ======================================================================
def _is_open_dict(typ: Any) -> bool:
    """True for Dict[...]-typed fields, whose children are free-form."""
    return get_origin(typ) is dict


def _unwrap_optional(typ: Any) -> Any:
    if get_origin(typ) is not None and type(None) in get_args(typ):
        real = [a for a in get_args(typ) if a is not type(None)]
        if len(real) == 1:
            return real[0]
    return typ


def _walk_schema(obj: Any, prefix: str,
                 values: Dict[str, Any], types: Dict[str, Any],
                 open_prefixes: List[str]) -> None:
    for f in fields(obj):
        path = f"{prefix}{f.name}"
        value = getattr(obj, f.name)
        ftype = _unwrap_optional(f.type)
        if is_dataclass(ftype):
            _walk_schema(value, path + ".", values, types, open_prefixes)
        elif _is_open_dict(ftype):
            open_prefixes.append(path + ".")
            for k, v in (value or {}).items():
                values[f"{path}.{k}"] = v
                types[f"{path}.{k}"] = Any
        else:
            values[path] = value
            types[path] = ftype


def schema() -> Tuple[Dict[str, Any], Dict[str, Any], List[str]]:
    """(default values, leaf types, open dict prefixes) keyed by dotted path."""
    values: Dict[str, Any] = {}
    types: Dict[str, Any] = {}
    open_prefixes: List[str] = []
    _walk_schema(Config(), "", values, types, open_prefixes)
    return values, types, open_prefixes


# ======================================================================
# Coercion
# ======================================================================
def _auto_scalar(s: str) -> Any:
    """Best-effort typing for values headed into an open namespace."""
    t = s.strip()
    low = t.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null", ""):
        return None
    for cast in (int, float):
        try:
            return cast(t)
        except ValueError:
            pass
    if t[:1] in "[{\"":
        try:
            return json.loads(t)
        except (ValueError, TypeError):
            pass
    return t


def _coerce(value: Any, typ: Any, path: str) -> Any:
    if value is None or typ is Any:
        return _auto_scalar(value) if isinstance(value, str) and typ is Any else value

    origin = get_origin(typ)
    try:
        if origin in (tuple, list) or typ in (tuple, list):
            if isinstance(value, str):
                items = [x.strip() for x in value.split(",") if x.strip()]
            else:
                items = list(value)
            args = get_args(typ)
            if args and args[0] is not Ellipsis:
                items = [_coerce(x, args[0], path) for x in items]
            return tuple(items) if origin is tuple or typ is tuple else items
        if origin is dict or typ is dict:
            return dict(value)
        if typ is bool:
            return _optbool(value) if isinstance(value, str) else bool(value)
        if typ is int:
            return int(value)
        if typ is float:
            return float(value)
        if typ is str:
            return str(value)
    except (ValueError, TypeError, argparse.ArgumentTypeError) as e:
        raise ConfigError(f"{path}: cannot read {value!r} as {typ}: {e}") from e
    return value


# ======================================================================
# Resolution
# ======================================================================
class ConfigError(Exception):
    pass


LAYER_DEFAULT = "config.py:default"
LAYER_OVERRIDE = "config.py:override"
LAYER_CLI = "cli"


@dataclass
class Resolution:
    """The resolved config plus where every value came from."""
    config: Config
    values: Dict[str, Any]
    origins: Dict[str, str]
    yaml_files: List[str]

    def explain(self, changed_only: bool = False) -> str:
        width = max((len(k) for k in self.values), default=10)
        lines = [
            "resolved configuration  (cli > config.py:override > yaml:example "
            "> yaml:base > config.py:default)",
            "-" * (width + 46),
        ]
        for path in sorted(self.values):
            origin = self.origins.get(path, LAYER_DEFAULT)
            if changed_only and origin == LAYER_DEFAULT:
                continue
            lines.append(f"  {path:<{width}}  {str(self.values[path]):<22}  {origin}")
        if self.yaml_files:
            lines += ["-" * (width + 46), "  yaml layers applied:"]
            lines += [f"    - {f}" for f in self.yaml_files]
        return "\n".join(lines)


def _flatten_mapping(node: Any, prefix: str = "") -> Dict[str, Any]:
    """Flatten nested YAML into dotted keys. Already-dotted keys pass through."""
    out: Dict[str, Any] = {}
    if not isinstance(node, dict):
        return {prefix.rstrip("."): node} if prefix else {}
    for k, v in node.items():
        path = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten_mapping(v, path + "."))
        else:
            out[path] = v
    return out


def _apply_layer(layer: Dict[str, Any], origin: str,
                 values: Dict[str, Any], origins: Dict[str, str],
                 types: Dict[str, Any], open_prefixes: List[str]) -> None:
    for path, raw in layer.items():
        if path in types:
            typ = types[path]
        elif any(path.startswith(p) for p in open_prefixes):
            typ = Any                     # open namespace: accept and auto-type
        else:
            hint = difflib.get_close_matches(path, list(types), n=1)
            suffix = f" Did you mean {hint[0]!r}?" if hint else ""
            raise ConfigError(
                f"unknown config key {path!r} (from {origin}).{suffix}\n"
                f"Free-form keys are only allowed under: "
                f"{', '.join(p + '*' for p in open_prefixes)}"
            )
        values[path] = _coerce(raw, typ, path)
        origins[path] = origin


def _unflatten(values: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for path, v in values.items():
        parts = path.split(".")
        node = out
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = v
    return out


def _build(cls: Any, data: Dict[str, Any]) -> Any:
    kwargs: Dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        ftype = _unwrap_optional(f.type)
        value = data[f.name]
        kwargs[f.name] = _build(ftype, value) if is_dataclass(ftype) else value
    return cls(**kwargs)


def _to_plain(obj: Any) -> Any:
    if is_dataclass(obj):
        return {f.name: _to_plain(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    return obj


def _read_yaml(path: Path) -> Dict[str, Any]:
    with open(path) as fh:
        loaded = yaml.safe_load(fh) or {}
    if not isinstance(loaded, dict):
        raise ConfigError(f"{path}: top level must be a mapping, got {type(loaded).__name__}")
    return loaded


def example_config_path(name: str) -> Path:
    return EXAMPLES_DIR / name / "config.yaml"


def list_examples() -> List[str]:
    if not EXAMPLES_DIR.is_dir():
        return []
    return sorted(d.name for d in EXAMPLES_DIR.iterdir()
                  if d.is_dir() and not d.name.startswith("_")
                  and (d / "config.yaml").exists())


def load_config(argv: Optional[List[str]] = None) -> Resolution:
    """Resolve the full configuration. See the module docstring for the order."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.list_examples:
        found = list_examples()
        print("\n".join(found) if found else f"(no examples under {EXAMPLES_DIR})")
        sys.exit(0)

    values, types, open_prefixes = schema()
    origins = {path: LAYER_DEFAULT for path in values}
    yaml_files: List[str] = []

    def apply(layer: Dict[str, Any], origin: str) -> None:
        _apply_layer(layer, origin, values, origins, types, open_prefixes)

    # ---- layer 4: base YAML ------------------------------------------
    if not args.no_base_yaml and BASE_YAML.exists():
        apply(_flatten_mapping(_read_yaml(BASE_YAML)), "yaml:base")
        yaml_files.append(str(BASE_YAML))

    # ---- layer 3: example YAML ---------------------------------------
    # --example selects it; base.yaml's `example.name` is the fallback, so a
    # bare `python main.py` still resolves to a complete example.
    example_name = args.example or values.get("example.name")
    if example_name:
        values["example.name"] = example_name
        if args.example:
            origins["example.name"] = LAYER_CLI
        path = example_config_path(example_name)
        if path.exists():
            apply(_flatten_mapping(_read_yaml(path)), "yaml:example")
            yaml_files.append(str(path))
        elif args.example:
            raise ConfigError(
                f"no config for example {example_name!r} at {path}. "
                f"Available: {', '.join(list_examples()) or '(none)'}"
            )

    for extra in (args.config or []):
        path = Path(extra)
        if not path.exists():
            raise ConfigError(f"--config path not found: {path}")
        apply(_flatten_mapping(_read_yaml(path)), f"yaml:{path.name}")
        yaml_files.append(str(path))

    # `--example` must survive an example YAML that names something else.
    if args.example:
        values["example.name"] = args.example
        origins["example.name"] = LAYER_CLI

    # ---- layer 2: framework overrides from this file -----------------
    if FRAMEWORK_OVERRIDES:
        apply(dict(FRAMEWORK_OVERRIDES), LAYER_OVERRIDE)

    # ---- layer 1: CLI ------------------------------------------------
    cli_layer: Dict[str, Any] = {}
    for flag, path, _typ, _help in _CLI_SPEC:
        given = getattr(args, _flag_to_dest(flag), None)
        if given is not None:
            cli_layer[path] = given
    apply(cli_layer, LAYER_CLI)

    # --set last, so it wins ties inside the CLI layer
    set_layer: Dict[str, Any] = {}
    for item in (args.set or []):
        if "=" not in item:
            raise ConfigError(f"--set expects PATH=VALUE, got {item!r}")
        path, _, raw = item.partition("=")
        set_layer[path.strip()] = raw
    apply(set_layer, LAYER_CLI)

    cfg = _build(Config, _unflatten(values))
    resolution = Resolution(config=cfg, values=values, origins=origins,
                            yaml_files=yaml_files)

    if args.print_config:
        print(resolution.explain())
        sys.exit(0)
    if args.print_config_json:
        print(json.dumps(cfg.to_dict(), indent=2, default=str))
        sys.exit(0)
    return resolution


if __name__ == "__main__":
    print(load_config().explain())
