"""A transient GLiDE process; the training process never imports its runtime."""

from __future__ import annotations

from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import select
import subprocess
import sys


MAX_CONTEXT = 40000


class GlideWorker:
    def __init__(self, model_path, gpu_id, log_path, *, before_start=None,
                 after_stop=None):
        self.model_path = str(model_path)
        self.gpu_id = int(gpu_id)
        self.log_path = Path(log_path)
        self.before_start, self.after_stop = before_start, after_stop
        self.process = self.log = None
        self._prepared = False

    def __enter__(self):
        path = Path(self.model_path).expanduser()
        for name in ("glide/__init__.py", "decision_config.json",
                     "readout.safetensors", "model.safetensors.index.json"):
            if not (path / name).is_file():
                raise FileNotFoundError(
                    f"puct-jev needs the complete local fastino/GLiDE-9B "
                    f"repository; missing {path / name}. Set --jev-model-path.")
        try:
            if self.before_start:
                self.before_start()
            self._prepared = True
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log = self.log_path.open("a", encoding="utf-8")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(self.gpu_id),
                       CUDA_DEVICE_ORDER="PCI_BUS_ID", TOKENIZERS_PARALLELISM="false")
            self.process = subprocess.Popen(
                [sys.executable, "-m", "search.jev_worker", str(path.resolve())],
                cwd=str(Path(__file__).resolve().parents[1]), env=env,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
                text=True, encoding="utf-8", bufsize=1)
            ready = self._receive()
            if ready != {"ready": True}:
                raise RuntimeError(f"invalid GLiDE startup response: {ready!r}")
            return self
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise

    def _receive(self):
        if not select.select([self.process.stdout], [], [], 600)[0]:
            raise TimeoutError(f"GLiDE timed out; see {self.log_path}")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"GLiDE worker exited; see {self.log_path}")
        try:
            response = json.loads(line)
        except ValueError as error:
            raise RuntimeError(f"invalid GLiDE response; see {self.log_path}") from error
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"GLiDE: {response['error']}; see {self.log_path}")
        return response

    def compare(self, request):
        self.process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
        self.process.stdin.flush()
        return self._receive()

    def __exit__(self, exc_type, exc, tb):
        try:
            if self.process is not None:
                process, self.process = self.process, None
                try:
                    if process.poll() is None:
                        if exc_type is None:
                            try:
                                process.stdin.write('{"stop": true}\n')
                                process.stdin.flush()
                            except (BrokenPipeError, OSError):
                                pass
                        else:
                            process.terminate()
                        try:
                            process.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                finally:
                    process.stdin.close()
                    process.stdout.close()
        finally:
            if self.log is not None:
                self.log.close()
                self.log = None
            if self._prepared:
                self._prepared = False
                if self.after_stop:
                    self.after_stop()


def _decide(model, request):
    from glide.prompt import options_for

    question = request["questions"]["relationship"]
    prompt = model.prompter.text(request["state"], question, options_for(question))
    tokens = len(model.tokenizer(prompt, add_special_tokens=False)["input_ids"])
    if tokens + 8 > MAX_CONTEXT:
        raise ValueError(
            f"comparison uses {tokens} tokens plus GLiDE's 8-token reserve; "
            f"limit is {MAX_CONTEXT}. Both programs were kept intact.")
    return model.decide(state=request["state"], questions=request["questions"])


def main():
    protocol = sys.stdout
    try:
        # Dependency and checkpoint messages must not enter the JSON protocol.
        with redirect_stdout(sys.stderr):
            sys.path.insert(0, sys.argv[1])
            import torch
            from glide import Glide

            if not torch.cuda.is_available():
                raise RuntimeError("GLiDE-9B requires a CUDA GPU")
            model = Glide.from_pretrained(
                sys.argv[1], device="cuda:0", dtype=torch.bfloat16,
                vision=False, max_len=MAX_CONTEXT)
        print(json.dumps({"ready": True}), file=protocol, flush=True)
        for line in sys.stdin:
            request = json.loads(line)
            if request.get("stop"):
                break
            with redirect_stdout(sys.stderr), torch.inference_mode():
                result = _decide(model, request)
            print(json.dumps(result, allow_nan=False), file=protocol, flush=True)
    except Exception as error:
        print(json.dumps({"error": f"{type(error).__name__}: {error}"}),
              file=protocol, flush=True)
        raise


if __name__ == "__main__":
    main()
