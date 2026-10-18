"""vLLM GPU worker with bounded GPT-OSS weight-conversion scratch."""

from vllm.v1.worker.gpu_worker import Worker

from vllm_compat.gpt_oss_repack import bounded_marlin_repacking


class GptOssWorker(Worker):
    def load_model(self, *args, **kwargs):
        print("[vllm] GPT-OSS bounded Marlin repacking: one expert at a time, "
              "reusing checkpoint storage; native kernel and sleep/wake retained",
              flush=True)
        with bounded_marlin_repacking():
            return super().load_model(*args, **kwargs)
