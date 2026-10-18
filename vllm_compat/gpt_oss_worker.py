"""vLLM GPU worker with reusable GPT-OSS weight-conversion scratch."""

from vllm.v1.worker.gpu_worker import Worker

from vllm_compat.gpt_oss_repack import bounded_marlin_repacking


class GptOssWorker(Worker):
    def _scoped_allocator_max_split(self, max_split_size_mb):
        # vLLM 0.28 requests 20 MiB during loading. PyTorch also allocates
        # 20 MiB segments for the 4/8 MiB per-expert conversion tensors. Once
        # freed/coalesced, those segments are >= that limit and cannot serve
        # another small request. The active sleep pool cannot release them,
        # so even in-place conversion eventually exhausts the device.
        # Allow reuse of conversion/scale scratch while leaving GiB-sized
        # checkpoint allocations unsplit. The parent restores the previous
        # allocator setting on both success and failure.
        return super()._scoped_allocator_max_split(
            max_split_size_mb=max(512, max_split_size_mb))

    def load_model(self, *args, **kwargs):
        print("[vllm] GPT-OSS bounded Marlin repacking: one expert at a time, "
              "reusing checkpoint storage and allocator scratch "
              "(loading split limit >=512 MiB); "
              "native kernel and sleep/wake retained",
              flush=True)
        with bounded_marlin_repacking():
            return super().load_model(*args, **kwargs)
