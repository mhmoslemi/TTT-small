"""Worker-side RAM accounting for vLLM's normal level-1 sleep/wake path."""


class SleepMemoryExtension:
    def ttt_host_memory(self):
        import torch
        from vllm.device_allocator.cumem import CuMemAllocator
        from vllm.device_allocator import cumem
        from runtime.host_memory import reclaim_unused_host_memory

        # A 70-GiB model per engine must not become eight permanently pinned
        # host copies. vLLM uses synchronous cudaMemcpy, which also supports
        # pageable destinations. Keep its exact allocation/copy/restore logic;
        # change only the host allocation type in the sleep allocator.
        cumem.PIN_MEMORY = False
        cleanup = reclaim_unused_host_memory(
            torch, cuda_devices=(int(torch.cuda.current_device()),), force=True)
        allocator = CuMemAllocator.get_instance()
        backup_bytes = sum(
            int(data.handle[1]) for data in allocator.pointer_to_data.values()
            if data.tag == "weights")
        return {"backup_bytes": backup_bytes,
                "rss_bytes": cleanup["rss_after"]}
