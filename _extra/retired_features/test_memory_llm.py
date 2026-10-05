import types
import unittest

from memory.llm import PoolMemoryLLM


class _Tokenizer:
    def apply_chat_template(self, messages, **_kwargs):
        return messages[0]["content"]


class _Pool:
    sequential = True

    def __init__(self, active=False, fail=False):
        self.active = active
        self.fail = fail
        self.releases = 0

    def iter_group_jobs(self, prompts_by_group, **_kwargs):
        self.active = True
        if self.fail:
            raise RuntimeError("generation failed")
        for index, prompt in enumerate(prompts_by_group):
            yield index, [(f"reply:{prompt}", [index])]

    def release(self):
        self.releases += 1
        self.active = False


class PoolMemoryLLMTests(unittest.TestCase):
    def _llm(self, pool):
        cfg = types.SimpleNamespace(
            max_new_tokens=128, temperature=0.7, top_p=0.95)
        return PoolMemoryLLM(cfg, pool, _Tokenizer())

    def test_phase_shared_pool_releases_after_memory_call(self):
        pool = _Pool(active=False)
        replies = self._llm(pool).complete_many([
            [{"role": "user", "content": "one"}],
            [{"role": "user", "content": "two"}],
        ])

        self.assertEqual(replies, ["reply:one", "reply:two"])
        self.assertEqual(pool.releases, 1)
        self.assertFalse(pool.active)

    def test_memory_call_releases_phase_shared_pool_after_failure(self):
        pool = _Pool(active=False, fail=True)
        with self.assertRaisesRegex(RuntimeError, "generation failed"):
            self._llm(pool).complete_many(
                [[{"role": "user", "content": "one"}]])

        self.assertEqual(pool.releases, 1)
        self.assertFalse(pool.active)

    def test_memory_call_does_not_release_pool_owned_by_outer_phase(self):
        pool = _Pool(active=True)
        self._llm(pool).complete_many(
            [[{"role": "user", "content": "one"}]])

        self.assertEqual(pool.releases, 0)
        self.assertTrue(pool.active)


if __name__ == "__main__":
    unittest.main()
