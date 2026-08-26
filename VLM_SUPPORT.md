# Vision-language model support

The runner supports text LLMs and image-conditioned VLMs through the same
search, reward, memory, feedback, and LoRA update loop. The original LLM path
remains the default.

Use a decoder-style image-to-text model supported by the installed Unsloth and
Transformers versions. The policy update assumes causal response tokens, so an
encoder-decoder captioning model is not a compatible replacement. Install a
current `unsloth`, `transformers`, `peft`, and `Pillow` in the GPU environment.

## Quick start

Use an absolute or repository-relative local image path:

```bash
python3 train_multy.py \
  --problem circle_packing \
  --vlm \
  --backend unsloth \
  --model-name unsloth/Qwen3-VL-8B-Instruct \
  --vision-image /path/to/observation.png \
  --num-gpus 1
```

Repeat `--vision-image` to attach multiple images. `--backend auto` tries
Unsloth first and falls back to the Hugging Face image-to-text model plus PEFT.
The generation pool supports the same VLM prompt payload when `--num-gpus` is
greater than one.

Equivalent YAML fields are:

```yaml
model_kind: vlm
model_name: unsloth/Qwen3-VL-8B-Instruct
vision_images:
  - /path/to/observation.png
vlm_finetune_vision_layers: false
```

The vision tower is frozen by default. The LoRA policy still learns how to use
the pretrained visual representation through its language attention and MLP
layers. Use `--vlm-finetune-vision-layers` only when perception itself must
adapt; it increases trainable parameters and memory use.

## State-dependent scientific images

Static `vision_images` are useful for one fixed diagram or observation. A
problem whose image changes with the search state should override the problem
hook:

```python
class MyVisualProblem(Problem):
    def vision_inputs(self, parent: ParentContext) -> list[str]:
        image_path = self.render_parent_observation(parent)
        return [str(image_path)]
```

The returned files must remain available for the whole step. Paths are used
instead of PIL objects because every generation GPU runs in a spawned worker
process and reopens the same image.

For VLM runs, each rollout records its image paths in `vision_images` inside
the `.meta.json` file. The rendered text (including the model's image marker)
continues to be saved in `.prompt.txt`.

## What is image-conditioned

The same processor output is used for:

- rollout generation;
- the trainable-policy and adapter-disabled reference log-probabilities;
- feedback-conditioned teacher log-probabilities.

Memory extraction and lookup are textual meta-reasoning calls on the same VLM
backbone. They do not automatically inherit the rollout image; the image is
kept on the scientific rollout and its feedback reprompt, where token-level
credit must remain aligned.

Only local image files are accepted. This keeps deterministic/offline worker
runs independent of network availability.
