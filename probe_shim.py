import torch, sys
sys.path.insert(0, '.')
from src.models import SpatialAttnVanillaModel, patch_attention_layers_spatial
from transformers import AutoConfig, AutoProcessor
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
from peft import PeftModel

base = 'checkpoints/Qwen3.5-4B'
ckpt = 'train_records/atten_vst/step_3000'
config = AutoConfig.from_pretrained(base, trust_remote_code=True)
spa = Qwen3_5ForConditionalGeneration.from_pretrained(
    base, config=config, torch_dtype=torch.bfloat16, attn_implementation='sdpa')
new_inner = SpatialAttnVanillaModel(spa.config)
new_inner.load_state_dict(spa.model.state_dict(), strict=True)
spa.model = new_inner.to(dtype=torch.bfloat16)
spa.tie_weights()
patch_attention_layers_spatial(spa)
spa = PeftModel.from_pretrained(spa, ckpt, is_trainable=False)
spa = spa.merge_and_unload()
print("[probe] model loaded")

_orig = spa.model.forward
lm = spa.model.language_model
call_n = [0]

def _shim(*args, image_xyz=None, mm_token_type_ids=None, **kw):
    call_n[0] += 1
    n = call_n[0]
    before = lm._spatial_cache is not None
    if image_xyz is None:
        image_xyz = getattr(spa.model, '_eval_image_xyz', None)
    ret = _orig(*args, image_xyz=image_xyz, mm_token_type_ids=mm_token_type_ids, **kw)
    after = lm._spatial_cache is not None
    print(f"  call#{n}: xyz={image_xyz is not None}, mm={mm_token_type_ids is not None}  spatial_cache: {before}->{after}")
    return ret

spa.model.forward = _shim

from PIL import Image
import numpy as np
proc = AutoProcessor.from_pretrained(base, trust_remote_code=True)
img = Image.fromarray(np.zeros((112, 112, 3), dtype=np.uint8))
msgs = [{'role': 'user', 'content': [{'type': 'image', 'image': img}, {'type': 'text', 'text': 'A or B?'}]}]
text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
from qwen_vl_utils import process_vision_info
imgs, _ = process_vision_info(msgs)
inputs = proc(text=[text], images=imgs, return_tensors='pt')
print(f"[probe] inputs keys: {list(inputs.keys())}")
print(f"[probe] mm_token_type_ids sum: {inputs['mm_token_type_ids'].sum().item()}")

spa.model._eval_image_xyz = [torch.zeros(7, 7, 3)]
print("[probe] running generate ...")
out = spa.generate(
    **{k: v for k, v in inputs.items()},
    max_new_tokens=1, do_sample=False,
    pad_token_id=proc.tokenizer.eos_token_id,
)
print(f"[probe] done, new tokens={out.shape[1] - inputs['input_ids'].shape[1]}")
