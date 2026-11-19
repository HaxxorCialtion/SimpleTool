"""Reuse 26B's exact cache only after proving tokenizer and template equality."""
import hashlib
import json
from pathlib import Path
import time
from transformers import AutoTokenizer
from prepare_data import SPECIAL_TOKENS, THOUGHT, write

ROOT = Path(__file__).resolve().parent

def main():
    configs = {n: json.loads((ROOT/'configs'/f'{n}.json').read_text()) for n in ['26b_a4b', '31b']}
    tokens = {n: AutoTokenizer.from_pretrained(c['model_dir'], local_files_only=True) for n,c in configs.items()}
    a, b = tokens.values()
    assert a.backend_tokenizer.to_str() == b.backend_tokenizer.to_str()
    assert a.chat_template == b.chat_template
    rendered = b.apply_chat_template([{'role':'system','content':'SYS'},{'role':'user','content':'USR'}],
                                    tokenize=False, add_generation_prompt=True, enable_thinking=False)
    assert rendered == '<bos><|turn>system\nSYS<turn|>\n<|turn>user\nUSR<turn|>\n<|turn>model\n' + THOUGHT
    source = ROOT/'data/26b_a4b'
    manifest = json.loads((source/'MANIFEST.json').read_text())
    assert manifest['valid'] and manifest['fraction'] == .05
    assert hashlib.sha256(b.backend_tokenizer.to_str().encode()).hexdigest() == manifest['base_backend_tokenizer_sha256']
    examples = json.loads((source/'prompt_examples.json').read_text())
    assert examples and all(x['prompt'].endswith(THOUGHT + '<' + x['id'].rsplit(':',1)[1] + '>') for x in examples)
    b.add_tokens(list(SPECIAL_TOKENS), special_tokens=True)
    existing = list(getattr(b,'additional_special_tokens',getattr(b,'extra_special_tokens',[])))
    b.add_special_tokens({'additional_special_tokens': list(dict.fromkeys([*map(str,existing),*SPECIAL_TOKENS]))})
    assert {t:b.convert_tokens_to_ids(t) for t in SPECIAL_TOKENS} == manifest['special_token_ids']
    cached = AutoTokenizer.from_pretrained(source/'tokenizer', local_files_only=True)
    assert b.backend_tokenizer.to_str() == cached.backend_tokenizer.to_str()
    assert b.chat_template == cached.chat_template
    destination = ROOT/'data/31b'
    destination.mkdir(exist_ok=True)
    assert not (destination/'MANIFEST.json').exists(), 'already prepared'
    for child in source.iterdir():
        if child.name == 'MANIFEST.json':
            continue
        (destination/child.name).symlink_to(child)
    result = {**manifest, 'model':'31b', 'base_model':configs['31b']['model_dir'], 'thought_suffix':THOUGHT,
              'independently_tokenized_for_this_model':False, 'reused_from_model':'26b_a4b',
              'reuse_verified':{'backend_equal':True, 'template_equal':True,
                                'extended_backend_equal':True, 'rendered_no_thinking':rendered}, 'time':time.time()}
    write(destination/'MANIFEST.json', result)
    write(ROOT/'receipts/DATA_READY_31b.json', result)
    print(json.dumps(result, ensure_ascii=False))

if __name__ == '__main__':
    main()
