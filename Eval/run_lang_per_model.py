"""Run lsm_analysis / sentiment_analysis / word_analysis per model with PSYCH_MODEL set,
producing model-tagged per-patient outputs.

Usage:
    python Eval/run_lang_per_model.py --models gpt-5-mini qwen/qwen3.8-27b z-ai/glm-5.3-flash
"""
import argparse, os, subprocess, sys

models_map = {
    'gpt-5-mini': ('Model_Runs/gpt-5-mini/Session_Metadata', 'Model_Runs/gpt-5-mini/Saved_Sessions', 'gpt-5-mini'),
    'qwen/qwen3.8-27b': ('Model_Runs/qwen-qwen3.8-27b/Session_Metadata', 'Model_Runs/qwen-qwen3.8-27b/Saved_Sessions', 'qwen-qwen3.8-27b'),
    'z-ai/glm-5.3-flash': ('Model_Runs/z-ai-glm-5.3-flash/Session_Metadata', 'Model_Runs/z-ai-glm-5.3-flash/Saved_Sessions', 'z-ai-glm-5.3-flash'),
}

SCRIPTS = ['lsm_analysis.py', 'sentiment_analysis.py', 'word_analysis.py']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='*', default=list(models_map.keys()))
    ap.add_argument('--scripts', nargs='*', default=SCRIPTS)
    args = ap.parse_args()

    for mid in args.models:
        if mid not in models_map:
            print(f'  WARNING: unknown model {mid}, skipping')
            continue
        meta, saved, tag = models_map[mid]
        env = dict(os.environ)
        env['PSYCH_META_DIR'] = meta
        env['PSYCH_SAVE_DIR'] = saved
        env['PSYCH_MODEL'] = tag
        print(f'\n===== {mid} (tag={tag}) =====')
        for script in args.scripts:
            print(f'  --- {script} ---')
            r = subprocess.run([sys.executable, f'Eval/{script}'], env=env, capture_output=True, text=True)
            print(r.stdout[-1200:])
            if r.returncode != 0:
                print('  [stderr]', r.stderr[-600:])


if __name__ == '__main__':
    main()