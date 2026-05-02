"""Validate the 114B-for-90% capacity ceiling against Together-API capacity-curve data.

The Colab logistic fit on Qwen2.5 0.5B→14B predicted:
  - Application asymptote ≈ 100% (capped)
  - 90% reached at ~114B
  - 99% reached at ~1600B

We have data on 4 larger models:
  - Llama-3.3-70B (70B)
  - Qwen3-235B-A22B (235B total, 22B active)
  - DeepSeek-V3 (671B total, 37B active)
  - (Qwen3-8B 8B already in capacity-curve data)

Question: do these models' Zorbetik application accuracy land on the
projected curve? If yes → ceiling is valid. If they over-perform →
ceiling is conservative (good news). If they under-perform →
something else limits, possibly active-params (MoE) rather than total.
"""
from __future__ import annotations
import sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from pathlib import Path
from collections import defaultdict
import numpy as np

HERE = Path(__file__).resolve().parents[1]
COLAB = HERE / 'experiments/encoding_through_loading/v5_data/precal_eval_all.jsonl'
TOGETHER = HERE / 'data/results/capacity_curve_together.jsonl'


def main():
    # Colab data (Qwen2.5 0.5-14B, condition='icl' only)
    print('=== READING COLAB (0.5B → 14B, Qwen2.5) ===')
    crows = [json.loads(l) for l in open(COLAB, encoding='utf-8') if l.strip()]
    by_size_qt = defaultdict(lambda: {'n': 0, 'correct': 0})
    for r in crows:
        if r.get('condition') != 'icl':
            continue
        sz = r.get('size_B')
        qt = r.get('question_type')
        if sz is None or qt is None:
            continue
        d = by_size_qt[(sz, qt)]
        d['n'] += 1
        d['correct'] += 1 if r.get('correct') else 0
    colab_pts = []
    print(f'{"Size":>6s} {"cloze":>10s} {"app":>10s}')
    for sz in sorted(set(k[0] for k in by_size_qt)):
        clz = by_size_qt[(sz, 'cloze')]
        app = by_size_qt[(sz, 'application')]
        clz_acc = clz['correct']/clz['n'] if clz['n'] else 0
        app_acc = app['correct']/app['n'] if app['n'] else 0
        print(f'{sz:>6.1f}B {clz_acc*100:>9.1f}% {app_acc*100:>9.1f}%')
        colab_pts.append({'size': sz, 'active_size': sz, 'cloze': clz_acc, 'app': app_acc})

    # Together data
    print(f'\n=== READING TOGETHER/FIREWORKS (8B → 671B) ===')
    trows = [json.loads(l) for l in open(TOGETHER, encoding='utf-8') if l.strip()]
    by_model_qt = defaultdict(lambda: {'n': 0, 'correct': 0, 'size': 0, 'active': 0})
    for r in trows:
        m = r['model']
        qt = r['q_type']
        d = by_model_qt[(m, qt)]
        d['n'] += 1
        d['correct'] += 1 if r.get('correct') else 0
        d['size'] = r.get('size_B', 0)
        d['active'] = r.get('active_B', r.get('size_B', 0))
    together_pts = []
    print(f'{"Model":>50s} {"size":>5s} {"act":>5s} {"cloze%":>7s} {"app%":>6s}')
    for m in sorted(set(k[0] for k in by_model_qt), key=lambda x: by_model_qt[(x, 'cloze')]['size']):
        clz = by_model_qt.get((m, 'cloze'), {})
        app = by_model_qt.get((m, 'application'), {})
        if clz.get('n', 0) == 0 or app.get('n', 0) == 0:
            continue
        clz_acc = clz['correct']/clz['n']
        app_acc = app['correct']/app['n']
        sz = clz['size']
        ac = clz['active']
        short = m.split('/')[-1][:48]
        print(f'{short:>50s} {sz:>4.0f}B {ac:>4.0f}B {clz_acc*100:>6.1f}% {app_acc*100:>5.1f}%')
        together_pts.append({'model': m, 'size': sz, 'active_size': ac, 'cloze': clz_acc, 'app': app_acc})

    # Combined curve — fit on Colab, predict on Together
    print(f'\n=== TWO PROJECTION FRAMES ===')
    print(f'(a) Total params (treat MoE total as effective)')
    print(f'(b) Active params (treat MoE active as effective)')

    from scipy.optimize import curve_fit
    sizes_colab = np.array([p['size'] for p in colab_pts])
    app_colab = np.array([p['app'] for p in colab_pts])

    def logistic(x, asymptote, midpoint, slope):
        return asymptote / (1 + np.exp(-slope * (np.log10(x) - midpoint)))

    # Fit on Colab application only
    p0 = [1.0, np.log10(10), 2.0]
    try:
        popt, _ = curve_fit(logistic, sizes_colab, app_colab, p0=p0, maxfev=5000)
        asymp, mid, slope = popt
        print(f'\nLogistic fit on Colab application: asymptote={asymp*100:.1f}%, midpoint@{10**mid:.1f}B, slope={slope:.2f}')
        x_90 = 10**(mid + np.log(9)/slope)
        x_99 = 10**(mid + np.log(99)/slope)
        print(f'  Projection: 90% at {x_90:.0f}B, 99% at {x_99:.0f}B')

        print(f'\n--- (a) Predictions vs actuals using TOTAL params ---')
        print(f'{"Model":>50s} {"size":>5s} {"pred":>6s} {"actual":>7s} {"diff":>6s}')
        for p in together_pts:
            sz = p['size']
            pred = logistic(sz, *popt) * 100
            actual = p['app'] * 100
            diff = actual - pred
            short = p['model'].split('/')[-1][:48]
            verdict = '✅' if abs(diff) < 10 else ('UP' if diff > 0 else 'DOWN')
            print(f'{short:>50s} {sz:>4.0f}B {pred:>5.1f}% {actual:>6.1f}% {diff:>+5.1f}pp {verdict}')

        print(f'\n--- (b) Predictions vs actuals using ACTIVE params ---')
        print(f'{"Model":>50s} {"act":>5s} {"pred":>6s} {"actual":>7s} {"diff":>6s}')
        for p in together_pts:
            ac = p['active_size']
            pred = logistic(ac, *popt) * 100
            actual = p['app'] * 100
            diff = actual - pred
            short = p['model'].split('/')[-1][:48]
            verdict = '✅' if abs(diff) < 10 else ('UP' if diff > 0 else 'DOWN')
            print(f'{short:>50s} {ac:>4.0f}B {pred:>5.1f}% {actual:>6.1f}% {diff:>+5.1f}pp {verdict}')

        # Combined fit (Colab + Together) using BOTH active and total
        print(f'\n=== COMBINED FIT (Colab + Together, all 9 points) ===')
        for use_active in [False, True]:
            if use_active:
                sizes = np.concatenate([sizes_colab, [p['active_size'] for p in together_pts]])
            else:
                sizes = np.concatenate([sizes_colab, [p['size'] for p in together_pts]])
            apps = np.concatenate([app_colab, [p['app'] for p in together_pts]])
            try:
                p0 = [1.0, np.log10(20), 1.5]
                popt2, _ = curve_fit(logistic, sizes, apps, p0=p0, maxfev=5000)
                a, m, s = popt2
                x_90_2 = 10**(m + np.log(9)/s)
                x_99_2 = 10**(m + np.log(99)/s)
                kind = 'ACTIVE' if use_active else 'TOTAL'
                print(f'  ({kind} params) asymp={a*100:.1f}%, mid@{10**m:.1f}B, slope={s:.2f}, 90%-at={x_90_2:.0f}B, 99%-at={x_99_2:.0f}B')
            except Exception as e:
                print(f'  fit failed: {e}')

        # Cloze same analysis briefly
        print(f'\n=== Cloze comparison (saturation analysis) ===')
        clz_colab = np.array([p['cloze'] for p in colab_pts])
        # Cloze is non-monotonic on Colab so logistic fit is unreliable
        for p in together_pts:
            print(f'  {p["model"].split("/")[-1][:40]:>40s} ({p["size"]:>4.0f}B): cloze {p["cloze"]*100:.1f}%')

    except Exception as e:
        print(f'fit failed: {e}')


if __name__ == '__main__':
    main()
