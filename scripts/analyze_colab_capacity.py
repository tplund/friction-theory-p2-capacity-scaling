"""Analyze Colab ETL v5 pre-cal capacity curve (Qwen2.5 0.5B → 14B).

Tests the empirical capacity-scaling hypothesis on the ICL condition:
  - cloze (retrieval): can the model find a fact in context?
  - application (chain reasoning): can the model derive across facts?

Predictions from Paper 1 §3 (capacity C as architectural variable):
  - Both axes scale with size
  - Application scales steeper than cloze (derivation needs more capacity)
  - There may be a "knee" where application becomes useful

Output:
  - Per-model accuracy table for cloze/application (ICL-only)
  - Friction (CR) per model — robust statistics
  - Per-model first5 / middle / last5 CR
  - Spearman size×accuracy and size×friction
  - Theoretical-ceiling estimate from saturation behavior
"""
from __future__ import annotations
import sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from pathlib import Path
from collections import defaultdict
import numpy as np

HERE = Path(__file__).resolve().parents[1]
SRC = HERE / 'experiments/encoding_through_loading/v5_data/precal_eval_all.jsonl'


def robust_cr(cr_list):
    """Median is robust to extreme outliers from low-prob continuations."""
    if not cr_list:
        return None
    arr = np.array([float(x) for x in cr_list if x is not None and np.isfinite(x)], dtype=float)
    if len(arr) == 0:
        return None
    return float(np.median(arr))


def main():
    print(f'Reading {SRC}')
    rows = [json.loads(l) for l in open(SRC, encoding='utf-8') if l.strip()]
    print(f'Total rows: {len(rows)}')

    # Group by (model, size_B, condition, question_type)
    by = defaultdict(lambda: {
        'n': 0, 'correct': 0,
        'cr_list': [], 'first5': [], 'middle': [], 'last5': [],
        'n_tokens': [], 'response_lens': [],
    })
    for r in rows:
        k = (r.get('model'), r.get('size_B'), r.get('condition', 'icl'), r.get('question_type'))
        d = by[k]
        d['n'] += 1
        d['correct'] += 1 if r.get('correct') else 0
        cr = r.get('per_token_cr', []) or []
        if cr:
            med = robust_cr(cr)
            if med is not None:
                d['cr_list'].append(med)
            if len(cr) >= 5:
                d['first5'].append(float(np.median(cr[:5])))
                d['last5'].append(float(np.median(cr[-5:])))
            if len(cr) > 15:
                d['middle'].append(float(np.median(cr[5:-5])))
        d['n_tokens'].append(r.get('n_tokens', len(cr)))
        d['response_lens'].append(len(r.get('response', '') or ''))

    # Detect thinking-contamination per cell (response << n_tokens)
    print(f'\n=== CONTAMINATION CHECK (response<20 chars but n_tokens>30) ===')
    for k, d in sorted(by.items(), key=lambda x: (x[0][1] or 0, x[0][2], x[0][3])):
        contam = sum(1 for ntok, rlen in zip(d['n_tokens'], d['response_lens']) if ntok > 30 and rlen < 20)
        pct = contam/d['n']*100 if d['n'] else 0
        if pct > 5:
            print(f'  {pct:5.1f}%  {k[0][:30]:>30s} {k[1]}B  cond={k[2]}  qt={k[3]}')
    print('  (no models flagged → all data clean)')

    # Main capacity table — ICL only
    print(f'\n=== ICL CAPACITY CURVE (no fine-tuning, raw model on context) ===')
    print(f'{"Size":>5s} {"n":>3s} {"cloze_acc":>10s} {"app_acc":>8s} {"medCR_clz":>10s} {"medCR_app":>10s} {"f5_clz":>7s} {"f5_app":>7s}')
    sizes = sorted(set(k[1] for k in by.keys() if k[2] == 'icl' and k[1] is not None))
    icl_data = []
    for sz in sizes:
        clz = by.get((next(k[0] for k in by.keys() if k[1] == sz), sz, 'icl', 'cloze'))
        app = by.get((next(k[0] for k in by.keys() if k[1] == sz), sz, 'icl', 'application'))
        if not clz or not app:
            continue
        clz_acc = clz['correct']/clz['n']*100 if clz['n'] else 0
        app_acc = app['correct']/app['n']*100 if app['n'] else 0
        clz_cr = np.median(clz['cr_list']) if clz['cr_list'] else 0
        app_cr = np.median(app['cr_list']) if app['cr_list'] else 0
        clz_f5 = np.median(clz['first5']) if clz['first5'] else 0
        app_f5 = np.median(app['first5']) if app['first5'] else 0
        print(f'{sz:>5.1f} {clz["n"]:>3d} {clz_acc:>9.1f}% {app_acc:>7.1f}% {clz_cr:>10.2f} {app_cr:>10.2f} {clz_f5:>7.2f} {app_f5:>7.2f}')
        icl_data.append({'size': sz, 'cloze_acc': clz_acc/100, 'app_acc': app_acc/100,
                         'cloze_cr': clz_cr, 'app_cr': app_cr,
                         'cloze_f5': clz_f5, 'app_f5': app_f5})

    # FT capacity table
    print(f'\n=== FT CAPACITY CURVE (after LoRA fine-tuning, no context) ===')
    print(f'{"Size":>5s} {"n":>3s} {"cloze_acc":>10s} {"app_acc":>8s} {"medCR_clz":>10s} {"medCR_app":>10s}')
    ft_data = []
    for sz in sizes:
        try:
            mname = next(k[0] for k in by.keys() if k[1] == sz)
            clz = by.get((mname, sz, 'ft', 'cloze'))
            app = by.get((mname, sz, 'ft', 'application'))
        except StopIteration:
            continue
        if not clz or not app:
            continue
        clz_acc = clz['correct']/clz['n']*100 if clz['n'] else 0
        app_acc = app['correct']/app['n']*100 if app['n'] else 0
        clz_cr = np.median(clz['cr_list']) if clz['cr_list'] else 0
        app_cr = np.median(app['cr_list']) if app['cr_list'] else 0
        print(f'{sz:>5.1f} {clz["n"]:>3d} {clz_acc:>9.1f}% {app_acc:>7.1f}% {clz_cr:>10.2f} {app_cr:>10.2f}')
        ft_data.append({'size': sz, 'cloze_acc': clz_acc/100, 'app_acc': app_acc/100})

    # Scaling analysis
    print(f'\n=== SCALING (log10 size vs accuracy) ===')
    if len(icl_data) >= 3:
        from scipy.stats import spearmanr
        xs = np.log10([d['size'] for d in icl_data])
        for axis in ['cloze_acc', 'app_acc']:
            ys = np.array([d[axis] for d in icl_data])
            slope, intercept = np.polyfit(xs, ys, 1)
            rho, pv = spearmanr(xs, ys)
            print(f'  ICL {axis}: slope={slope*100:+5.1f}pp/decade  intercept@1B={intercept*100:5.1f}%  spearman_rho={rho:+.3f} p={pv:.3f}')
        # FT scaling
        if ft_data and len(ft_data) >= 3:
            for axis in ['cloze_acc', 'app_acc']:
                ys = np.array([d[axis] for d in ft_data])
                if all(y == 0 for y in ys):
                    print(f'  FT  {axis}: ALL ZERO — fine-tuning eval is broken (separate issue)')
                else:
                    slope, intercept = np.polyfit(xs[:len(ys)], ys, 1)
                    print(f'  FT  {axis}: slope={slope*100:+5.1f}pp/decade  intercept@1B={intercept*100:5.1f}%')

    # Theoretical ceiling estimate
    # For cloze: if we fit a saturating curve y = 1 - exp(-k*x^a), the asymptote is 1
    # Better: fit 4-param logistic and read off the asymptote
    print(f'\n=== THEORETICAL CEILING (saturation analysis) ===')
    if len(icl_data) >= 4:
        from scipy.optimize import curve_fit
        sizes_arr = np.array([d['size'] for d in icl_data])
        for axis_name in ['cloze_acc', 'app_acc']:
            ys = np.array([d[axis_name] for d in icl_data])
            # Logistic in log10(size)
            def logistic(x, asymptote, midpoint, slope):
                return asymptote / (1 + np.exp(-slope * (np.log10(x) - midpoint)))
            try:
                # Initial guess: asymptote ~ max+5%, midpoint at log10(size where y=max/2), slope ~ 2
                p0 = [min(1.0, ys.max()*1.05), np.log10(sizes_arr[len(sizes_arr)//2]), 2.0]
                popt, _ = curve_fit(logistic, sizes_arr, ys, p0=p0, maxfev=5000)
                asymptote, midpoint, slope = popt
                # Where does it hit 90% of asymptote?
                x_90 = 10**(midpoint + np.log(9)/slope)
                # And 99%?
                x_99 = 10**(midpoint + np.log(99)/slope)
                print(f'  {axis_name}: asymptote~{asymptote*100:.1f}%  midpoint@{10**midpoint:.1f}B  slope={slope:.2f}  90%-at={x_90:.1f}B  99%-at={x_99:.1f}B')
            except Exception as e:
                print(f'  {axis_name}: fit failed ({str(e)[:60]})')

    # Within-question gap: cloze_correct vs app_correct on SAME fact_id?
    # Probably can't compute (different fact_id sets) but try
    print(f'\n=== Per-fact_id retrieval-vs-derivation gap ===')
    # For 14B specifically: how often is cloze correct AND app wrong?
    target_size = 14.0
    target_model = next((k[0] for k in by.keys() if k[1] == target_size), None)
    if target_model:
        cloze_rows = {r['fact_id']: r for r in rows if r.get('size_B') == target_size and r.get('condition') == 'icl' and r.get('question_type') == 'cloze' and 'fact_id' in r}
        app_rows = {r['fact_id']: r for r in rows if r.get('size_B') == target_size and r.get('condition') == 'icl' and r.get('question_type') == 'application' and 'fact_id' in r}
        common = set(cloze_rows.keys()) & set(app_rows.keys())
        if common:
            both = sum(1 for fid in common if cloze_rows[fid].get('correct') and app_rows[fid].get('correct'))
            cloze_only = sum(1 for fid in common if cloze_rows[fid].get('correct') and not app_rows[fid].get('correct'))
            app_only = sum(1 for fid in common if not cloze_rows[fid].get('correct') and app_rows[fid].get('correct'))
            neither = sum(1 for fid in common if not cloze_rows[fid].get('correct') and not app_rows[fid].get('correct'))
            print(f'  14B on {len(common)} matched fact_ids:')
            print(f'    both correct: {both} ({both/len(common)*100:.0f}%)')
            print(f'    cloze only:   {cloze_only} ({cloze_only/len(common)*100:.0f}%)  ← retrieval works, derivation fails')
            print(f'    app only:     {app_only} ({app_only/len(common)*100:.0f}%)  ← derivation works without confirming retrieval')
            print(f'    neither:      {neither} ({neither/len(common)*100:.0f}%)  <- model cannot even retrieve')
    print(f'\nDONE. {len(icl_data)} size-points analyzed.')


if __name__ == '__main__':
    main()
