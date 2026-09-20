import re

svg = open('Eval/plots/sentiment_distribution.svg', encoding='utf-8').read()

print('=== viewBox ===')
m = re.search(r'viewBox="([\d\.\- ]+)"', svg)
print(m.group(1) if m else 'n/a')

# Boxplot boxes are filled paths with opacity 0.7. Extract their x-ranges per color.
grouped = {}
for pm in re.finditer(r'<path([\s\S]*?)/>', svg):
    tag = pm.group(1)
    dm = re.search(r'd="([\s\S]*?)"', tag)
    fm = re.search(r'fill:\s*(#[0-9a-fA-F]{6}); opacity: 0\.7', tag)
    if dm and fm:
        d = dm.group(1)
        xs = [float(v[0]) for v in re.findall(r'[ML]\s*([\d\.\-]+),', d)]
        if xs:
            grouped.setdefault(fm.group(1), []).append((min(xs), max(xs)))

print('\n=== boxes (opacity 0.7) per color: x-ranges ===')
for c in sorted(grouped):
    spans = sorted(grouped[c])
    print(f'{c}: {len(spans)} box parts')
    for s in spans:
        print(f'    [{s[0]:.1f} .. {s[1]:.1f}]')