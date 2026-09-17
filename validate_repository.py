from pathlib import Path
root = Path(__file__).resolve().parent
files = sorted((root / 'src').glob('*.py'))
failed = []
markers = ['write_text(code', 'Path(\"/mnt/data/']
for path in files:
    text = path.read_text(encoding='utf-8')
    try:
        compile(text, str(path), 'exec')
    except Exception as exc:
        failed.append(f'syntax: {path.relative_to(root)}: {exc}')
    for marker in markers:
        if marker in text:
            failed.append(f'wrapper marker {marker!r}: {path.relative_to(root)}')
required = ['README.md','CITATION.cff','REPRODUCIBILITY_STATUS.md','data/DATASET_CITATION.md','docs/RUN_ORDER.md','docs/SCIENTIFIC_LOCK.md']
for rel in required:
    if not (root / rel).exists(): failed.append(f'missing: {rel}')
if failed:
    print('\n'.join('FAIL  '+x for x in failed))
    raise SystemExit(1)
print(f'PASS: compiled and wrapper-checked {len(files)} Python source files.')
print('PASS: required release-candidate metadata files are present.')
