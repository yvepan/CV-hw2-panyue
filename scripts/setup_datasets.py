from __future__ import annotations

import argparse
import tarfile
import zipfile
from pathlib import Path


def extract_tar(src: Path, dst: Path) -> None:
    marker = dst / f'.{src.stem}.extracted'
    if marker.exists():
        print(f'skip {src.name}: already extracted')
        return
    dst.mkdir(parents=True, exist_ok=True)
    with tarfile.open(src) as tf:
        tf.extractall(dst)
    marker.write_text(src.name, encoding='utf-8')
    print(f'extracted {src} -> {dst}')


def extract_zip(src: Path, dst: Path) -> None:
    marker = dst / f'.{src.stem}.extracted'
    if marker.exists():
        print(f'skip {src.name}: already extracted')
        return
    dst.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(src) as zf:
        zf.extractall(dst)
    marker.write_text(src.name, encoding='utf-8')
    print(f'extracted {src} -> {dst}')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='.', help='hw2 root containing downloaded archives')
    args = parser.parse_args()
    root = Path(args.root)
    extract_tar(root / 'images.tar.gz', root / 'datasets' / 'oxford_pet' / 'oxford-iiit-pet')
    extract_tar(root / 'annotations.tar.gz', root / 'datasets' / 'oxford_pet' / 'oxford-iiit-pet')
    extract_zip(root / 'archive.zip', root / 'datasets' / 'VisDrone')


if __name__ == '__main__':
    main()



