"""Verify checkpoint contents and materialize them in a new directory without replacing a working tree."""
import argparse,hashlib,json,shutil
from pathlib import Path
from zipfile import ZipFile
root=Path(__file__).resolve().parent
def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('selection',choices=('qualified','experimental','live-host'));p.add_argument('destination',type=Path)
    args=p.parse_args();dest=args.destination.resolve()
    if dest.exists():p.error('Choose a new destination directory.')
    dest.parent.mkdir(parents=True,exist_ok=True)
    from tempfile import TemporaryDirectory
    with TemporaryDirectory(dir=dest.parent) as tmp:
        stage=Path(tmp)/'restore';stage.mkdir()
        if args.selection=='live-host':
            expected=json.loads((root/'live-host-manifest.json').read_text())
            with ZipFile(root/'live-host.zip') as archive:
                assert set(archive.namelist())==set(expected)
                for key in expected:
                    file=(stage/key).resolve();assert file.is_relative_to(stage)
                    file.parent.mkdir(parents=True,exist_ok=True)
                    with archive.open(key) as src,file.open('xb') as out:shutil.copyfileobj(src,out)
        else:
            base=root/args.selection;expected=json.loads((base/'sha256.json').read_text())
            for key in [*expected,'sha256.json']:
                file=(stage/key).resolve();assert file.is_relative_to(stage)
                file.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(base/key,file)
        for key,checksum in expected.items():
            if digest(stage/key)!=checksum:raise ValueError(f'Checksum mismatch for {key}')
        stage.rename(dest)
    print(f'Restored and verified {len(expected)} files at {dest}')

