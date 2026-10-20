"""Keep the per-stock chain files on the GitHub Release 'options-data'.

    python -m atrade.release download            # Release -> data/chains/<TICKER>.parquet
    python -m atrade.release upload [--since ISO] # changed stocks -> Release

Each stock has its own parquet file. On the Release they are grouped into 27
zip bundles by first letter (chains_A.zip holds AAPL.parquet, ABBV.parquet,
...; chains_0.zip holds tickers not starting with a letter), because GitHub
lets a workflow make only 1,000 API requests an hour and replacing 516
separate files every night would need about 1,500.

Uses the GitHub REST API through the `gh` command (in GitHub Actions it
authenticates with GITHUB_TOKEN). Replacing a file is done safely: the new
file is uploaded under a temporary name, the old one deleted, and the new
one renamed, so a reader never finds the stock missing for long.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .options_store import CHAIN_DIR, RELEASE_TAG, REPO


def gh(*args, input_path: Path | None = None, method: str = "GET", fields: dict | None = None):
    cmd = ["gh", "api", "-X", method]
    for k, v in (fields or {}).items():
        cmd += ["-F", f"{k}={str(v).lower()}"] if isinstance(v, bool) else ["-f", f"{k}={v}"]
    if input_path is not None:
        cmd += ["-H", "Content-Type: application/octet-stream", "--input", str(input_path)]
    cmd += list(args)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"gh api {' '.join(args)} failed: {r.stderr.strip() or r.stdout.strip()}")
    return json.loads(r.stdout) if r.stdout.strip() else {}


INBOX_TAG = "import-inbox"


def get_release(create: bool = True, tag: str = RELEASE_TAG) -> dict:
    try:
        return gh(f"repos/{REPO}/releases/tags/{tag}")
    except RuntimeError as e:
        if "404" not in str(e) and "Not Found" not in str(e):
            raise
        if not create:
            raise
        if tag == INBOX_TAG:
            return gh(f"repos/{REPO}/releases", method="POST", fields={
                "tag_name": INBOX_TAG, "name": "Import inbox", "prerelease": True,
                "body": "Drop history files here (Edit -> attach files), then run the 'Options history import' "
                        "workflow. Accepted: chain_history*.parquet, oi_history*.parquet, gex_history*.csv, and "
                        "*.zip files holding Watchtower-style chains/ticker=XXX/date=YYYY-MM-DD.parquet "
                        "(+ features/). Files are removed after a successful import."})
        return gh(f"repos/{REPO}/releases", method="POST", fields={
            "tag_name": RELEASE_TAG, "name": "Options data (one file per stock)",
            "body": "Full contract-level options chains, one parquet file per stock, every day appended. "
                    "Updated automatically by the 'Options data sync' workflow. Not a software release."})


def list_assets(release_id: int) -> dict:
    out, page = {}, 1
    while True:
        batch = gh(f"repos/{REPO}/releases/{release_id}/assets?per_page=100&page={page}")
        for a in batch:
            out[a["name"]] = a
        if len(batch) < 100:
            return out
        page += 1


def shard_of(ticker: str) -> str:
    c = ticker.upper()[:1]
    return c if "A" <= c <= "Z" else "0"


def shard_name(shard: str) -> str:
    return f"chains_{shard}.zip"


def pack_shard(src: Path, shard: str, out_dir: Path) -> Path:
    import zipfile
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / shard_name(shard)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED) as z:   # parquet is already compressed
        for f in sorted(src.glob("*.parquet")):
            if shard_of(f.stem) == shard:
                z.write(f, f.name)
    return out


def unpack(zip_path: Path, dest: Path, only: str | None = None) -> list[str]:
    import zipfile
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if n.endswith(".parquet") and (only is None or n == only)]
        for n in names:
            (dest / n).write_bytes(z.read(n))
    return names


def upload_file(release: dict, path: Path, existing: dict) -> str:
    name = path.name
    tmp_name = name + ".new"
    if tmp_name in existing:
        gh(f"repos/{REPO}/releases/assets/{existing[tmp_name]['id']}", method="DELETE")
    up = gh(f"https://uploads.github.com/repos/{REPO}/releases/{release['id']}/assets?name={tmp_name}",
            method="POST", input_path=path)
    if name in existing:
        gh(f"repos/{REPO}/releases/assets/{existing[name]['id']}", method="DELETE")
    gh(f"repos/{REPO}/releases/assets/{up['id']}", method="PATCH", fields={"name": name})
    return name


def download_all(dest: Path = CHAIN_DIR, workers: int = 8) -> int:
    import requests
    import tempfile
    rel = get_release(create=False)
    assets = list_assets(rel["id"])
    dest.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp())
    count = []

    def get(a):
        r = requests.get(a["browser_download_url"], timeout=600)
        r.raise_for_status()
        z = tmp / a["name"]; z.write_bytes(r.content)
        count.extend(unpack(z, dest)); z.unlink()
    todo = [a for n, a in assets.items() if n.startswith("chains_") and n.endswith(".zip")]
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(get, todo))
    return len(count)


def upload_changed(src: Path = CHAIN_DIR, since: dt.datetime | None = None, workers: int = 4) -> int:
    import tempfile
    rel = get_release(create=True)
    existing = list_assets(rel["id"])
    changed = sorted(src.glob("*.parquet"))
    if since is not None:
        changed = [f for f in changed if dt.datetime.fromtimestamp(f.stat().st_mtime) >= since]
    shards = sorted({shard_of(f.stem) for f in changed})
    out_dir = Path(tempfile.mkdtemp())
    files = [pack_shard(src, sh, out_dir) for sh in shards]
    errors = []

    def put(f):
        try:
            upload_file(rel, f, existing)
        except Exception as e:      # retry once
            try:
                upload_file(rel, f, list_assets(rel["id"]))
            except Exception as e2:
                errors.append((f.name, str(e2)[:200]))
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(put, files))
    for n, e in errors[:20]:
        print(f"  upload failed: {n}: {e}")
    if errors:
        raise SystemExit(f"{len(errors)} uploads failed")
    print(f"{len(changed)} changed stock files in {len(files)} bundle(s).")
    return len(changed)


def inbox_download(dest: Path) -> list[Path]:
    """Download everything in the import inbox; returns the local paths."""
    import requests
    rel = get_release(create=True, tag=INBOX_TAG)
    dest.mkdir(parents=True, exist_ok=True)
    out = []
    for name, a in list_assets(rel["id"]).items():
        r = requests.get(a["browser_download_url"], timeout=600)
        r.raise_for_status()
        (dest / name).write_bytes(r.content)
        out.append(dest / name)
    return out


def inbox_clear():
    rel = get_release(create=False, tag=INBOX_TAG)
    for a in list_assets(rel["id"]).values():
        gh(f"repos/{REPO}/releases/assets/{a['id']}", method="DELETE")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["download", "upload", "inbox-download", "inbox-clear"])
    ap.add_argument("--dest", default="import")
    ap.add_argument("--since", help="only upload files changed since this ISO time")
    a = ap.parse_args(argv)
    if a.action == "download":
        try:
            print(f"Downloaded {download_all()} stock files.")
        except RuntimeError as e:
            print(f"No release yet ({e}); starting empty.")
    elif a.action == "upload":
        since = dt.datetime.fromisoformat(a.since) if a.since else None
        print(f"Uploaded {upload_changed(since=since)} stock files.")
    elif a.action == "inbox-download":
        files = inbox_download(Path(a.dest))
        print(f"Downloaded {len(files)} inbox file(s): {', '.join(f.name for f in files) or 'none'}")
    else:
        inbox_clear(); print("Inbox cleared.")


if __name__ == "__main__":
    sys.exit(main())
