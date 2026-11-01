"""Fetch pinned task-model sources from their authors' public repositories."""

import argparse
import io
import json
from pathlib import Path, PurePosixPath
import urllib.request
import zipfile

from _bootstrap import ROOT


def fetch_source(name, specification, destination):
    target = destination / name
    if target.exists():
        raise FileExistsError(f"Source directory already exists: {target}")
    repository, revision = specification["repository"], specification["revision"]
    url = f"https://codeload.github.com/{repository}/zip/{revision}"
    request = urllib.request.Request(
        url, headers={"User-Agent": "representative-context"}
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        prefix = archive.namelist()[0].split("/")[0]
        base = PurePosixPath(prefix) / specification["subdirectory"]
        selected = []
        for item in archive.infolist():
            path = PurePosixPath(item.filename)
            if item.is_dir() or ".." in path.parts or path.is_absolute():
                continue
            try:
                relative = path.relative_to(base)
            except ValueError:
                continue
            if not relative.parts or not any(
                relative == PurePosixPath(entry)
                or PurePosixPath(entry) in relative.parents
                for entry in specification["include"]
            ):
                continue
            if ((item.external_attr >> 16) & 0o170000) == 0o120000:
                raise ValueError(
                    f"Symbolic links are not supported in model sources: {path}"
                )
            selected.append((item, relative))
        if not selected:
            raise ValueError(f"No requested model files found in {url}")
        for item, relative in selected:
            output = target.joinpath(*relative.parts)
            output.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(item) as source, output.open("xb") as sink:
                sink.write(source.read())
    (target / ".source.json").write_text(
        json.dumps(specification, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Prepared {name}: {revision}")


def main():
    specifications = json.loads(
        (ROOT / "third_party/sources.json").read_text(encoding="utf-8")
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+", choices=tuple(specifications))
    parser.add_argument(
        "--destination", type=Path, default=ROOT / "third_party/sources"
    )
    args = parser.parse_args()
    for name in args.models:
        fetch_source(name, specifications[name], args.destination)


if __name__ == "__main__":
    main()
