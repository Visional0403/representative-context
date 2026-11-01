"""Download the specified model files or complete task split at fixed Hub versions."""

import argparse
import json
from pathlib import Path, PurePosixPath

from _bootstrap import ROOT


def inside_root(relative):
    target = (ROOT / relative).resolve()
    if not target.is_relative_to(ROOT.resolve()):
        raise ValueError("Asset destinations must be inside the repository.")
    return target


def get_file(specification, item, name, download):
    target = inside_root(item["local"])
    if target.is_file():
        return False
    source = Path(download(
        repo_id=specification["repo_id"], repo_type=specification["repo_type"],
        revision=specification["revision"], filename=item["remote"],
        local_dir=ROOT / "outputs/downloads/hub" / name))
    target.parent.mkdir(parents=True, exist_ok=True)
    source.replace(target)
    return True


def selected_files(specification):
    selection = specification.get("selection")
    if selection == "gamus_test":
        samples = json.loads((ROOT / "configs/gamus/samples.json").read_text())
        if len(samples) != specification["evaluation_units"]:
            raise ValueError("The fixed GAMUS test manifest has an unexpected size.")
        return [dict(remote=row[key], local="datasets/gamus/" + row[key])
                for row in samples for key in ("image", "height")]
    if selection == "fsc147_test":
        split = json.loads((ROOT / "datasets/fsc147/annotations/Train_Test_Val_FSC_147.json").read_text())["test"]
        if len(split) != specification["evaluation_units"]:
            raise ValueError("The official FSC147 test list has an unexpected size.")
        if any(PurePosixPath(name).name != name for name in split):
            raise ValueError("FSC147 image identifiers must be file names.")
        return [dict(remote="images_384_VarV2/" + name,
                     local="datasets/fsc147/images_384_VarV2/" + name) for name in split]
    return []


def main():
    specifications = json.loads((ROOT / "configs/assets/hub_assets.json").read_text())["assets"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets", nargs="+", choices=tuple(specifications))
    parser.add_argument("--list", action="store_true", help="Show the fixed specifications without downloading.")
    args = parser.parse_args()
    if args.list:
        print(json.dumps({name: specifications[name] for name in args.assets}, indent=2))
        return
    from huggingface_hub import hf_hub_download

    for name in args.assets:
        specification = specifications[name]
        written = sum(get_file(specification, item, name, hf_hub_download)
                      for item in specification["files"])
        selected = selected_files(specification)
        for index, item in enumerate(selected, 1):
            written += get_file(specification, item, name, hf_hub_download)
            if index % 100 == 0:
                print(f"{name}: checked {index}/{len(selected)} selected files", flush=True)
        print(f"{name}: ready; downloaded {written} files", flush=True)


if __name__ == "__main__":
    main()
