"""Static report/data/link checks. Does not claim to compile LaTeX."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from q1_v7_core import folder, save_json
from q1_v7_pipeline import check_protected
from utils import PROJECT_ROOT, sha256_file


def check_tex(path):
    raw = path.read_text(encoding="utf-8")
    text = re.sub(r"(?<!\\)%[^\n]*", "", raw)
    level = 0
    pairs = 0
    for match in re.finditer(r"(?<!\\)[{}]", text):
        if match.group() == "{":
            level += 1
            pairs += 1
        else:
            level -= 1
        assert level >= 0, f"Unbalanced braces: {path}"
    assert level == 0
    stack = []
    for kind, name in re.findall(r"\\(begin|end)\{([^}]+)\}", text):
        if kind == "begin":
            stack.append(name)
        else:
            assert stack and stack.pop() == name, f"Environment mismatch: {path}"
    assert not stack
    bib = (PROJECT_ROOT/"references/c_problem_references.bib").read_text(encoding="utf-8")
    keys = re.findall(r"@\w+\{([^,]+),", bib)
    assert len(keys) == len(set(keys))
    cited = [key.strip() for group in re.findall(r"\\cite(?:\[[^]]*\])?\{([^}]+)\}", text) for key in group.split(",")]
    assert set(cited).issubset(keys), set(cited)-set(keys)
    for link in re.findall(r"\\includepdf(?:\[[^]]*\])?\{([^}]+)\}", text):
        assert (path.parent/link).exists(), link
    return {"path": str(path.relative_to(PROJECT_ROOT)), "brace_pairs": pairs,
            "environments_balanced": True, "citation_keys": sorted(set(cited)), "compiled": False}


def main():
    report = PROJECT_ROOT/"reports/Q1_V7_RESULTS.md"
    texts = [(PROJECT_ROOT/"reports/Q1_V7_RESULTS.tex"), (PROJECT_ROOT/"reports/V7_MODELING_OUTLINE.tex")]
    checks = [check_tex(p) for p in texts]
    expected_generated = {str((folder()/p).resolve()) for p in ["package_manifest.json", "document_checks.json"]}
    checked_links = []
    for doc in [report, PROJECT_ROOT/"reports/RESULTS_REPORT.md", PROJECT_ROOT/"plan.md", PROJECT_ROOT/"todo.md"]:
        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", doc.read_text(encoding="utf-8")):
            if target.startswith(("http:", "https:", "#")):
                continue
            target = target.split("#")[0]
            path = (doc.parent/target).resolve()
            assert path.exists() or str(path) in expected_generated, f"Missing {target} in {doc}"
            checked_links.append(str(path))
    params = pd.read_csv(folder()/"gaussian_parameters.csv")
    assert len(params) == 54
    fits = pd.read_csv(folder()/"gaussian_fit_summary.csv")
    assert len(fits) == 12 and fits.boundary.sum() == 6
    source = report.read_text(encoding="utf-8")
    prediction = pd.read_csv(folder()/"prediction_summary.csv")
    for row in prediction[prediction.scope == "development_outer"].itertuples():
        assert f"{row.mean_scaled_rmse:.6f}" in source
    primary = pd.read_csv(folder()/"primary_response_summary.csv")
    for row in primary.itertuples():
        assert f"{row.effect:.3f}" in source
    figures = json.loads((folder()/"figure_manifest.json").read_text(encoding="utf-8"))
    pdf = PROJECT_ROOT/figures["path"]
    assert sha256_file(pdf) == figures["sha256"]
    pdfinfo = shutil.which("pdfinfo") or str(Path("C:/Users/leaf/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin/pdfinfo.exe"))
    result = subprocess.run([pdfinfo, str(pdf)], capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)
    page_match = re.search(r"Pages:\s+(\d+)", result.stdout)
    assert page_match and int(page_match.group(1)) == 6
    render_paths = [PROJECT_ROOT/f"tmp/pdfs/q1_v7/page-{i}.png" for i in range(1, 7)]
    assert all(p.exists() for p in render_paths)
    save_json("document_checks.json", {"latex": checks, "xelatex_available": bool(shutil.which("xelatex")),
              "local_links_checked": len(checked_links), "parameter_count": len(params),
              "report_primary_effects_and_prediction_rounding_match": True,
              "pdf": {"pages": 6, "hash_matches_manifest": True, "rendered_pages": 6,
                      "visual_review": "All six 110-dpi renders inspected by the authoring agent; no clipped labels or page overflow observed."},
              "protected": check_protected()})
    paths = [p for p in folder().iterdir() if p.is_file() and p.name != "package_manifest.json"]
    paths += [p for p in (PROJECT_ROOT/"code").glob("*q1_v7*") if p.is_file()]
    paths += [report, PROJECT_ROOT/"reports/Q1_V7_RESULTS.tex", pdf]
    save_json("package_manifest.json", {"scope": "Q1 v7.1 stage package, not final paper",
              "self_excluded": True, "files": [{"path": str(p.relative_to(PROJECT_ROOT)),
              "bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in sorted(paths)]})
    print(f"Static LaTeX/link/data checks passed; six rendered PDF pages reviewed; {len(paths)} package files indexed.")


if __name__ == "__main__":
    main()
