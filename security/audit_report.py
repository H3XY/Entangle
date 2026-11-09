"""Turn pip-audit JSON output into a short Markdown report and a console summary.

Usage: python security/audit_report.py <pip-audit.json> <report.md>
"""
import datetime
import json
import sys


def main(src, dst):
    with open(src) as f:
        data = json.load(f)
    deps = data.get("dependencies", [])
    vulns = [(d["name"], d["version"], v) for d in deps for v in d.get("vulns", [])]
    today = datetime.date.today().isoformat()

    lines = ["# Dependency vulnerability audit", "",
             f"Scanned {len(deps)} installed packages with pip-audit on {today}.", ""]
    if not vulns:
        lines.append("No known vulnerabilities found.")
    else:
        lines += ["| Package | Version | ID | Aliases | Fixed in | Summary |",
                  "|---|---|---|---|---|---|"]
        for name, version, v in vulns:
            desc = (v.get("description") or "").replace("\n", " ").replace("|", "\\|")
            lines.append(f"| {name} | {version} | {v['id']} | {', '.join(v.get('aliases', []))} | "
                         f"{', '.join(v.get('fix_versions', [])) or 'none yet'} | {desc[:160]} |")
    with open(dst, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Scanned {len(deps)} packages: "
          + ("no known vulnerabilities." if not vulns else f"{len(vulns)} vulnerability(ies) found:"))
    for name, version, v in vulns:
        fix = ", ".join(v.get("fix_versions", [])) or "no fix yet"
        print(f"  {name} {version}: {v['id']} ({', '.join(v.get('aliases', []))}) -> fixed in {fix}")
    print(f"Report: {dst}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
