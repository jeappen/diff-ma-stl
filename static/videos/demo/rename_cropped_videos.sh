#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# Renames:
#   "..._n8_..._fr75_sr75stlpy,30,mcover3,True,None,False.mp4"
# to
#   "n8_stlpy,30,mcover3.mp4"
#
# Works for planners/specs like "diffusion,15,mseq3-...", "stlpy,30,mcover3", etc.
# It does NOT hardcode any template; it parses specs from the filename.

# Optional prefix argument: normalize so we end with exactly one underscore if provided
prefix_arg="${1-}"
prefix=""
if [[ -n "${prefix_arg}" ]]; then
  # strip any trailing underscores from the arg, then add one
  trimmed="${prefix_arg%_}"
  if [[ -n "$trimmed" ]]; then
    prefix="${trimmed}_"
  else
    prefix=""  # if arg was only underscores, treat as empty
  fi
fi

# Find cropped*.mp4 safely (handles weird chars)
while IFS= read -r -d '' f; do
  base="${f##*/}"

  # 1) Extract nNN
  if [[ "$base" =~ _n([0-9]+) ]]; then
    n="${BASH_REMATCH[1]}"
  else
    echo "Skipping (no _nNN): $base"
    continue
  fi

  # 2) Take everything after the LAST "_sr"
  after_last_sr="${base##*_sr}"         # e.g., "97diffusion,15,...mp4" or "100stlpy,30,...mp4"
  if [[ "$after_last_sr" == "$base" ]]; then
    echo "Skipping (no _srNN): $base"
    continue
  fi

  # 3) Drop .mp4 and then strip leading digits after the last _srNN
  spec_with_flags="${after_last_sr%.mp4}"
  spec="$spec_with_flags"
  if [[ "$spec_with_flags" =~ ^[0-9]+(.*)$ ]]; then
    spec="${BASH_REMATCH[1]}"
  fi

  # 4) Remove trailing ,True/False/None flags (repeat until none remain)
  while [[ "$spec" =~ ,(True|False|None)$ ]]; do
    spec="${spec%,True}"
    spec="${spec%,False}"
    spec="${spec%,None}"
  done

  # 5) Trim any leading separators that might remain
  while [[ "$spec" =~ ^[,_] ]]; do
    spec="${spec#?}"
  done

  if [[ -z "$spec" ]]; then
    echo "Skipping (empty spec after stripping): $base"
    continue
  fi

  # Build new name with optional prefix
  new="${prefix}n${n}_${spec}.mp4"

  # Avoid overwriting (use same directory as source)
  dir="${f%/*}"
  [[ "$dir" == "$f" ]] && dir="."
  dest="$dir/$new"
  i=1
  while [[ -e "$dest" ]]; do
    dest="$dir/${prefix}n${n}_${spec}_$i.mp4"
    ((i++))
  done

  echo "Renaming:"
  echo "  $base"
  echo "  -> $dest"
  mv -- "$f" "$dest"
done < <(find . -maxdepth 1 -type f -name 'cropped*.mp4' -print0)