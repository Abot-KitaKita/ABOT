#!/usr/bin/env bash
# Generic segmented fetch: seg_fetch.sh URL TOTAL OUT NPARTS
# Max 6 parallel connections (observed cap), per-part retries, exact-size verification.
set -euo pipefail
URL="$1"; TOTAL="$2"; OUT="$3"; N="${4:-6}"
CHUNK=$(( (TOTAL + N - 1) / N ))
PARTS="$OUT.parts"
mkdir -p "$(dirname "$OUT")" "$PARTS"

[ -f "$OUT" ] && [ "$(stat -c%s "$OUT")" -eq "$TOTAL" ] && { echo "already complete"; exit 0; }

part_want() { local i=$1; if [ "$i" -eq $((N - 1)) ]; then echo $((TOTAL - i * CHUNK)); else echo "$CHUNK"; fi; }

fetch_part() {
  local i=$1 start end want f t
  start=$((i * CHUNK)); end=$((start + CHUNK - 1))
  [ "$end" -ge "$TOTAL" ] && end=$((TOTAL - 1))
  want=$(part_want "$i")
  f=$(printf "%s/part_%02d" "$PARTS" "$i")
  [ -f "$f" ] && [ "$(stat -c%s "$f")" -eq "$want" ] && { echo "part $i ok"; return 0; }
  for t in 1 2 3 4 5 6; do
    rm -f "$f.tmp"
    if curl -sS --connect-timeout 25 --max-time 1800 -r "${start}-${end}" -o "$f.tmp" "$URL" \
       && [ "$(stat -c%s "$f.tmp" 2>/dev/null || echo 0)" -eq "$want" ]; then
      mv "$f.tmp" "$f"; echo "part $i ok ($want bytes, try $t)"; return 0
    fi
    sleep $((t * 2))
  done
  rm -f "$f.tmp"
  echo "FAILED part $i after retries" >&2
  return 1
}
export -f fetch_part part_want
export URL TOTAL CHUNK PARTS N

ok=0
for round in $(seq 1 12); do
  missing=""
  for i in $(seq 0 $((N - 1))); do
    f=$(printf "%s/part_%02d" "$PARTS" "$i")
    want=$(part_want "$i")
    if [ ! -f "$f" ] || [ "$(stat -c%s "$f")" -ne "$want" ]; then missing="$missing $i"; fi
  done
  if [ -z "$missing" ]; then ok=1; break; fi
  echo "round $round fetching:$missing"
  echo $missing | tr ' ' '\n' | grep -v '^$' | xargs -P 6 -I{} bash -c 'fetch_part {}' || true
done
[ "$ok" -eq 1 ] || { echo "parts still incomplete after retries"; exit 1; }

size=0
for i in $(seq 0 $((N - 1))); do
  f=$(printf "%s/part_%02d" "$PARTS" "$i")
  sz=$(stat -c%s "$f"); want=$(part_want "$i")
  [ "$sz" -eq "$want" ] || { echo "BAD part $i: $sz != $want"; exit 1; }
  size=$((size + sz))
done
[ "$size" -eq "$TOTAL" ] || { echo "BAD total $size"; exit 1; }
: > "$OUT.assembled"
for i in $(seq 0 $((N - 1))); do cat "$(printf "$PARTS/part_%02d" "$i")" >> "$OUT.assembled"; done
[ "$(stat -c%s "$OUT.assembled")" -eq "$TOTAL" ] || { echo "BAD assembled size"; exit 1; }
mv "$OUT.assembled" "$OUT"
rm -rf "$PARTS"
echo "SEG_DONE $(stat -c%s "$OUT") bytes -> $OUT"
