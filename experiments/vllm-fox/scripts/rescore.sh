find ./results -maxdepth 1 -type f -name '*_ocr_*.json' ! -name '*_eval.json' -print0 | \
  xargs -0 -n1 -I{} bash -c 'echo "Scoring {}"; python3 eval_tools/eval_ocr_test.py --out_file "{}"'