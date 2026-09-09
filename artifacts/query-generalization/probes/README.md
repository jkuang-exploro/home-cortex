# G2 / G3 probes

Not on pytest `testpaths`. Default `python -m pytest -q` stays green.

```sh
# G2 desired-behavior counterexamples (currently 5 fail, 4 pass)
PYTHONPATH=src python -m pytest -q artifacts/query-generalization/probes/test_counterexamples.py

# G3 frozen interpreter eval (synthetic graphs, V1 and V2)
PYTHONPATH=src python artifacts/query-generalization/probes/run_frozen_interpreter.py \
  --ollama-url http://ollama:11434 --model qwen3.5:9b --profile both
```

Do not pass sequence YAML to `load_probe_dataset`. The G3 runner uses `composition_eval.load_sequences()`.
