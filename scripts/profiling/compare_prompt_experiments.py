"""Compare measured prompt candidates without hiding losses behind aggregate gains."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def compare(before, after):
    for key in ('cases', 'repeat', 'evaluation_cases_sha256'):
        if before.get(key) != after.get(key):
            raise ValueError(f'Incomparable evaluation: {key}')
    for key in ('data_tree_sha256', 'schema_tree_sha256', 'model_digest', 'ollama_version', 'request_settings'):
        if before['provenance'][key] != after['provenance'][key]:
            raise ValueError(f'Incomparable provenance: {key}')
    result = {}
    for field in ('plan_match', 'answer_correct'):
        old = {(r['case_id'], r['sample_index']) for r in before['failures'] if not r[field]}
        new = {(r['case_id'], r['sample_index']) for r in after['failures'] if not r[field]}
        result[field] = {'regressions': sorted(new - old), 'improvements': sorted(old - new)}
    result['metrics'] = {}
    for metric in before['metrics']:
        a, b = before['metrics'][metric]['p50'], after['metrics'][metric]['p50']
        result['metrics'][metric] = {'before_p50': a, 'after_p50': b, 'delta': round(b - a, 3),
                                     'change_percent': round(100 * (b / a - 1), 2) if a else None}
    result['semantic_regression'] = any(result[key]['regressions'] for key in ('plan_match', 'answer_correct'))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('before', type=Path)
    parser.add_argument('after', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = compare(json.loads(args.before.read_text()), json.loads(args.after.read_text()))
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
