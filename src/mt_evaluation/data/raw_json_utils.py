# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import json
from pathlib import Path


def main():
    # Use relative path from project root
    project_root = Path(__file__).parent.parent.parent.parent.parent
    wmt25_ja_zh_CN_filepath = project_root / "data/wmt25/data/mqm_generalMT2025_en-ko_KR_with_errors.json"

    wmt25_ja_zh_CN = json.loads(open(str(wmt25_ja_zh_CN_filepath)).read())

    keys = list(wmt25_ja_zh_CN.keys())
    num_samples = len(wmt25_ja_zh_CN[keys[0]])
    samples_keys = wmt25_ja_zh_CN[keys[0]].keys()
    samples = []
    for sample_key in samples_keys:
        sample = {}
        for field_key, values in wmt25_ja_zh_CN.items():
            sample[field_key] = values[sample_key]
        samples.append(sample)

    breakpoint()


if __name__ == "__main__":
    main()
