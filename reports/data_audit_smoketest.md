# Phase 1 Data Audit
**Assumptions**
- Source: OPUS OpenSubtitles en-ta v2016.
- All numbers in this report are derived from `data/processed/pairs.parquet`.
- QC thresholds are read from `configs/default.yaml`.
- LaBSE stage-2 scoring is optional; run `python -m tyto data qc --labse` to enable.

---
## 1. Summary Statistics
| Metric | Value |
|---|---|
| Total pairs | 40 |
| Usable pairs | 40 |
| Usable % | 100.0% |
| Distinct films | 2 |
| Romanised-Tamil rate | 0.00% (0 rows) |

## 2. Figures
![Pair and film counts](artifacts\fig_pair_counts.png)

![Lines per film distribution](artifacts\fig_lines_per_film.png)

![Length distributions](artifacts\fig_length_dist.png)

![Script ratios](artifacts\fig_script_ratios.png)

![QC flag share](artifacts\fig_flag_share.png)

*LaBSE histogram: not yet available (run stage-2 scoring first).*

## 3. QC Flag Counts
| flag       |   n_rows |
|:-----------|---------:|
| CODE_MIXED |       10 |

## 4. Top Films by Pair Count
| film_id   | film_title   |   year |   n_pairs |
|:----------|:-------------|-------:|----------:|
| f1        | Test Film    |   2016 |        30 |
| f2        | Film 2       |   2010 |        10 |

## 5. Random Sample (30 pairs)
| pair_id   | film_title   |   year |   sent_idx | en_raw       | ta_raw      | en           | ta          | align_type   | qc_flags   |
|:----------|:-------------|-------:|-----------:|:-------------|:------------|:-------------|:------------|:-------------|:-----------|
| f1:1      | Test Film    |   2016 |          1 | Hello world. | வணக்கம் உலகம். | Hello world. | வணக்கம் உலகம். | 1-1          |            |
| f1:2      | Test Film    |   2016 |          2 | Hello world. | வணக்கம் உலகம். | Hello world. | வணக்கம் உலகம். | 1-1          |            |
| f1:3      | Test Film    |   2016 |          3 | Hello world. | வணக்கம் உலகம். | Hello world. | வணக்கம் உலகம். | 1-1          |            |
| f1:4      | Test Film    |   2016 |          4 | Hello world. | வணக்கம் உலகம். | Hello world. | வணக்கம் உலகம். | 1-1          |            |
| f1:5      | Test Film    |   2016 |          5 | Hello world. | வணக்கம் உலகம். | Hello world. | வணக்கம் உலகம். | 1-1          |            |
| f1:8      | Test Film    |   2016 |          8 | Hello world. | வணக்கம் உலகம். | Hello world. | வணக்கம் உலகம். | 1-1          |            |
| f1:9      | Test Film    |   2016 |          9 | Hello world. | வணக்கம் உலகம். | Hello world. | வணக்கம் உலகம். | 1-1          |            |
| f1:10     | Test Film    |   2016 |         10 | Hello world. | வணக்கம் உலகம். | Hello world. | வணக்கம் உலகம். | 1-1          |            |
| f1:11     | Test Film    |   2016 |         11 | Hello world. | வணக்கம் உலகம். | Hello world. | வணக்கம் உலகம். | 1-1          |            |
| f1:12     | Test Film    |   2016 |         12 | Hello world. | வணக்கம் உலகம். | Hello world. | வணக்கம் உலகம். | 1-1          |            |
| f1:13     | Test Film    |   2016 |         13 | Hello world. | வணக்கம் உலகம். | Hello world. | வணக்கம் உலகம். | 1-1          |            |
| f1:15     | Test Film    |   2016 |         15 | Hello world. | வணக்கம் உலகம். | Hello world. | வணக்கம் உலகம். | 1-1          |            |
| f1:17     | Test Film    |   2016 |         17 | Hello world. | வணக்கம் உலகம். | Hello world. | வணக்கம் உலகம். | 1-1          |            |
| f1:18     | Test Film    |   2016 |         18 | Hello world. | வணக்கம் உலகம். | Hello world. | வணக்கம் உலகம். | 1-1          |            |
| f1:19     | Test Film    |   2016 |         19 | Hello world. | வணக்கம் உலகம். | Hello world. | வணக்கம் உலகம். | 1-1          |            |
| f1:20     | Test Film    |   2016 |         20 | Hello world. | வணக்கம் உலகம். | Hello world. | வணக்கம் உலகம். | 1-1          |            |
| f1:21     | Test Film    |   2016 |         21 | Hello world. | வணக்கம் உலகம். | Hello world. | வணக்கம் உலகம். | 1-1          |            |
| f1:24     | Test Film    |   2016 |         24 | Hello world. | வணக்கம் உலகம். | Hello world. | வணக்கம் உலகம். | 1-1          |            |
| f1:25     | Test Film    |   2016 |         25 | Hello world. | வணக்கம் உலகம். | Hello world. | வணக்கம் உலகம். | 1-1          |            |
| f1:27     | Test Film    |   2016 |         27 | Hello world. | வணக்கம் உலகம். | Hello world. | வணக்கம் உலகம். | 1-1          |            |
| f1:29     | Test Film    |   2016 |         29 | Hello world. | வணக்கம் உலகம். | Hello world. | வணக்கம் உலகம். | 1-1          |            |
| f1:30     | Test Film    |   2016 |         30 | Hello world. | வணக்கம் உலகம். | Hello world. | வணக்கம் உலகம். | 1-1          |            |
| f2:31     | Film 2       |   2010 |         31 | Goodbye.     | விடை.       | Goodbye.     | விடை.       | 1-1          | CODE_MIXED |
| f2:33     | Film 2       |   2010 |         33 | Goodbye.     | விடை.       | Goodbye.     | விடை.       | 1-1          | CODE_MIXED |
| f2:34     | Film 2       |   2010 |         34 | Goodbye.     | விடை.       | Goodbye.     | விடை.       | 1-1          | CODE_MIXED |
| f2:35     | Film 2       |   2010 |         35 | Goodbye.     | விடை.       | Goodbye.     | விடை.       | 1-1          | CODE_MIXED |
| f2:36     | Film 2       |   2010 |         36 | Goodbye.     | விடை.       | Goodbye.     | விடை.       | 1-1          | CODE_MIXED |
| f2:37     | Film 2       |   2010 |         37 | Goodbye.     | விடை.       | Goodbye.     | விடை.       | 1-1          | CODE_MIXED |
| f2:39     | Film 2       |   2010 |         39 | Goodbye.     | விடை.       | Goodbye.     | விடை.       | 1-1          | CODE_MIXED |
| f2:40     | Film 2       |   2010 |         40 | Goodbye.     | விடை.       | Goodbye.     | விடை.       | 1-1          | CODE_MIXED |

## 6. Stop-Gate Status
🚫 **FAIL**: Only 40 usable pairs < threshold 20,000.
Options:
- Lower `qc.min_usable_pairs` in `configs/default.yaml`.
- Relax QC thresholds (e.g. `max_length_ratio`, `min_en_tokens`).
- Add supplementary SRT data with `python -m tyto data srt-align`.

---
*Generated by `python -m tyto data audit`.*
