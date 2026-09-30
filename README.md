# Lad_Instructions: Optimized SBFL Evaluation Tool

**Lad_Instructions** is a high-performance CLI tool for evaluating **Spectrum-Based Fault Localization (SBFL)** techniques. It processes software coverage data to calculate suspiciousness scores and evaluates them using rank-based metrics (EXAM scores and Top-K).

The tool is optimized for **Big Data** using NumPy vectorization and multiprocessing, ensuring it can handle large datasets (like ISSTA13 and Siemens) with minimal memory footprint and maximum speed.

## Key Features

- **Vectorized SBFL Metrics:** Optimized implementations of Tarantula, Ochiai, Jaccard, and GP13.
- **Aggregation Techniques:** Combined scoring using Approval Voting (APV) and Majority Judgment (MJ).
- **Comprehensive Evaluation:** Calculates Optimistic (`oexam`), Pessimistic (`pexam`), Lexical (`lex-exam`), and Reverse-Lexical (`rev-exam`) scores.
- **Success-at-K (Top-K):** Supports Top-1, Top-2, and Top-3 evaluations with multiple tie-breaking strategies.
- **Multiprocessing:** Parallel processing of project versions for rapid evaluation.
- **Scalable Architecture:** Centralized metric registry and JSON-based configuration.

---

## Installation & Setup

### Prerequisites
- Python 3.8+
- NumPy (`pip install numpy`)
- `questionary` and `rich` for the interactive TUI (`pip install questionary rich`)
- `XlsxWriter` for the formatted Excel export (`pip install XlsxWriter`)

### Configuration
Dataset paths and project lists are managed in `config.json`. Ensure the `base_dir` paths match your local environment:
```json
{
    "issta13": {
        "base_dir": "C:\\AHPdatasets",
        "sub_dir": "1-fault\\Datasets",
        "projects": ["eventbus", "daikon", ...]
    }
}
```

---

## Interactive TUI

Don't want to memorize commands? Just run the app with no arguments:

```bash
python main.py
```

An arrow-key menu guides you through dataset → project → version selection,
report options, and optional CSV export. Every run shows the equivalent CLI
command, so you can learn (or script) it as you go. The classic CLI below
keeps working unchanged.

---

## Command Reference

The tool uses a sub-command structure: `version`, `project`, or `dataset`.

### 1. Version Service
Used to analyze a single version of a project.

| Command Option | Description | Example |
| :--- | :--- | :--- |
| `-examscore` | Print EXAM scores and Top-K results for the version. | `python main.py version -examscore issta13 eventbus v1` |
| `-scores` | Print a detailed table of scores for every instruction. | `python main.py version -scores issta13 eventbus v1` |

### 2. Project Service
Used to analyze all versions within a single project.

| Command Option | Description | Example |
| :--- | :--- | :--- |
| `-examscore` | Print a consolidated table showing results for every version. | `python main.py project -examscore issta13 eventbus` |
| `-overall` | Print the overall average EXAM and Top-K scores for the project. | `python main.py project -overall issta13 eventbus` |

### 3. Dataset Service
Used to analyze every project within a dataset.

| Command Option | Description | Example |
| :--- | :--- | :--- |
| `-overall` | Print a high-level summary of averages for every project in the dataset. | `python main.py dataset -overall issta13` |

### Selecting score metrics & evaluations

Calculation, tables, and the chart all follow your selection (omit a flag
to select everything):

- `--metrics tar,och,jac` limits the score metrics (any of `tar och jac gp
  op2 kul2 zol amp mj apv`). `mj`/`apv` aggregate only the selected base metrics.
- `--evals oexam,pexam` limits the evaluations (any of `oexam pexam lex-exam
  rev-exam` plus `l/r-Top1/3/5`). `deltaexam` appears automatically when
  `oexam` and `pexam` are both selected. Exam-only example:
  `python main.py project -overall --evals oexam,pexam,lex-exam,rev-exam issta13 eventbus`.

### Excel export

- `-o` routes by extension: `.csv` captures the console tables (all services),
  while `.xlsx` writes a formatted workbook — supported for
  `dataset -overall` only. The workbook highlights the winner cell(s) in every
  row (lowest EXAM, highest Top-K) and adds a bar chart below the table with
  one series per selected evaluation (TOTAL averages, `deltaexam` included).
  Note: Top-K averages are 0-100 scaled counts while EXAM is a
  percentage, so mixed selections share one axis:
  `python main.py dataset -overall issta13 -o results.xlsx`.

### Exported files

Every export is saved under the `output/` folder (auto-created), keeping any
relative structure you type: `-o issta/out.csv` → `output/issta/out.csv`,
`-o quick.csv` → `output/quick.csv`. Paths that would escape `output/`
(absolute paths, `..`) are clamped inside it. The folder is git-ignored.

---

## SBFL Metrics & Ordering
The tool maintains a strict metric ordering for all reports:
1. **tar** (Tarantula)
2. **och** (Ochiai)
3. **jac** (Jaccard)
4. **gp** (GP13)
5. **op2** (Op2, normalized to [0,1])
6. **kul2** (Kulczynski2)
7. **zol** (Zoltar)
8. **amp** (Ample)
9. **mj** (Majority Judgment)
10. **apv** (Approval Voting)

## Testing
To verify the evaluation logic, run the included unit tests:
```bash
python -m unittest evaluation_services/test_exam_score.py
```

## Optimization Trace
Progress on performance enhancements is tracked in [optimization_tasks.md](./optimization_tasks.md).
