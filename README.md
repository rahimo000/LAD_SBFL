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

---

## SBFL Metrics & Ordering
The tool maintains a strict metric ordering for all reports:
1. **tar** (Tarantula)
2. **och** (Ochiai)
3. **jac** (Jaccard)
4. **gp** (GP13)
5. **mj** (Majority Judgment)
6. **apv** (Approval Voting)

## Testing
To verify the evaluation logic, run the included unit tests:
```bash
python -m unittest evaluation_services/test_exam_score.py
```

## Optimization Trace
Progress on performance enhancements is tracked in [optimization_tasks.md](./optimization_tasks.md).
