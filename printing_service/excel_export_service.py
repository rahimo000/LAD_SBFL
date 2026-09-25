"""Excel export for the dataset overall summary (XlsxWriter).

Produces a formatted workbook: the overall average table with winner cells
highlighted (lowest EXAM / highest Top-K per row, ties included) and a bar
chart below the table comparing metrics on the TOTAL average EXAM values.
"""
from typing import Dict, List

import scores_services as ss
from printing_service.dataset_printing_service import calculate_grand_average

EXAM_TYPES = [
    ("oexam", "oexam"),
    ("pexam", "pexam"),
    ("lex-exam", "lexical"),
    ("rev-exam", "reverse"),
    ("deltaexam", "delta"),
]

# Series shown in the bar chart (TOTAL average EXAM values).
CHART_SERIES = ["oexam", "pexam", "lex-exam", "rev-exam"]

WINNER_FILL = "#C6EFCE"  # light green


def _row_values(avg_map: Dict[str, Dict[str, float]], metrics: List[str], key: str) -> List[float]:
    """Same values as the console table (percentages)."""
    if key == "delta":
        return [(avg_map[m]['pexam'] - avg_map[m]['oexam']) * 100 for m in metrics]
    return [avg_map[m][key] * 100 for m in metrics]


def _row_kinds(all_types) -> Dict[str, str]:
    """Maps a row label to 'exam' (lower wins) or 'topk' (higher wins)."""
    exam_labels = {label for label, _ in EXAM_TYPES}
    return {label: ('exam' if label in exam_labels else 'topk') for label, _ in all_types}


def export_dataset_overall(output_path: str, dataset_name: str,
                            project_averages: Dict[str, Dict[str, Dict[str, float]]],
                            include_topk: bool = True) -> str:
    """Writes the dataset overall workbook. Returns the output path."""
    import xlsxwriter

    metrics = ss.get_all_metric_keys()
    topk_keys = ([f"l-Top{k}" for k in [1, 3, 5]] + [f"r-Top{k}" for k in [1, 3, 5]]) if include_topk else []
    all_types = EXAM_TYPES + [(k, k) for k in topk_keys]
    kinds = _row_kinds(all_types)
    projects = sorted(project_averages.keys())
    grand_avg = calculate_grand_average(project_averages, all_types)

    workbook = xlsxwriter.Workbook(output_path)
    ws = workbook.add_worksheet("Overall")

    header_fmt = workbook.add_format({'bold': True, 'bg_color': '#1F4E78', 'font_color': 'white'})
    total_fmt = workbook.add_format({'bold': True, 'bg_color': '#D9E1F2'})
    total_num_fmt = workbook.add_format({'bold': True, 'bg_color': '#D9E1F2', 'num_format': '0.0000'})
    num_fmt = workbook.add_format({'num_format': '0.0000'})
    winner_fmt = workbook.add_format({'bg_color': WINNER_FILL, 'num_format': '0.0000'})
    total_winner_fmt = workbook.add_format({'bold': True, 'bg_color': WINNER_FILL, 'num_format': '0.0000'})

    # Header
    ws.write_row(0, 0, ["Project", "Type"] + metrics, header_fmt)
    row = 1
    total_row_of_label = {}  # label -> excel row number (1-based) for chart series

    def write_block(first_col_label: str, avg_map, bold: bool):
        nonlocal row
        for idx, (label, key) in enumerate(all_types):
            values = _row_values(avg_map, metrics, key)
            if kinds[label] == 'exam':
                best = min(values)
            else:
                best = max(values)
            ws.write(row, 0, first_col_label if idx == 0 else "", total_fmt if bold else None)
            ws.write(row, 1, label, total_fmt if bold else None)
            for col, val in enumerate(values, start=2):
                is_winner = (val == best)
                if bold:
                    fmt = total_winner_fmt if is_winner else total_num_fmt
                else:
                    fmt = winner_fmt if is_winner else num_fmt
                ws.write_number(row, col, val, fmt)
            if bold:
                total_row_of_label[label] = row + 1  # 1-based for chart ranges
            row += 1

    for project in projects:
        write_block(project, project_averages[project], bold=False)
        row += 1  # blank separator line between projects
    write_block("TOTAL (AVG)", grand_avg, bold=True)

    last_row = row  # 0-based count of used rows
    ws.freeze_panes(1, 2)
    ws.set_column(0, 1, 14)
    ws.set_column(2, 1 + len(metrics), 10)

    # Bar chart below the table: TOTAL average EXAM values per metric.
    chart = workbook.add_chart({'type': 'column'})
    chart.set_title({'name': f'Average EXAM by metric — TOTAL ({dataset_name})'})
    chart.set_x_axis({'name': 'Metric'})
    chart.set_y_axis({'name': 'EXAM (%)'})
    chart.set_style(10)
    first_data_col = 2  # 0-based: 'C'
    last_data_col = 1 + len(metrics)
    for label in CHART_SERIES:
        excel_row = total_row_of_label[label]
        chart.add_series({
            'name': label,
            'categories': ["Overall", 0, first_data_col, 0, last_data_col],
            'values': ["Overall", excel_row - 1, first_data_col, excel_row - 1, last_data_col],
        })
    ws.insert_chart(last_row + 2, 0, chart, {'x_scale': 1.6, 'y_scale': 1.4})

    workbook.close()
    return output_path
