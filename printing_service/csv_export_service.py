import csv
import sys
import io
from contextlib import contextmanager

import fileManagment as fm

@contextmanager
def capture_to_csv(file_path):
    """
    Captures stdout, parses the custom table format (using | as separator),
    and saves it to a proper CSV file under the output/ folder.
    """
    if not file_path:
        yield
        return

    # Capture stdout
    new_stdout = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = new_stdout
    
    try:
        yield
    finally:
        # Restore stdout
        sys.stdout = old_stdout
        output = new_stdout.getvalue()
        
        # Print to console as usual
        print(output)
        
        # Save to CSV (always under output/)
        try:
            real_path = fm.prepare_export_path(file_path)
            with open(real_path, 'w', newline='') as f:
                writer = csv.writer(f)
                lines = output.split('\n')
                for line in lines:
                    # Skip empty lines or divider lines
                    if not line.strip() or set(line.strip()) <= {'-', ' ', '|', '+'}:
                        continue
                    
                    # Split by | and strip whitespace from each cell
                    row = [cell.strip() for cell in line.split('|')]
                    # Filter out empty strings that might result from trailing |
                    row = [r for r in row if r]
                    if row:
                        writer.writerow(row)
            print(f"Exported results to {real_path}")
        except Exception as e:
            print(f"Error exporting to CSV: {e}")
