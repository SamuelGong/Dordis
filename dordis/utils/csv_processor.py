import csv
import os
from typing import List


def initialize_csv(result_csv_file: str, recorded_items: List,
                   result_dir: str) -> None:
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    with open(result_csv_file, 'w', newline='') as result_file:
        result_writer = csv.writer(result_file)
        first_row = recorded_items
        result_writer.writerow(first_row)


def write_csv(result_csv_file: str, new_row: List) -> None:
    with open(result_csv_file, 'a') as result_file:
        result_writer = csv.writer(result_file)
        result_writer.writerow(new_row)
