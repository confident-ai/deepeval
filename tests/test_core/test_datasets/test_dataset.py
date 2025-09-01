import pytest
import os
from deepeval.dataset import EvaluationDataset


class TestSaveAndLoad:
    def test_dataset_save_load_goldens(self):
        """Loads Goldens from csv and json files and asserts their length to be equal to 15 (the number of goldens in the files)"""

        json_file = "goldens.json"
        csv_file = "goldens.csv"
        current_dir = os.path.dirname(os.path.abspath(__file__))

        json_path = os.path.join(current_dir, json_file)
        csv_path = os.path.join(current_dir, csv_file)

        dataset1 = EvaluationDataset()
        dataset2 = EvaluationDataset()
        dataset1.add_goldens_from_json_file(file_path=json_path)
        dataset2.add_goldens_from_csv_file(file_path=csv_path)

        assert (
            len(dataset1.goldens) == len(dataset2.goldens) == 15
        )

    def test_dataset_save_load_conversational_goldens(self):
        """Loads ConversationalGoldens from csv and json files and asserts their length to be equal to 15 (the number of goldens in the files)"""

        json_file = "convo_goldens.json"
        csv_file = "convo_goldens.csv"
        current_dir = os.path.dirname(os.path.abspath(__file__))

        json_path = os.path.join(current_dir, json_file)
        csv_path = os.path.join(current_dir, csv_file)

        dataset1 = EvaluationDataset()
        dataset2 = EvaluationDataset()
        dataset1.add_goldens_from_json_file(file_path=json_path)
        dataset2.add_goldens_from_csv_file(file_path=csv_path)

        assert (
            len(dataset1.goldens)
            == len(dataset2.goldens)
            == 15
        )