import unittest
from unittest.mock import MagicMock
from ConvAssist.predictor import Predictor
from ConvAssist.tests.utils import safe_delete_file, safe_check_folder

class TestPredictor(unittest.TestCase):
    def setUp(self):
        self.config = MagicMock()
        self.context_tracker = MagicMock()
        self.predictor_name = "test_predictor"
        self.short_desc = "Short description"
        self.long_desc = "Long description"
        self.db_path = "tests/test_data/dbs/test_predictor.db"
        safe_check_folder("tests/test_data/dbs/")

        self.logger = MagicMock()

        self.predictor = Predictor(
            self.config,
            self.context_tracker,
            self.predictor_name,
            self.short_desc,
            self.long_desc,
            self.logger
        )

    def tearDown(self) -> None:
        safe_delete_file(self.db_path)
        return super().tearDown()

    def test_get_name(self):
        self.assertEqual(self.predictor.predictor_name, "test_predictor")

    def test_get_short_description(self):
        self.assertEqual(self.predictor.short_description, "Short description")

    def test_get_long_description(self):
        self.assertEqual(self.predictor.long_description, "Long description")

    def test_create_table(self):
        tablename = "test_table"
        columns = ["column1", "column2"]

        try:
            self.predictor.createTable(self.db_path, tablename, columns)

        except Exception as e:
            self.fail(f"Failed to create table: {e}")
        

if __name__ == "__main__":
    unittest.main()