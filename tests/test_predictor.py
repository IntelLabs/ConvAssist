import unittest
from unittest.mock import MagicMock
from ConvAssist.predictor import Predictor

class TestPredictor(unittest.TestCase):
    def setUp(self):
        self.config = MagicMock()
        self.context_tracker = MagicMock()
        self.predictor_name = "test_predictor"
        self.short_desc = "Short description"
        self.long_desc = "Long description"
        self.logger = MagicMock()

        self.predictor = Predictor(
            self.config,
            self.context_tracker,
            self.predictor_name,
            self.short_desc,
            self.long_desc,
            self.logger
        )

    def test_get_name(self):
        self.assertEqual(self.predictor.get_name(), "test_predictor")

    def test_get_description(self):
        self.assertEqual(self.predictor.get_description(), "Long description")

    def test_get_long_description(self):
        self.assertEqual(self.predictor.get_long_description(), "Long description")

    def test_predict(self):
        with self.assertRaises(NotImplementedError):
            self.predictor.predict()

    def test_learn(self):
        with self.assertRaises(NotImplementedError):
            self.predictor.learn()

    def test_read_config(self):
        with self.assertRaises(NotImplementedError):
            self.predictor._read_config()

    def test_create_table(self):
        dbname = "test_db"
        tablename = "test_table"
        columns = ["column1", "column2"]

        self.predictor.log = MagicMock()
        self.predictor.createTable(dbname, tablename, columns)

        self.predictor.log.critical.assert_not_called()

if __name__ == "__main__":
    unittest.main()