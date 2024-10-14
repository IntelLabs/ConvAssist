import os
import tempfile
import unittest

from convassist.predictor.utilities.canned_data import cannedData


class TestCannedData(unittest.TestCase):
    def setUp(self) -> None:
    
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tempdir.name, "test.db")
        self.canned_data_path = os.path.join(self.tempdir.name, "canned_data.txt")


    def test_canned_data_init_creates_db(self):
        with open(self.canned_data_path, "w") as f:
            f.write("Hello World\n")

        canned_data_instance = cannedData(self.db_path, self.canned_data_path)
        assert os.path.exists(self.db_path)

        sentences = canned_data_instance.all_phrases_as_dict()
        assert len(sentences) == 1
        self.assertDictEqual({"Hello World": 1}, sentences)

    def test_canned_data_init_updates_db(self):
        with open(self.canned_data_path, "w") as f:
            f.write("Hello World\n")

        canned_data_instance = cannedData(self.db_path, self.canned_data_path)
        sentences = canned_data_instance.all_phrases_as_dict()
        assert len(sentences) == 1
        self.assertDictEqual({"Hello World": 1}, sentences)

        canned_data_instance = None

        with open(self.canned_data_path, "a") as f:
            f.write("Hello Universe\n")

        canned_data_instance = cannedData(self.db_path, self.canned_data_path)

        sentences = canned_data_instance.all_phrases_as_dict()
        assert len(sentences) == 2
        self.assertDictEqual({"Hello World": 1, "Hello Universe": 1}, sentences)

    def test_canned_data_learn(self):
        with open(self.canned_data_path, "w") as f:
            f.write("Hello World\n")

        canned_data_instance = cannedData(self.db_path, self.canned_data_path)
        sentences = canned_data_instance.all_phrases_as_dict()
        assert len(sentences) == 1
        self.assertDictEqual({"Hello World": 1}, sentences)

        canned_data_instance.learn("Hello World")
        sentences = canned_data_instance.all_phrases_as_dict()
        assert len(sentences) == 1
        self.assertDictEqual({"Hello World": 2}, sentences)

        canned_data_instance.learn("Hello Universe")
        sentences = canned_data_instance.all_phrases_as_dict()
        assert len(sentences) == 2
        self.assertDictEqual({"Hello World": 2, "Hello Universe": 1}, sentences)

    def test_canned_data_retrieve(self):
        with open(self.canned_data_path, "w") as f:
            f.write("Hello World\n")

        canned_data_instance = cannedData(self.db_path, self.canned_data_path)
        sentences = canned_data_instance.all_phrases_as_dict()
        assert len(sentences) == 1
        self.assertDictEqual({"Hello World": 1}, sentences)

        result = canned_data_instance.retrieve("Hello World")
        self.assertDictEqual({"Hello World": 1}, result)

        result = canned_data_instance.retrieve("Hello Universe")
        self.assertDictEqual({}, result)

    def test_canned_data_remove(self):
        with open(self.canned_data_path, "w") as f:
            f.write("Hello World\nHello Universe\n")

        canned_data_instance = cannedData(self.db_path, self.canned_data_path)
        sentences = canned_data_instance.all_phrases_as_dict()
        assert len(sentences) == 2
        self.assertDictEqual({"Hello World": 1, "Hello Universe": 1}, sentences)

        canned_data_instance.remove("Hello World")
        sentences = canned_data_instance.all_phrases_as_dict()
        assert len(sentences) == 1
        self.assertDictEqual({"Hello Universe": 1}, sentences)

        canned_data_instance.remove("Hello Universe")
        sentences = canned_data_instance.all_phrases_as_dict()
        assert len(sentences) == 0
        self.assertDictEqual({}, sentences)

    def test_update_with_remove(self):
        with open(self.canned_data_path, "w") as f:
            f.write("Hello World\nHello Universe\n")

        canned_data_instance = cannedData(self.db_path, self.canned_data_path)
        sentences = canned_data_instance.all_phrases_as_dict()
        assert len(sentences) == 2
        self.assertDictEqual({"Hello World": 1, "Hello Universe": 1}, sentences)

        with open(self.canned_data_path, "w") as f:
            f.write("Hello World\n")

        canned_data_instance = cannedData(self.db_path, self.canned_data_path)
        sentences = canned_data_instance.all_phrases_as_dict()
        assert len(sentences) == 1
        self.assertDictEqual({"Hello World": 1}, sentences)
