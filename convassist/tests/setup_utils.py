import os
import shutil
import unittest

import backpedal

SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))


def find_resources(dir_name):
    return backpedal.find(dir_name, path=SOURCE_DIR, direction="up", item_type="directory")


def setup_static_resources():
    if not os.path.exists(f"{SOURCE_DIR}/test_data/static"):
        os.makedirs(f"{SOURCE_DIR}/test_data/static")

    with open(f"{SOURCE_DIR}/test_data/static/stopwords.txt", "w") as f:
        f.write("a\nthe\nan\n")

    with open(f"{SOURCE_DIR}/test_data/static/filter_words.txt", "w") as f:
        f.write("filter\n")

    third_party_files = find_resources("3rd_party_resources")
    if third_party_files:
        os.system(f"ln -s {third_party_files}/aac_gpt2 {SOURCE_DIR}/test_data/static")
        os.system(
            f"ln -s {third_party_files}/multi-qa-MiniLM-L6-cos-v1 {SOURCE_DIR}/test_data/static"
        )
        os.system(f"ln -s {third_party_files}/aac_dataset/* {SOURCE_DIR}/test_data/static")
        os.system(f"ln -s {third_party_files}/daily_dialog/* {SOURCE_DIR}/test_data/static")


def teardown_static_resources():
    if os.path.exists(f"{SOURCE_DIR}/test_data/static"):
        shutil.rmtree(f"{SOURCE_DIR}/test_data/static")


def setup_personalized_resources():
    if not os.path.exists(f"{SOURCE_DIR}/test_data/personalized"):
        os.makedirs(f"{SOURCE_DIR}/test_data/personalized")

    shutil.copy(
        f"{SOURCE_DIR}/test_data/token_data.txt",
        f"{SOURCE_DIR}/test_data/personalized/startSentences.txt",
    )
    shutil.copy(
        f"{SOURCE_DIR}/test_data/token_data.txt",
        f"{SOURCE_DIR}/test_data/personalized/personalizedCannedPhrases.txt",
    )


def teardown_personalized_resources():
    if os.path.exists(f"{SOURCE_DIR}/test_data/personalized"):
        shutil.rmtree(f"{SOURCE_DIR}/test_data/personalized")


class testSetupAndTeardown(unittest.TestCase):
    def test_setup_static_resources(self):
        setup_static_resources()
        self.assertTrue(os.path.exists(f"{SOURCE_DIR}/test_data/static"))

    def test_teardown_static_resources(self):
        teardown_static_resources()
        self.assertFalse(os.path.exists(f"{SOURCE_DIR}/test_data/static"))

    def test_setup_personalized_resources(self):
        setup_personalized_resources()
        self.assertTrue(os.path.exists(f"{SOURCE_DIR}/test_data/personalized"))

    def test_teardown_personalized_resources(self):
        teardown_personalized_resources()
        self.assertFalse(os.path.exists(f"{SOURCE_DIR}/test_data/personalized"))
