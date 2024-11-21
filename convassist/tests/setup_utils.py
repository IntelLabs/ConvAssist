# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

import os
import shutil
import unittest

import backpedal

SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))


def find_resources(dir_name):
    return backpedal.find(dir_name, path=SOURCE_DIR, direction="up", item_type="directory")


STATIC_DIR = find_resources("3rd_party_resources")


def copy_static_resources():
    if not os.path.exists(f"{SOURCE_DIR}/test_data/static"):
        os.makedirs(f"{SOURCE_DIR}/test_data/static")

    third_party_files = find_resources("3rd_party_resources")

    try:
        assert third_party_files
        # shutil.copytree(
        #     f"{third_party_files}/aac_gpt2",
        #     f"{SOURCE_DIR}/test_data/static/aac_gpt2",
        #     dirs_exist_ok=True,
        # )

        # shutil.copytree(
        #     f"{third_party_files}/multi-qa-MiniLM-L6-cos-v1",
        #     f"{SOURCE_DIR}/test_data/static/multi-qa-MiniLM-L6-cos-v1",
        #     dirs_exist_ok=True,
        # )

        shutil.copytree(
            f"{third_party_files}/aac_dataset/",
            f"{SOURCE_DIR}/test_data/static",
            dirs_exist_ok=True,
        )
        shutil.copytree(
            f"{third_party_files}/daily_dialog/",
            f"{SOURCE_DIR}/test_data/static",
            dirs_exist_ok=True,
        )
    except Exception as e:
        print(f"Error copying files: {e}")


def remove_static_resources():
    remove_directory(f"{SOURCE_DIR}/test_data/static")


def setup_static_resources():
    teardown_static_resources()

    with open(f"{SOURCE_DIR}/test_data/static/stopwords.txt", "w") as f:
        f.write("a\nthe\nan\n")

    with open(f"{SOURCE_DIR}/test_data/static/filter_words.txt", "w") as f:
        f.write("filter\n")

    # with open(f"{SOURCE_DIR}/test_data/static/all_aac.txt", "w") as f:
    #     f.write("all\nyour\nbases\nare\nmine\n")


def remove_directory(directory):
    if os.path.exists(directory):

        def onerror(func, path, exc_info):
            import stat

            if not os.access(path, os.W_OK):
                os.chmod(path, stat.S_IWUSR)
                func(path)
            else:
                raise

        shutil.rmtree(directory, onexc=onerror, ignore_errors=True)


def teardown_static_resources():
    if os.path.exists(f"{SOURCE_DIR}/test_data/static/stopwords.txt"):
        os.remove(f"{SOURCE_DIR}/test_data/static/stopwords.txt")

    if os.path.exists(f"{SOURCE_DIR}/test_data/static/filter_words.txt"):
        os.remove(f"{SOURCE_DIR}/test_data/static/filter_words.txt")


def setup_personalized_resources():
    teardown_personalized_resources()

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
    shutil.copy(
        f"{SOURCE_DIR}/test_data/start_words.json",
        f"{SOURCE_DIR}/test_data/personalized/startWords.json",
    )


def teardown_personalized_resources():
    remove_directory(f"{SOURCE_DIR}/test_data/personalized")


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


if __name__ == "__main__":
    unittest.main()
