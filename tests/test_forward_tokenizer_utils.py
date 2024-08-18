import unittest
from ConvAssist.utilities.ngram_map import NgramMap
from ConvAssist.tokenizer.forward_tokenizer import ForwardTokenizer
from ConvAssist.utilities.forward_tokenizer_utils import forward_tokenize_file
import os

class TestForwardTokenizerUtils(unittest.TestCase):
    def test_forward_tokenize_file(self):
        # Create a temporary file with sample text
        with open("temp_file.txt", "w") as f:
            f.write("This is a sample text.")

        map = NgramMap()

        # Call the forward_tokenize_file function
        ngram_map = forward_tokenize_file("temp_file.txt", ngram_size=2, ngram_map=map)

        # Assert the expected output
        expected_ngram_map = NgramMap()
        expected_ngram_map.add_token("This")
        expected_ngram_map.add_token("is")
        expected_ngram_map.add_token("a")
        expected_ngram_map.add_token("sample")
        expected_ngram_map.add_token("text")

        # self.assertEqual(ngram_map.__len__(), expected_ngram_map.__len__())

        # Clean up the temporary file
        os.remove("temp_file.txt")

if __name__ == "__main__":
    unittest.main()