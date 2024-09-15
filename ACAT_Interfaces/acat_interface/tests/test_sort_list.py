import unittest
from ACAT_ConvAssist_Interface.ConvAssistCPApp.ACATConvAssistInterface import ACATConvAssistInterface

class TestACATConvAssistInterface(unittest.TestCase):
    
    def test_sort_List_empty(self):
        result = ACATConvAssistInterface.sort_List([], 5)
        self.assertEqual(result, [])

    def test_sort_List_less_than_amount(self):
        prediction_list = [("a", 0.1), ("b", 0.2)]
        result = ACATConvAssistInterface.sort_List(prediction_list, 5)
        self.assertEqual(result, [("b", 0.2), ("a", 0.1)])

    def test_sort_List_more_than_amount(self):
        prediction_list = [("a", 0.1), ("b", 0.2), ("c", 0.3), ("d", 0.4), ("e", 0.5)]
        result = ACATConvAssistInterface.sort_List(prediction_list, 3)
        self.assertEqual(result, [("e", 0.5), ("d", 0.4), ("c", 0.3)])

    def test_sort_List_exact_amount(self):
        prediction_list = [("a", 0.1), ("b", 0.2), ("c", 0.3)]
        result = ACATConvAssistInterface.sort_List(prediction_list, 3)
        self.assertEqual(result, [("c", 0.3), ("b", 0.2), ("a", 0.1)])

    def test_sort_List_unsorted_input(self):
        prediction_list = [("a", 0.3), ("b", 0.1), ("c", 0.2)]
        result = ACATConvAssistInterface.sort_List(prediction_list, 3)
        self.assertEqual(result, [("a", 0.3), ("c", 0.2), ("b", 0.1)])

    def test_sort_List_unsorted_input_no_amount(self):
        prediction_list = [("a", 0.3), ("b", 0.1), ("c", 0.2)]
        result = ACATConvAssistInterface.sort_List(prediction_list, 0)
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()