class TestSpellCorrectPredictor(unittest.TestCase):
    def setUp(self):
        self.config = MagicMock()
        self.context_tracker = MagicMock()
        self.predictor_name = "spell_correct_predictor"
        self.short_desc = "Short description"
        self.long_desc = "Long description"
        self.logger = MagicMock()

        self.predictor = SpellCorrectPredictor(
            self.config,
            self.context_tracker,
            self.predictor_name,
            self.short_desc,
            self.long_desc,
            self.logger
        )

    def test_correction(self):
        self.assertEqual(self.predictor.correction("speling"), "spelling")

    def test_candidates(self):
        self.assertEqual(self.predictor.candidates("speling"), {"spelling"})

    def test_known(self):
        self.assertEqual(self.predictor.known(["spelling", "correct"]), {"spelling", "correct"})

    def test_edits1(self):