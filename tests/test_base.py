import unittest
import MDRefine

class Test(unittest.TestCase):
    def test1(self):
        # just test import
        import MDRefine
        self.assertIsNone(None)
        from MDRefine import data_loading, loss_and_minimizer, hyperminimizer, MDRefinement
    def test2(self):
        self.assertEqual(3,3)
    def test3(self):
        with self.assertRaises(TypeError):
            raise TypeError

if __name__ == "__main__":
    unittest.main()


