import unittest
import MDRefine

class Test(unittest.TestCase):
    def test1(self):
        self.assertIsNone(None)
    def test2(self):
        self.assertEqual(3,3)
    def test3(self):
        with self.assertRaises(TypeError):
            raise TypeError

if __name__ == "__main__":
    unittest.main()


