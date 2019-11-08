import unittest
import RWA as rwa
import Topology

class MyTestCase(unittest.TestCase):
    def __init__(self):
        super().__init__()
        self.RWA = rwa.RWA()
        self.topology = Topology.Topology("nsf")
        self.topology.init_nsf()
        self.graph = self.topology.create_ACMN(14, 21, 0.7, "test")

    def test_something(self):
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
