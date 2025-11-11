# General unit testing examples for a basic class. This class could be part of another
# module. If so, it must be imported first.

# unit tests https://docs.python.org/3/library/unittest.html

import unittest

class Test_Conditinosaur(unittest.TestCase):
    def test_attributes(self): # must start with word 'test_'
        self.assertEquals(class1.attribute1, 'Expectation 1')
        self.assertEquals(class1.attribute2, 'Expectation 2')

    def test_zero(self):
        with self.assertRaises(ValueError):
            calc.divide(10, 0)
            module.function

    def test_check_equalities(self):
        cond1 = Class('one', 'two', 'Expectation')


if __name__ == '__main__':
    unittest.main()



class enemy1(Health, MovePattern, Attacks):
    def __init__(self, name, status):
        self.name = name
        self.status = status
    
    def __str__(self):
        print(self.name, self.status)
        
'''