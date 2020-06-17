#!/usr/bin/env python
import unittest

from reinforce import calc_qvalues


class TestQValues(unittest.TestCase):

    def test_calcqvalues(self):
        r = [0., 0., 0., 0., 1.]

        qvalues = calc_qvalues(r, gamma=.5)

        self.assertEqual(qvalues[0], 1/16.)  # gamma=1/2: (1/2^4)

    def test_calcqvalues1(self):
        """results taken from lapan's book's example reinforce cartpole chapter9 w/o baseline"""
        r = [1.]*29

        qvalues = calc_qvalues(r, gamma=.99)

        self.assertEqual(qvalues, [25.282790566840355, 24.528071279636723, 23.76572856528962, 22.995685419484467, 22.21786406008532, 21.4321859192781, 20.638571635634445, 19.8369410460954, 19.027213177874142, 18.209306240276913, 17.383137616441328, 16.54862385499124, 15.705680661607312,
                                   14.854222890512437, 13.994164535871148, 13.12541872310217, 12.247897700103202, 11.361512828387072, 10.466174574128356, 9.561792499119552, 8.64827525163591, 7.72553055720799, 6.793465209301, 5.8519850599, 4.90099501, 3.9403989999999998, 2.9701, 1.99, 1.0])  # gamma=1/2: (1/2^4)


if __name__ == '__main__':
    unittest.main()
