#!/usr/bin/python
import unittest
import warnings
import cProfile
import pstats
import time
import StringIO
import numpy as np
from dragonet.basecall.common import stub


class TestStub(unittest.TestCase):

    def setUp(self):
        return

    def tearDown(self):
        return

    def test_001_list_to_vector(self):
        data = [1.0, 2.0, -3.5, 7.6]
        newdata = stub.check_vector(data)
        self.assertEqual(data, newdata)
        return
    
    def test_002_list_to_matrix(self):
        data = [[1.0, 2.0, -3.5],
                [7.6, 1.3, 12.8]]
        newdata = stub.check_matrix(data)
        self.assertEqual(data, newdata)
        return
    
    def test_003_dict_to_map(self):
        data = {'foo': 1, 'bar': 2, 'kill': 3, 'me': 4}
        newdata = stub.check_map(data)
        self.assertEqual(data, newdata)
        return

    def test_004_contiguous(self):
        data1d = np.array([1, 5, 7, 3, 8, 4, 10, 4, 12, 2], dtype=np.int32)
        data1d2 = np.zeros(10, dtype=np.float64)
        data1d2[:] = data1d * 0.5
        data2d = np.ndarray((2, 5), buffer=(data1d * 0.5), dtype=np.float64)
        new1d = stub.check_np_vector(data1d, 'int32')
        np.testing.assert_equal(data1d, new1d)
        new1d2 = stub.check_np_vector(data1d2, 'float64')
        np.testing.assert_equal(data1d2, new1d2)
        new2d = stub.check_np_matrix(data2d)
        np.testing.assert_equal(data2d, new2d)
        return
    
    def test_005_noncontiguous(self):
        data1d = np.array([1, 5, 7, 3, 8, 4, 10, 4, 12, 2], dtype=np.int32)
        data2d = np.ndarray((2, 5), buffer=(data1d * 0.5), dtype=np.float64)
        data1d = data1d[::2]
        data2d = data2d[::-1, ::2]
        new1d = stub.check_np_vector(data1d, 'int32')
        np.testing.assert_equal(data1d, new1d)
        new2d = stub.check_np_matrix(data2d)
        np.testing.assert_equal(data2d, new2d)
        return
    
    def test_005_record_array(self):
        desc = np.dtype({'names': ['a', 'b', 'c'], 'formats': [np.int32, np.float64, np.int64]}, align=True)
        data = np.zeros(10, dtype=desc)
        data1 = data['a']
        data2 = data['b']
        data3 = data['c']
        data1[:] = [1, 5, 7, 3, 8, 4, 10, 4, 12, 2]
        data2[:] = data1 * 0.5
        data3[:] = data1 * 2
        new1 = stub.check_np_vector(data1, 'int32')
        np.testing.assert_equal(data1, new1)
        new2 = stub.check_np_vector(data2, 'float64')
        np.testing.assert_equal(data2, new2)
        new3 = stub.check_np_vector(data3, 'int64')
        np.testing.assert_equal(data3, new3)
        return

    @unittest.skip('Not needed')
    def test_006_accumulate(self):
        data = np.arange(0.0, 10.0, dtype=np.float32)
        result = stub.check_accumulator(data)
        self.assertEqual(result, 45.0)
        result = stub.check_accumulator(data[:-1])
        self.assertEqual(result, 36.0)
        data = np.empty(1024, dtype=np.float32)
        data[:] = np.random.random(1024)
        
        #print 'Fast version'
        pr = cProfile.Profile()
        pr.enable()
        result1 = self._loop_checker(data, 1000000, True)
        pr.disable()
        s = StringIO.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        #ps.print_stats()
        #print s.getvalue()

        #print 'Normal version'
        pr = cProfile.Profile()
        pr.enable()
        result2 = self._loop_checker(data, 1000000, False)
        pr.disable()
        s = StringIO.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        #ps.print_stats()
        #print s.getvalue()
        
        return

    def test_007_check_exp(self):
        warnings.filterwarnings("error")
        raw = np.arange(-80.0, 80.0, 1e-4, dtype=np.float32)
        data1 = np.empty(raw.size, dtype=np.float32)
        data2 = np.empty(raw.size, dtype=np.float32)
        data1[:] = raw
        data2[:] = raw
        #print 'Total count is:', raw.size
        t1 = time.clock()
        stub.check_exp(data1, False)
        t2 = time.clock()
        stub.check_exp(data2, True)
        t3 = time.clock()
        #print 'Stats for positive range.'
        #print 'Time for normal exponential:', t2 - t1
        #print 'Time for fast exponential:', t3 - t2
        error = np.abs((data2 - data1) / data1)
        max_idx = np.argmax(error)
        #print 'Max error:', error[max_idx], 'raw = ', raw[max_idx], 'data1 =', data1[max_idx], 'data2 =', data2[max_idx], 'diff =', data2[max_idx] - data1[max_idx]
        data3 = -1.0 * raw
        data4 = -1.0 * raw
        t1 = time.clock()
        stub.check_exp(data3, False)
        t2 = time.clock()
        stub.check_exp(data4, True)
        t3 = time.clock()
        #print 'Stats for negative range.'
        #print 'Time for normal exponential:', t2 - t1
        #print 'Time for fast exponential:', t3 - t2
        error = np.abs((data4 - data3) / data3)
        max_idx = np.argmax(error)
        #print 'Max error:', error[max_idx], 'raw =', raw[max_idx], 'data1 =', data3[max_idx], 'data2 =', data4[max_idx], 'diff =', data4[max_idx] - data3[max_idx]
        return


    def _loop_checker(self, data, n, fast):
        return stub.check_accumulator_loop(data, n, fast)


if __name__ == '__main__':
    unittest.main()
