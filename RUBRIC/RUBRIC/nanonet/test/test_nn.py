import unittest

from RUBRIC import nanonet as nn
import numpy as np

class ANNTest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        print '* ANN'
        np.random.seed(0xdeadbeef)
        self.prec = 6
        self._NSTEP = 10
        self._NFEATURES = 4
        self._SIZE = 5

        self.W = np.random.normal(size=(self._NFEATURES, self._SIZE))
        self.b = np.random.normal(size=self._SIZE)
        self.x = np.random.normal(size=(self._NSTEP, self._NFEATURES))
        self.res = (self.x.dot(self.W) + self.b).astype(nn.dtype)

    def test_000_single_layer_linear(self):
        network = nn.FeedForward(self.W, self.b, nn.linear)
        self.assertEqual(network.in_size, self._NFEATURES)
        self.assertEqual(network.out_size, self._SIZE)
        np.testing.assert_array_almost_equal(network.run(self.x), self.res)

    def test_001_single_layer_tanh(self):
        network = nn.FeedForward(self.W, self.b, nn.tanh)
        self.assertEqual(network.in_size, self._NFEATURES)
        self.assertEqual(network.out_size, self._SIZE)
        np.testing.assert_array_almost_equal(network.run(self.x), np.tanh(self.res))

    def test_002_parallel_layers(self):
        l1 = nn.FeedForward(self.W, self.b, nn.tanh)
        l2 = nn.FeedForward(self.W, self.b, nn.tanh)
        network = nn.Parallel([l1, l2])
        self.assertEqual(network.in_size, self._NFEATURES)
        self.assertEqual(network.out_size, 2 * self._SIZE)

        res = network.run(self.x)
        np.testing.assert_array_equal(res[:,:self._SIZE], res[:,self._SIZE:])

    def test_003_simple_serial(self):
        W2 = np.random.normal(size=(self._SIZE, self._SIZE))
        res = self.x.dot(self.W).dot(W2)

        l1 = nn.FeedForward(self.W, fun=nn.linear)
        l2 = nn.FeedForward(W2, fun=nn.linear)
        network = nn.Serial([l1, l2])
        self.assertEqual(network.in_size, self._NFEATURES)
        self.assertEqual(network.out_size, self._SIZE)

        np.testing.assert_array_almost_equal(network.run(self.x), res)

    def test_004_reverse(self):
        network1 = nn.FeedForward(self.W, self.b, nn.tanh)
        res1 = network1.run(self.x)
        network2 = nn.Reverse(network1)
        res2 = network2.run(self.x)
        self.assertEqual(network1.in_size, network2.in_size,)
        self.assertEqual(network1.out_size, network2.out_size)

        np.testing.assert_array_equal(res1, res2, self.prec)

    def test_005_poormans_birnn(self):
        layer1 = nn.FeedForward(self.W, self.b, nn.tanh)
        layer2 = nn.FeedForward(self.W, self.b, nn.tanh)
        network = nn.BiRNN(layer1, layer2)

        res = network.run(self.x)
        np.testing.assert_array_equal(res[:,:self._SIZE], res[:,self._SIZE:], self.prec)

    def test_006_softmax(self):
        network = nn.SoftMax(self.W, self.b)

        res = network.run(self.x)
        res_sum = res.sum(axis=1)
        self.assertTrue(np.allclose(res_sum, 1.0))

    def test_007_rnn_no_state(self):
        W1 = np.vstack((np.zeros((self._SIZE, self._SIZE)), self.W))
        network = nn.SimpleRNN(W1, b=self.b, fun=nn.linear)

        res = network.run(self.x)
        np.testing.assert_almost_equal(res, self.res, self.prec)

    def test_008_rnn_no_input(self):
        W1 = np.random.normal(size=(self._SIZE, self._SIZE))
        W2 = np.vstack((W1, np.zeros((self._NFEATURES, self._SIZE))))
        network = nn.SimpleRNN(W2, fun=nn.linear)

        res = network.run(self.x)
        np.testing.assert_almost_equal(res, 0.0, self.prec)

    def test_009_rnn_no_input_with_bias(self):
        W1 = np.random.normal(size=(self._SIZE, self._SIZE))
        W2 = np.vstack((W1, np.zeros((self._NFEATURES, self._SIZE))))
        network = nn.SimpleRNN(W2, b=self.b, fun=nn.linear)

        res = network.run(self.x)
        res2 = np.zeros(self._SIZE, dtype=nn.dtype)
        for i in xrange(self._NSTEP):
            res2 = res2.dot(W1) + self.b
            np.testing.assert_allclose(res[i], res2, self.prec)

    def test_010_birnn_no_input_with_bias(self):
        W1 = np.random.normal(size=(self._SIZE, self._SIZE))
        W2 = np.vstack((W1, np.zeros((self._NFEATURES, self._SIZE))))
        layer1 = nn.SimpleRNN(W2, b=self.b)
        layer2 = nn.SimpleRNN(W2, b=self.b)
        network = nn.BiRNN(layer1, layer2)

        res = network.run(self.x)
        np.testing.assert_almost_equal(res[:,:self._SIZE], res[::-1,self._SIZE:], self.prec)
