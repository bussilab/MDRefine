import unittest
import MDRefine
from MDRefine import compute_new_weights, compute_chi2, compute_D_KL, l2_regularization

class Test(unittest.TestCase):
    def test_compute_new_weights_and_DKL(self):
        # import jax.numpy as np
        import numpy as np
        
        w0 = np.array([0.5, 0.5])
        correction = np.array([0., 1.])

        new_weights, logZ = compute_new_weights(w0, correction)

        self.assertAlmostEqual(np.sum(new_weights - np.array([0.73105858, 0.26894142]))**2, 0)
        self.assertAlmostEqual(logZ, -0.37988549)

        D_KL = compute_D_KL(weights_P=new_weights, correction_ff=1/2*correction, temperature=2, logZ_P=logZ)
        self.assertAlmostEqual(D_KL, 0.31265014)

    def test_l2_regularization(self):
        import numpy as np

        pars = np.array([1.2, 1.5])
        
        loss, grad = l2_regularization(pars)

        self.assertAlmostEqual(loss, 3.69)
        self.assertAlmostEqual(np.sum(grad - np.array([2.4, 3. ]))**2, 0)


if __name__ == "__main__":
    unittest.main()