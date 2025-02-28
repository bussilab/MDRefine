import unittest
# import MDRefine
from MDRefine import compute_new_weights, compute_D_KL, l2_regularization  # , compute_chi2
import scipy

class my_testcase(unittest.TestCase):
    def assertEqualObjs(self, obj1, obj2):
        
        import numpy as np
        import jax.numpy as jnp

        print(obj1, obj2)

        if isinstance(obj1, np.ndarray) or isinstance(obj1, jnp.ndarray):
            if obj1.shape == (1,): obj1 = obj1[0]
            elif obj1.shape == (): obj1 = np.array([obj1])[0]
        if isinstance(obj2, np.ndarray) or isinstance(obj2, jnp.ndarray):
            if obj2.shape == (1,): obj2 = obj2[0]
            elif obj2.shape == (): obj2 = np.array([obj2])[0]

        if isinstance(obj1, dict) and isinstance(obj2, dict):
            self.assertSetEqual(set(obj1.keys()), set(obj2.keys()))
            for k in obj1.keys():
                self.assertEqualObjs(obj1[k], obj2[k])
        
        elif isinstance(obj1, list) and isinstance(obj2, list):
            self.assertEqual(len(obj1), len(obj2))
            for i in range(len(obj1)):
                self.assertEqualObjs(obj1[i], obj2[i])
        
        elif isinstance(obj1, tuple) and isinstance(obj2, tuple):
            self.assertEqual(len(obj1), len(obj2))
            for i in range(len(obj1)):
                self.assertEqualObjs(obj1[i], obj2[i])
        
        else:
            if (isinstance(obj1, np.ndarray) or isinstance(obj1, jnp.ndarray)) and (
                    isinstance(obj2, np.ndarray) or isinstance(obj2, jnp.ndarray)):
                self.assertAlmostEqual(np.sum((obj1 - obj2)**2), 0)
            elif isinstance(obj1, bool) and isinstance(obj2, bool):
                self.assertTrue(obj1 == obj2)
            elif isinstance(obj1, float) and isinstance(obj2, float):
                self.assertAlmostEqual(obj1, obj2)
            elif isinstance(obj1, int) and isinstance(obj2, int):
                self.assertEqual(obj1, obj2)
            else:
                if isinstance(obj1, bytes) or isinstance(obj2, bytes):
                    if isinstance(obj1, bytes): obj1 = str(obj1, 'utf-8')
                    else: obj2 = str(obj2, 'utf-8')
                elif not (isinstance(obj1, scipy.optimize._lbfgsb_py.LbfgsInvHessProduct) or isinstance(obj1, scipy.optimize._lbfgsb_py.LbfgsInvHessProduct)):
                    print('WARNING: obj1 is ', type(obj1), 'while obj2 is ', type(obj2))
                    
                    self.assertEqual(obj1, obj2)

class Test(my_testcase):

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

    def test_compute_DeltaDeltaG_terms(self):

        import jax.numpy as jnp
        from MDRefine import load_data, compute_DeltaDeltaG_terms

        ###################################### load data #############################
        infos = {'global': {'temperature': 2.476, 'path_directory': 'tests/DATA_test'}}

        cycle_names = ['A1']

        names = {}
        for name in cycle_names:
            names[name] = []
            for string in ['AS','AD','MS','MD']:
                names[name].append((name + '_' + string))

        infos['global']['cycle_names'] = names
        infos['global']['system_names'] = [s2 for s in list(names.values()) for s2 in s]

        # force-field correction terms

        n_charges = 5

        infos['global']['names_ff_pars'] = ['DQ %i' % (i+1) for i in range(n_charges)] + ['cos eta']

        columns = []
        for i in range(n_charges):
            columns.append('DQ %i' % (i+1))
            columns.append('DQ %i%i' % (i+1,i+1))
        for i in range(n_charges):
            for j in range(i+1,n_charges):
                columns.append('DQ %i%i' % (i+1,j+1))
        columns.append('cos eta')

        # only methylated (M) systems have a force-field correction

        for name in infos['global']['system_names']: infos[name] = {}

        for name in infos['global']['cycle_names'].keys():
            for s in ['D', 'S']:
                infos[name + '_M' + s]['ff_terms'] = columns

        names_charges = ['N6', 'H61', 'N1', 'C10', 'H101/2/3']

        def ff_correction(phi, ff_terms):

            n_charges = 5

            phi_vector = []
            for i in range(n_charges):
                phi_vector.extend([phi[i], phi[i]**2])
            for i in range(n_charges):
                for j in range(i+1,n_charges):
                    phi_vector.append(phi[i]*phi[j])
            phi_vector.append(-phi[-1])
            phi_vector = jnp.array(phi_vector)

            correction = jnp.matmul(ff_terms, phi_vector)

            return correction

        for k in infos['global']['system_names']:
            if k[-2] == 'M': 
                infos[k]['ff_correction'] = ff_correction

        data = load_data(infos)

        ############ test #########################################################
        
        out = compute_DeltaDeltaG_terms(data, logZ_P={'A1_MS': 1., 'A1_MD': 1.5})
        out_test = ({'A1_MS': 255.7655459570046, 'A1_MD': 256.2379948027602}, {'A1': 135.84140982133923}, 67.92070491066961)

        self.assertEqualObjs(out_test, out)

    def test_compute_chi2(self):

        import jax.numpy as jnp
        import numpy as np
        from MDRefine import load_data, compute_chi2

        infos = {'global': {
            'path_directory': 'tests/DATA_test',
            'system_names': ['AAAA', 'CAAU'],
            'g_exp': ['backbone1_gamma_3J', 'backbone2_beta_epsilon_3J', 'sugar_3J', 'NOEs' , ('uNOEs', '<')],
            'forward_qs': ['backbone1_gamma', 'backbone2_beta_epsilon','sugar'],
            'obs': ['NOEs', 'uNOEs'],
            'forward_coeffs': 'original_fm_coeffs'}}

        def forward_model_fun(fm_coeffs, forward_qs, selected_obs=None):

            # 1. compute the cosine (which is the quantity you need in the forward model;
            # you could do this just once before loading data)
            forward_qs_cos = {}

            for type_name in forward_qs.keys():
                forward_qs_cos[type_name] = jnp.cos(forward_qs[type_name])

            # if you have selected_obs, compute only the corresponding observables
            if selected_obs is not None:
                for type_name in forward_qs.keys():
                    forward_qs_cos[type_name] = forward_qs_cos[type_name][:,selected_obs[type_name+'_3J']]

            # 2. compute observables (forward_qs_out) through forward model
            forward_qs_out = {
                'backbone1_gamma_3J': fm_coeffs[0]*forward_qs_cos['backbone1_gamma']**2 + fm_coeffs[1]*forward_qs_cos['backbone1_gamma'] + fm_coeffs[2],
                'backbone2_beta_epsilon_3J': fm_coeffs[3]*forward_qs_cos['backbone2_beta_epsilon']**2 + fm_coeffs[4]*forward_qs_cos['backbone2_beta_epsilon'] + fm_coeffs[5],
                'sugar_3J': fm_coeffs[6]*forward_qs_cos['sugar']**2 + fm_coeffs[7]*forward_qs_cos['sugar'] + fm_coeffs[8] }

            return forward_qs_out

        infos['global']['forward_model'] = forward_model_fun
        infos['global']['names_ff_pars'] = ['sin alpha', 'cos alpha']

        def ff_correction(pars, f):
            out = jnp.matmul(pars, (f[:, [0, 6]] + f[:, [1, 7]] + f[:, [2, 8]]).T)
            return out

        infos['global']['ff_correction'] = ff_correction

        data = load_data(infos)

        out = compute_chi2(data.mol['AAAA'].ref, data.mol['AAAA'].weights, data.mol['AAAA'].g, data.mol['AAAA'].gexp)

        out_test = ({'backbone1_gamma_3J': np.array([2.2820567 , 2.37008063]),
            'backbone2_beta_epsilon_3J': np.array([6.39268088, 3.86126331]),
            'sugar_3J': np.array([3.71089481, 4.77456358]),
            'NOEs': np.array([1.87342536e-03, 4.30196379e-05]),
            'uNOEs': np.array([1.33028693e-05, 5.82998086e-06])},
            {'backbone1_gamma_3J': np.array(1.08493846),
            'backbone2_beta_epsilon_3J': np.array(1.88280674),
            'sugar_3J': np.array(2.14070494),
            'NOEs': np.array(6.1036602),
            'uNOEs': np.array(0.)},
            {'backbone1_gamma_3J': np.array([-1.0119622 ,  0.24672042]),
            'backbone2_beta_epsilon_3J': np.array([-1.37154608,  0.0408422 ]),
            'sugar_3J': np.array([1.14059654, 0.91637572]),
            'NOEs': np.array([ 2.40941428, -0.54624448]),
            'uNOEs': np.array([0., 0.])},
            np.array(11.21211034))

        self.assertEqualObjs(out_test, out)

        # if_separate = True (no change)
        out = compute_chi2(data.mol['AAAA'].ref, data.mol['AAAA'].weights, data.mol['AAAA'].g, data.mol['AAAA'].gexp, True)

        self.assertEqualObjs(out_test, out)

    def test_gamma_function(self):
        
        import jax.numpy as jnp
        import numpy as np
        from MDRefine import load_data, gamma_function

        infos = {'global': {
            'path_directory': 'tests/DATA_test',
            'system_names': ['AAAA'],
            'g_exp': ['backbone1_gamma_3J', 'backbone2_beta_epsilon_3J', 'sugar_3J', 'NOEs' , ('uNOEs', '<')],
            'forward_qs': ['backbone1_gamma', 'backbone2_beta_epsilon','sugar'],
            'obs': ['NOEs', 'uNOEs'],
            'forward_coeffs': 'original_fm_coeffs'}}

        def forward_model_fun(fm_coeffs, forward_qs, selected_obs=None):

            # 1. compute the cosine (which is the quantity you need in the forward model;
            # you could do this just once before loading data)
            forward_qs_cos = {}

            for type_name in forward_qs.keys():
                forward_qs_cos[type_name] = jnp.cos(forward_qs[type_name])

            # if you have selected_obs, compute only the corresponding observables
            if selected_obs is not None:
                for type_name in forward_qs.keys():
                    forward_qs_cos[type_name] = forward_qs_cos[type_name][:,selected_obs[type_name+'_3J']]

            # 2. compute observables (forward_qs_out) through forward model
            forward_qs_out = {
                'backbone1_gamma_3J': fm_coeffs[0]*forward_qs_cos['backbone1_gamma']**2 + fm_coeffs[1]*forward_qs_cos['backbone1_gamma'] + fm_coeffs[2],
                'backbone2_beta_epsilon_3J': fm_coeffs[3]*forward_qs_cos['backbone2_beta_epsilon']**2 + fm_coeffs[4]*forward_qs_cos['backbone2_beta_epsilon'] + fm_coeffs[5],
                'sugar_3J': fm_coeffs[6]*forward_qs_cos['sugar']**2 + fm_coeffs[7]*forward_qs_cos['sugar'] + fm_coeffs[8] }

            return forward_qs_out

        infos['global']['forward_model'] = forward_model_fun
        infos['global']['names_ff_pars'] = ['sin alpha', 'cos alpha']

        def ff_correction(pars, f):
            out = jnp.matmul(pars, (f[:, [0, 6]] + f[:, [1, 7]] + f[:, [2, 8]]).T)
            return out

        infos['global']['ff_correction'] = ff_correction

        data = load_data(infos)

        flatten_g = np.hstack([data.mol['AAAA'].g[k] for k in data.mol['AAAA'].n_experiments.keys()])
        flatten_gexp = np.vstack([data.mol['AAAA'].gexp[k] for k in data.mol['AAAA'].n_experiments.keys()])

        alpha = 1.5

        # fixed random values
        lambdas = np.array([0.02276649, 0.92055914, 0.54435632, 0.28184011, 0.75414035,
            0.75551687, 0.47772936, 0.8749338 , 0.7059772 , 0.96640172])

        out = gamma_function(lambdas, flatten_g, flatten_gexp, data.mol['AAAA'].weights, alpha, True)

        out_test = (np.array([(6.27231311)]),
            np.array([ 3.34791024e-01,  3.63254555e+00,  6.39012045e+00,  1.29484769e+00,
                    4.05246153e+00,  1.92475534e+00, -8.35131574e-06,  5.11595544e-05,
                    1.48046374e-04,  7.04939569e-05]),
            np.array([3.54204586e+00, 1.47434153e+00, 3.89708214e+00,
                        3.45636268e+00, 4.92762134e-01, 4.02511408e+00,
                        7.82813097e-04, 3.06092488e-05, 1.01479652e-05,
                        1.75379015e-06]))

        self.assertEqualObjs(out_test, out)

    def test_minimizer(self):
    
        import pickle
        import jax.numpy as jnp
        import numpy as np
        from MDRefine import load_data, minimizer

        # define infos

        import jax.numpy as jnp

        def forward_model_fun(fm_coeffs, forward_qs, selected_obs=None):

            # 1. compute the cosine (which is the quantity you need in the forward model;
            # you could do this just once before loading data)
            forward_qs_cos = {}

            for type_name in forward_qs.keys():
                forward_qs_cos[type_name] = jnp.cos(forward_qs[type_name])

            # if you have selected_obs, compute only the corresponding observables
            if selected_obs is not None:
                for type_name in forward_qs.keys():
                    forward_qs_cos[type_name] = forward_qs_cos[type_name][:,selected_obs[type_name+'_3J']]

            # 2. compute observables (forward_qs_out) through forward model
            forward_qs_out = {
                'backbone1_gamma_3J': fm_coeffs[0]*forward_qs_cos['backbone1_gamma']**2 + fm_coeffs[1]*forward_qs_cos['backbone1_gamma'] + fm_coeffs[2],
                'backbone2_beta_epsilon_3J': fm_coeffs[3]*forward_qs_cos['backbone2_beta_epsilon']**2 + fm_coeffs[4]*forward_qs_cos['backbone2_beta_epsilon'] + fm_coeffs[5],
                'sugar_3J': fm_coeffs[6]*forward_qs_cos['sugar']**2 + fm_coeffs[7]*forward_qs_cos['sugar'] + fm_coeffs[8] }

            return forward_qs_out

        def ff_correction(pars, f):
            out = jnp.matmul(pars, (f[:, [0, 6]] + f[:, [1, 7]] + f[:, [2, 8]]).T)
            return out

        infos = {'global': {
            'path_directory': 'tests/DATA_test',
            'system_names': ['AAAA', 'CAAU'],
            'g_exp': ['backbone1_gamma_3J', 'backbone2_beta_epsilon_3J', 'sugar_3J', 'NOEs', ('uNOEs', '<')],
            'forward_qs': ['backbone1_gamma', 'backbone2_beta_epsilon','sugar'],
            'obs': ['NOEs', 'uNOEs'],
            'forward_coeffs': 'original_fm_coeffs', 'forward_model': forward_model_fun,
            'names_ff_pars': ['sin alpha', 'cos alpha'], 'ff_correction': ff_correction}}


        data = load_data(infos)

        result = minimizer(data, alpha=1.5)

        test_result = pickle.load(open('tests/DATA_test/result1.pkl', 'rb'))

        del test_result['time'], result.time

        self.assertEqualObjs(vars(result), test_result)

if __name__ == "__main__":
    unittest.main()