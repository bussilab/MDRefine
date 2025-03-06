import unittest
import MDRefine

class my_testcase(unittest.TestCase):
    def assertEqualObjs(self, obj1, obj2, tol = 1e-4, if_relative = False):
        
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

                if if_relative == False:
                    q = np.sum((obj1 - obj2)**2)
                    print('value: ', q)
                    self.assertTrue(q < tol)
                
                else:

                    wh = np.argwhere(obj1 == 0)
                    if wh.shape[0] != 0:
                        q = np.sum((obj1[obj1 == 0] - obj2[obj1 == 0])**2)
                        print('value: ', q)
                        self.assertTrue(q < tol)

                    wh = np.argwhere(obj1 != 0)
                    if wh.shape[0] != 0:
                        q = np.sum(((obj2[obj1 != 0] - obj1[obj1 != 0])/obj1[obj1 != 0])**2)
                        print('value: ', q)
                        self.assertTrue(q < tol)

            elif isinstance(obj1, bool) and isinstance(obj2, bool):
                self.assertTrue(obj1 == obj2)
            elif isinstance(obj1, float) and isinstance(obj2, float):
                self.assertTrue((obj1 - obj2)**2 < tol)
            elif isinstance(obj1, int) and isinstance(obj2, int):
                self.assertEqual(obj1, obj2)
            else:
                if isinstance(obj1, bytes) or isinstance(obj2, bytes):
                    if isinstance(obj1, bytes): obj1 = str(obj1, 'utf-8')
                    else: obj2 = str(obj2, 'utf-8')
                else: print('WARNING: obj1 is ', type(obj1), 'while obj2 is ', type(obj2))
                
                self.assertEqual(obj1, obj2)

class Test(my_testcase):
    def test_MDRefinement(self):

        import os
        import numpy as np
        import pandas
        from MDRefine import MDRefinement

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


        def forward_model_regularization(coeffs, coeffs_0):
            regularization = (
            3/8*(coeffs[0]-coeffs_0['A_gamma'])**2+1/2*(coeffs[1]-coeffs_0['B_gamma'])**2+(coeffs[2]-coeffs_0['C_gamma'])**2+(coeffs[0]-coeffs_0['A_gamma'])*(coeffs[2]-coeffs_0['C_gamma'])+
            3/8*(coeffs[3]-coeffs_0['A_beta'])**2+1/2*(coeffs[4]-coeffs_0['B_beta'])**2+(coeffs[5]-coeffs_0['C_beta'])**2+(coeffs[3]-coeffs_0['A_beta'])*(coeffs[5]-coeffs_0['C_beta'])+
            3/8*(coeffs[6]-coeffs_0['A_sugar'])**2+1/2*(coeffs[7]-coeffs_0['B_sugar'])**2+(coeffs[8]-coeffs_0['C_sugar'])**2+(coeffs[6]-coeffs_0['A_sugar'])*(coeffs[8]-coeffs_0['C_sugar']))

            return regularization

        regularization = {'force_field_reg': 'KL divergence', 'forward_model_reg': forward_model_regularization}
        
        my_string = 'results'

        MDRefinement(infos, regularization=regularization, starting_alpha=1, starting_beta=1, starting_gamma=1, which_set='validation', results_folder_name='tests/DATA_test/' + my_string, n_parallel_jobs=1)
        
        path_list = sorted(['tests/DATA_test/' + s + '/' for s in os.listdir('tests/DATA_test/') if s[:7] == my_string])

        print('path_list: ', path_list)

        for s in ['ff_AAAA', 'ff_CAAU', 'new_AAAA', 'new_CAAU']:

            my_vec0 = np.load(path_list[0] + 'weights_%s.npy' % s)
            my_vec1 = np.load(path_list[1] + 'weights_%s.npy' % s)

            self.assertEqualObjs(my_vec0, my_vec1)

        for s in ['hyper_search', 'min_lambdas', 'result']:

            # relax equalities because of small numerical variations for different Python versions
            if s == 'result': usecols = lambda x: x not in ['time', 'norm gradient', 'success']
            else: usecols = None

            if s in ['min_lambdas', 'result']: 
                if_relative = True
                tol = 1e-1
            else:
                if_relative = False
                tol = 1e-3

            my_vec0 = np.array(pandas.read_csv(path_list[0] + s, index_col=0, usecols=usecols))
            my_vec1 = np.array(pandas.read_csv(path_list[1] + s, index_col=0, usecols=usecols))

            self.assertEqualObjs(my_vec0, my_vec1, if_relative=if_relative, tol=tol)
        
        my_df0 = pandas.read_csv(path_list[0] + 'input', index_col=0, usecols=usecols)
        my_df1 = pandas.read_csv(path_list[1] + 'input', index_col=0, usecols=usecols)

        my_df0 = my_df0.drop(columns=['stride'])

        self.assertEqualObjs(list(my_df0.columns), list(my_df1.columns))

        for s in list(my_df0.columns):
            self.assertEqualObjs(my_df0[s].iloc[0], my_df1[s].iloc[0])

        ### os.rmdir(path_list[1])  # it works only for empty directories

        for root, dirs, files in os.walk(path_list[1], topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))

        os.rmdir(path_list[1])

if __name__ == "__main__":
    unittest.main()