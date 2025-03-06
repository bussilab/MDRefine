import unittest
# import MDRefine
# from MDRefine import compute_new_weights, compute_chi2, compute_D_KL, l2_regularization
from MDRefine import load_data

class Test(unittest.TestCase):
    def test_load_data(self):

        import pickle
        import jax.numpy as jnp
        import numpy as np

        #%% define test_function

        def test_function(infos, stride, path_pickle):

            # 1. load data with load_data
            data = load_data(infos, stride=stride)
            del data.properties.infos

            # 2. load pickle into loaded_data
            with open(path_pickle, 'rb') as f:
                loaded_data = pickle.load(f)

            # add ff_correction and forward_model to loaded_data (since you cannot load them from pickle)

            for k in loaded_data['mol'].keys():
                if k in infos.keys():
                    info = {**infos[k], **infos['global']}
                else:
                    info = infos['global']

                if 'ff_correction' in info.keys():
                    loaded_data['mol'][k].ff_correction = info['ff_correction']

                if 'forward_model' in info.keys():
                    def my_forward_model(a, b, c=None):
                        try: out = info['forward_model'](a, b, c)
                        except:
                            assert c is None, 'you have selected_obs but the forward model is not suitably defined!'
                            out = info['forward_model'](a, b)
                        return out

                    loaded_data['mol'][k].forward_model = my_forward_model

            # 3. compare

            ### this does not work because of the structure: dict contains dictionaries which contain numpy arrays...
            # for s in loaded_data['mol'].keys():
            #     assert vars(data.mol[s]) == vars(loaded_data['mol'][s])
            # self.assertDictEqual(vars(data.mol[s]), vars(loaded_data['mol'][s]))

            ### so, let's do in this way

            # self.assertListEqual(list(vars(data).keys()), list(loaded_data.keys()))

            # 3a. global properties
            self.assertListEqual(dir(data.properties), dir(loaded_data['properties']))

            self.assertListEqual(data.properties.system_names, loaded_data['properties'].system_names)

            if hasattr(loaded_data['properties'], 'forward_coeffs_0'):
                self.assertListEqual(list(loaded_data['properties'].forward_coeffs_0), list(data.properties.forward_coeffs_0))
                self.assertListEqual(list(loaded_data['properties'].forward_coeffs_0.keys()), list(data.properties.forward_coeffs_0.keys()))

            if hasattr(loaded_data['properties'], 'names_ff_pars'):
                self.assertListEqual(loaded_data['properties'].names_ff_pars, data.properties.names_ff_pars)

            # assert tot_n_experiments
            class my_data():
                def __init__(self):
                    self.mol = {}

            my_loaded_data = my_data()

            for k in loaded_data['mol'].keys():
                my_loaded_data.mol[k] = loaded_data['mol'][k]
            
            self.assertEqual(data.properties.tot_n_experiments(data), loaded_data['properties'].tot_n_experiments(my_loaded_data))

            # 3b. molecular systems
            self.assertSetEqual(set(data.mol.keys()), set(loaded_data['mol'].keys()))

            for s in infos['global']['system_names']:

                my_dict1 = vars(data.mol[s])
                my_dict2 = vars(loaded_data['mol'][s])

                self.assertSetEqual(set(my_dict1.keys()), set(my_dict2.keys()))

                for k in my_dict1.keys():

                    if k in ['temperature', 'n_frames', 'logZ']:
                        self.assertAlmostEqual(my_dict1[k], my_dict2[k])
                    
                    elif k in ['ref', 'n_experiments']:
                        self.assertDictEqual(my_dict1[k], my_dict2[k])

                    ###### it does not work on some versions of python
                    # elif k in ['gexp', 'names', 'g']:
                    #     for k2 in data.mol[s].gexp.keys():
                    #         # self.assertTrue((my_dict1[k][k2] == my_dict2[k][k2]).all())
                    #         # self.assertTrue(np.array_equal(my_dict1[k][k2], my_dict2[k][k2]))
                            
                    #         # np.allclose does not work on strings
                    #         # self.assertTrue(np.allclose(my_dict1[k][k2], my_dict2[k][k2]))

                    elif k in ['gexp', 'g']:
                        for k2 in data.mol[s].gexp.keys():
                            self.assertAlmostEqual(np.sum((my_dict1[k][k2] - my_dict2[k][k2])**2), 0)

                    elif k in ['names']:
                        # check element-wise
                        for k2 in my_dict1['names'].keys():

                            my_array1 = my_dict1['names'][k2]
                            my_array2 = my_dict2['names'][k2]

                            x1, y1 = np.shape(my_array1)
                            x2, y2 = np.shape(my_array2)

                            self.assertEqual(x1, x2)
                            self.assertEqual(y1, y2)

                            for ix in range(x1):
                                for iy in range(y1):
                                    self.assertEqual(my_array1[ix][iy], my_array2[ix][iy])

                    elif k in ['forward_qs']:
                        for k2 in data.mol[s].forward_qs.keys():
                            self.assertAlmostEqual(np.sum((my_dict1[k][k2] - my_dict2[k][k2])**2), 0)

                    elif k in ['weights', 'f']:
                        self.assertAlmostEqual(np.sum((my_dict1[k] - my_dict2[k])**2), 0)
                
            # 3c. cycles

            if hasattr(loaded_data['properties'], 'cycle_names'):
                self.assertSetEqual(set(loaded_data['properties'].cycle_names), set(data.properties.cycle_names))
                
                for s in infos['global']['cycle_names']:

                    my_dict1 = vars(data.cycle[s])
                    my_dict2 = vars(loaded_data['cycle'][s])

                    self.assertSetEqual(set(my_dict1.keys()), set(my_dict2.keys()))

                    for k in my_dict1.keys():

                        if k in ['temperature']:
                            self.assertAlmostEqual(my_dict1[k], my_dict2[k])

                        elif k in ['gexp_DDG']:
                            self.assertListEqual(my_dict1[k], my_dict2[k])

        #%% test n. 1: without forward model nor force-field correction """

        infos = {}
        infos['global'] = {'path_directory': 'tests/DATA_test', 'system_names': ['AAAA', 'CAAU']}

        for name in infos['global']['system_names']:
            infos[name] = {}
            infos[name]['g_exp'] = ['NOEs', ('uNOEs','<')]
            infos[name]['obs'] = ['NOEs', 'uNOEs']
        
        infos['global']['temperature'] = 1 # namely, energies are in unit of k_B T (default value)
        stride = 2

        path_pickle = 'tests/DATA_test/data_stride2.pkl'
        
        test_function(infos, stride, path_pickle)

        #%% test n. 2: complete

        infos = {'global': {
            'path_directory': 'tests/DATA_test',
            'system_names': ['AAAA', 'CAAU'],
            'g_exp': ['backbone1_gamma_3J', 'backbone2_beta_epsilon_3J', 'sugar_3J', 'NOEs' , ('uNOEs', '<')],
            'forward_qs': ['backbone1_gamma', 'backbone2_beta_epsilon','sugar'],
            'obs': ['NOEs', 'uNOEs'],
            'forward_coeffs': 'original_fm_coeffs'}}

        stride = 2

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

        path_pickle = 'tests/DATA_test/data_complete_stride2.pkl'

        test_function(infos, stride, path_pickle)

        #%% test n. 3: alchemical calculations

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

        stride = 2
        path_pickle = 'tests/DATA_test/data_alchemical_stride2.pkl'

        test_function(infos, stride, path_pickle)

if __name__ == "__main__":
    unittest.main()