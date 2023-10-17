import numpy as np
from main import multi_layer_nn_tensorflow


def get_data():
    X = np.array([[0.685938, -0.5756752], [0.944493, -0.02803439], [0.9477775, 0.59988844], [0.20710745, -0.12665261],
                  [-0.08198895, 0.22326154], [-0.77471393, -0.73122877], [-0.18502127, 0.32624513],
                  [-0.03133733, -0.17500992], [0.28585237, -0.01097354], [-0.19126464, 0.06222228],
                  [-0.0303282, -0.16023481], [-0.34069192, -0.8288299], [-0.20600465, 0.09318836],
                  [0.29411194, -0.93214977], [-0.7150941, 0.74259764], [0.13344735, 0.17136675],
                  [0.31582892, 1.0810335], [-0.22873795, 0.98337173], [-0.88140666, 0.05909261],
                  [-0.21215424, -0.05584779]], dtype=np.float32)
    y = np.array(
        [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0],
         [1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0],
         [0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    return (X, y)


def get_data_2():
    X = np.array(
        [[0.55824741, 0.8871946, 0.69239914], [0.25242493, 0.77856301, 0.66000716], [0.4443564, 0.1092453, 0.96508663],
         [0.66679551, 0.49591846, 0.9536062], [0.07967996, 0.61238854, 0.89165257],
         [0.36541977, 0.02095794, 0.49595849], [0.56918241, 0.45609922, 0.05487656],
         [0.38711358, 0.02771098, 0.27910454], [0.16556168, 0.9003711, 0.5345797], [0.70774465, 0.5294432, 0.77920751]],
        dtype=np.float32)
    y = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
                  [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    return (X, y)


def test_random_weight_init():
    np.random.seed(1234)
    (X, y) = get_data()
    [W, err, Out] = multi_layer_nn_tensorflow(X, y, [8, 2], ['relu', 'linear'], alpha=0.01, batch_size=32, epochs=0,
                                              loss='mse')
    assert W[0].dtype == np.float32
    assert W[1].dtype == np.float32
    assert W[0].shape == (3, 8)
    assert W[1].shape == (9, 2)
    assert np.allclose(W[0], np.array(
        [[-0.41675785, -0.05626683, -2.1361961, 1.6402708, -1.7934356, -0.84174734, 0.5028814, -1.2452881],
         [-1.0579522, -0.9090076, 0.55145407, 2.292208, 0.04153939, -1.1179254, 0.5390583, -0.5961597],
         [-0.0191305, 1.1750013, -0.7478709, 0.00902525, -0.8781079, -0.15643416, 0.25657046, -0.98877907]],
        dtype=np.float32))
    assert np.allclose(W[1], np.array(
        [[-0.41675785, -0.05626683], [-2.1361961, 1.6402708], [-1.7934356, -0.84174734], [0.5028814, -1.2452881],
         [-1.0579522, -0.9090076], [0.55145407, 2.292208], [0.04153939, -1.1179254], [0.5390583, -0.5961597],
         [-0.0191305, 1.1750013]], dtype=np.float32))


def test_weight_update_mse():
    np.random.seed(1234)
    (X, y) = get_data()
    [W, err, Out] = multi_layer_nn_tensorflow(X, y, [8, 2], ['relu', 'linear'], alpha=0.01, batch_size=32, epochs=1,
                                              loss='mse')

    assert np.allclose(W[0], np.array(
        [[-0.4207438, -0.0833396, -2.1361961, 1.587104, -1.7934356, -0.8420822, 0.5038378, -1.2452881],
         [-1.0550305, -0.9031968, 0.5514541, 2.2804956, 0.0415394, -1.117666, 0.537059, -0.5961597],
         [-0.0203267, 1.1672614, -0.7478709, 0.0097109, -0.8781079, -0.1561893, 0.2576031, -0.9887791]],
        dtype=np.float32))
    assert np.allclose(W[1], np.array(
        [[-0.3868868, -0.031267], [-2.1351278, 1.6408045], [-1.7877496, -0.838592], [0.5028814, -1.2452881],
         [-0.9997215, -0.8513143], [0.5514541, 2.292208], [0.0416522, -1.1178796], [0.5562765, -0.5800341],
         [-0.0191305, 1.1750013]], dtype=np.float32))


def test_weight_update_cross_entropy():
    np.random.seed(1234)
    (X, y) = get_data()
    [W, err, Out] = multi_layer_nn_tensorflow(X, y, [8, 2], ['relu', 'linear'], alpha=0.01, batch_size=32, epochs=1,
                                              loss='cross_entropy')

    assert np.allclose(W[0], np.array(
        [[-0.4162578, -0.0578952, -2.1361961, 1.640059, -1.7934356, -0.8418525, 0.5044962, -1.2452881],
         [-1.0583302, -0.9088461, 0.5514541, 2.2923458, 0.0415394, -1.117844, 0.5380082, -0.5961597],
         [-0.0192638, 1.1747341, -0.7478709, 0.0089414, -0.8781079, -0.1563573, 0.2572094, -0.9887791]],
        dtype=np.float32))
    assert np.allclose(W[1], np.array(
        [[-0.415426, -0.0575986], [-2.1362476, 1.6403222], [-1.7930477, -0.8421353], [0.5028814, -1.2452881],
         [-1.0577341, -0.9092256], [0.5514541, 2.292208], [0.0415268, -1.1179128], [0.5394195, -0.5965208],
         [-0.0191305, 1.1750013]], dtype=np.float32))


def test_weight_update_svm():
    np.random.seed(1234)
    (X, y) = get_data()
    [W, err, Out] = multi_layer_nn_tensorflow(X, y, [8, 2], ['relu', 'linear'], alpha=0.01, batch_size=32, epochs=1,
                                              loss='svm')

    assert np.allclose(W[0], np.array(
        [[-0.4157327, -0.0593321, -2.1361961, 1.6355909, -1.7934356, -0.8420967, 0.5032797, -1.2452881],
         [-1.0587158, -0.908522, 0.5514541, 2.2917249, 0.0415394, -1.1176548, 0.5387031, -0.5961597],
         [-0.0191247, 1.1743149, -0.7478709, 0.0091823, -0.8781079, -0.1561787, 0.2568288, -0.9887791]],
        dtype=np.float32))
    assert np.allclose(W[1], np.array(
        [[-0.4139453, -0.0540793], [-2.1361961, 1.6405028], [-1.7930509, -0.8412891], [0.5028814, -1.2452881],
         [-1.0534091, -0.9046338], [0.5514541, 2.292208], [0.0415394, -1.117882], [0.5404882, -0.5949928],
         [-0.0191305, 1.1750013]], dtype=np.float32))


def test_assign_weights_by_value():
    (X, y) = get_data()
    W_0 = np.array([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
                    [16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0]], dtype=np.float32)
    W_1 = np.array(
        [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0], [8.0, 9.0], [10.0, 11.0], [12.0, 13.0], [14.0, 15.0],
         [16.0, 17.0]], dtype=np.float32)
    [W, err, Out] = multi_layer_nn_tensorflow(X, y, [W_0, W_1], ['relu', 'linear'], alpha=0.01, batch_size=32, epochs=0,
                                              loss='cross_entropy')
    assert np.allclose(W[0], W_0)
    assert np.allclose(W[1], W_1)


def test_error_output_dimensions():
    (X, y) = get_data()
    [W, err, Out] = multi_layer_nn_tensorflow(X, y, [8, 2], ['relu', 'linear'], alpha=0.01, batch_size=32, epochs=1,
                                              loss='mse', validation_split=[0.5, 0.7])
    assert np.allclose(err, np.array([4.8405566])) or np.allclose(err, [4.8405566])


    [W, err, Out] = multi_layer_nn_tensorflow(X, y, [8, 2], ['relu', 'linear'], alpha=0.01, batch_size=32, epochs=3,
                                              loss='mse', validation_split=[0.5, 1.0])

    assert np.allclose(err, np.array([6.8585029, 6.0209093, 5.3415179])) or np.allclose(err, [6.8585029, 6.0209093, 5.3415179])

def test_error_vals_mse():
    (X, y) = get_data()
    [W, err, Out] = multi_layer_nn_tensorflow(X, y, [8, 2], ['relu', 'linear'], alpha=0.01, batch_size=32, epochs=4,
                                              loss='mse', validation_split=[0.5, 1.0])
    assert np.allclose(err, np.array([6.8585029, 6.0209093, 5.3415179, 4.7825942])) or np.allclose(err, [6.8585029, 6.0209093, 5.3415179, 4.7825942])
    (X, y) = get_data_2()
    [W, err2, Out] = multi_layer_nn_tensorflow(X, y, [7, 3], ['relu', 'linear'], alpha=0.01, batch_size=32, epochs=4,
                                               loss='mse', validation_split=[0.5, 1.0])

    assert np.allclose(err2, np.array([3.1490805, 2.8836415, 2.6525667, 2.4497211])) or np.allclose(err2, [3.1490805, 2.8836415, 2.6525667, 2.4497211])


def test_error_vals_cross_entropy():
    (X, y) = get_data()
    [W, err, Out] = multi_layer_nn_tensorflow(X, y, [8, 2], ['relu', 'linear'], alpha=0.01, batch_size=32, epochs=4,
                                              loss='cross_entropy', validation_split=[0.5, 1.0])

    assert np.allclose(err, np.array([0.5052376, 0.5049465, 0.5046608, 0.5043806])) or np.allclose(err, [0.5052376, 0.5049465, 0.5046608, 0.5043806])
    np.random.seed(5368)
    X = np.random.rand(50, 3)
    y = np.random.randint(0, 4, size=(50, 1))
    y = np.eye(4)[y]
    y = y.reshape(50, 4)
    [W, err, Out] = multi_layer_nn_tensorflow(X, y, [8, 4], ['relu', 'linear'], alpha=0.01, batch_size=32, epochs=4,
                                              loss='cross_entropy', validation_split=[0.5, 1.0])
    assert np.allclose(err, np.array([4.0795054, 4.0008411, 3.9278107, 3.8588321])) or np.allclose(err, [4.0795054, 4.0008411, 3.9278107, 3.8588321])


def test_initial_validation_output():
    (X, y) = get_data()
    [W, err, Out] = multi_layer_nn_tensorflow(X, y, [8, 2], ['relu', 'linear'], alpha=0.01, batch_size=32, epochs=0,
                                              loss='cross_entropy', validation_split=[0.5, 1.0])
    assert Out.shape == (10, 2)
    assert np.allclose(Out, np.array(
        [[-1.8369007, -1.7483202], [-1.2605281, -0.8941443], [-1.8605977, -1.5690917], [-2.6287963, -2.4041958],
         [-3.5842671, -0.94719946], [-2.1864333, -2.2156622], [-4.0781965, -3.561052], [-3.6103907, -2.5557148],
         [-2.9478502, -0.07346541], [-1.5626245, -1.3875837]], dtype=np.float32))


def test_many_layers():
    np.random.seed(1234)
    (X, y) = get_data()
    [W, err, Out] = multi_layer_nn_tensorflow(X, y, [8, 6, 7, 5, 3, 1, 9, 2],
                                              ['relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'linear'],
                                              alpha=0.01, batch_size=32, epochs=2, loss='cross_entropy')
    assert W[0].shape == (3, 8)
    assert W[1].shape == (9, 6)
    assert W[2].shape == (7, 7)
    assert W[3].shape == (8, 5)
    assert W[4].shape == (6, 3)
    assert W[5].shape == (4, 1)
    assert W[6].shape == (2, 9)
    assert W[7].shape == (10, 2)
    assert Out.shape == (4, 2)
    assert (isinstance(err, np.ndarray) or isinstance(err, list)) and len(err) == 2
