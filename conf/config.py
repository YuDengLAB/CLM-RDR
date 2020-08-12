p = 0.7
base_rate = 0.0005
momentum = 0.8
decay_step = 400
decay_rate = 0.98
epoches = 100
evaluate_every = 20
checkpoint_every = 100


conv_layers = [[256, 7, 3],
                [256, 3, None],
                [256, 3, None],
                [256, 3, 3]]

fully_connected_layers = [1024, 1024]
th = 1e-6
    

alphabet = "abcdefghijklmnopqrstuvwxyz0123456789.%"
alphabet_size = len(alphabet)
l0 = 220
batch_size = 128
no_of_classes = 5
train_data_source = "train_0812.csv"
dev_data_source = "test_0812.csv"
test_data_source = "pred_0812.csv"

