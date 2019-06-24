# config = {}
# class TrainingConfig(object):
p = 0.7
base_rate = 0.0001
momentum = 0.9
decay_step = 500
decay_rate = 0.95
epoches = 100
evaluate_every = 20
checkpoint_every = 100


# class ModelConfig(object):
conv_layers = [[256, 7, 3],
                [256, 7, 3],
                [256, 3, None],
                [256, 3, None],
                [256, 3, None],
                [256, 3, 3]]

fully_connected_layers = [1024, 1024]
th = 1e-6
    
    
# class Config(object):

alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
alphabet_size = len(alphabet)
l0 = 1000
batch_size = 64
no_of_classes = 5

train_data_source = 'data/train.csv'
dev_data_source = 'data/test.csv'
    
    # training = TrainingConfig()
    
#     model = ModelConfig()

# config = Config()
