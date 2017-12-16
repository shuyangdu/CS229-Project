############ Parameters to control run #############
#Use CPU only or Nvidia GPU
use_cpu_only = False

#Use a fixed initial seed
deterministic_init = True

#Batch size
batch_size = 1

#Number of timesteps to look backwards - Backprop length
backprop_length = 20

#Number of steps to look ahead
forward_length = 64

#LSTM statesize
state_size = 20

#Number of Epochs to Iterate over
num_epochs = 10

#Sequence Length
sequence_length = 803

#Dropout value
dropout_value = 0.4

#Validation set size - 64 days
#(i.e., from 7/9/2017 to 9/10/2017 prediction)
V = 64

#Learning rate, decay factor
lrate_initial = 2.0
lrate_end = 0.2
ldecay_rate = (lrate_initial - lrate_end)/num_epochs

#Turn on/off animated display with learning
display_on = True

if use_cpu_only:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

if deterministic_init:   
    tf.set_random_seed(1)
