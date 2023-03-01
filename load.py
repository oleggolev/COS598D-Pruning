import torch
import torchvision.models as models
from Utils import load
from torch.utils.tensorboard import SummaryWriter
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

model = load.model('vgg16', 'lottery')((3, 32, 32), 10, False, False)
device = torch.device('cpu')
model.load_state_dict(torch.load('./Saved/hp-rand-0.5/model.pt', map_location=device))



for name, layer in model.named_modules():
    if isinstance(layer, torch.nn.Conv2d):
        tf.summary.histogram(f"{name}", layer.state_dict()['weight'])
        # writer.add_histogram(name, layer.weight, global_step=epoch, bins='tensorflow')
        # writer.add_histogram(f'{name}_bias', layer.weight, global_step=epoch, bins='tensorflow')
    elif isinstance(layer, torch.nn.Linear):
        tf.summary.histogram(f"{name}", layer.state_dict()['weight'])
        # writer.add_histogram(name, layer.weight, global_step=epoch, bins='tensorflow')
        # writer.add_histogram(f'{name}_bias', layer.weight, global_step=epoch, bins='tensorflow')

# Setup a session and summary writer
sess = tf.Session()
writer = tf.summary.FileWriter("./hp-rand-0.5")
summaries = tf.summary.merge_all()

# Setup a loop and write the summaries to disk
for step in range(16):
  summ = sess.run(summaries)
  writer.add_summary(summ, global_step=step)
  
#   weights = np.random.normal(loc=step, scale=1, size=[100])
#   summ = sess.run(summaries, feed_dict={dist: mean_moving_normal})
#   writer.add_summary(summ, global_step=step)

# for i in range(16):
#     try:
#         conv_layer = model.layers[i].conv
#         weights = conv_layer.state_dict()['weight']
#         print(weights)
#     except:
#         # this is a MaxPool2D layer, so we don't care about this one?                                                                                                                        
#         continue

# writer = SummaryWriter("hp-rand-0.5.hist")
# for name, layer in model.named_modules():
#     if isinstance(layer, torch.nn.Conv2d):
#         writer.add_histogram(name, layer.weight, global_step=epoch, bins='tensorflow')
#         writer.add_histogram(f'{name}_bias', layer.weight, global_step=epoch, bins='tensorflow')
#     elif isinstance(layer, torch.nn.Linear):
#         writer.add_histogram(name, layer.weight, global_step=epoch, bins='tensorflow')
#         writer.add_histogram(f'{name}_bias', layer.weight, global_step=epoch, bins='tensorflow')
# writer.close()
