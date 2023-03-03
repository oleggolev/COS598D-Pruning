import torch
import torchvision.models as models
from Utils import load
from torch.utils.tensorboard import SummaryWriter
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

prunes = ["rand", "mag", "snip", "grasp", "synflow"]

for prune in prunes:
    k = tf.placeholder(tf.float32)
    model = load.model('vgg16', 'lottery')((3, 32, 32), 10, False, False)
    device = torch.device('cpu')
    model.load_state_dict(torch.load('/scratch/network/vikashm/Results/data2/singleshot/hp-' + prune + '-0.5/model.pt', map_location=device))
    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear):
            tf.summary.histogram(f"vgg16-cifar10-{prune}-{name}", layer.state_dict()['weight'])

    # Setup a session and summary writer
    sess = tf.Session()
    writer = tf.summary.FileWriter("./histograms")
    summaries = tf.summary.merge_all()

    # Setup a loop and write the summaries to disk
    N = 400
    for step in range(N):
        k_val = step/float(N)
        summ = sess.run(summaries, feed_dict={k: k_val})
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
