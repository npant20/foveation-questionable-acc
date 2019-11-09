from foolbox import *
from image_warp import *
from get_data import *

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.backend import set_session
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.initializers import glorot_uniform
from google.colab import drive
drive.mount('/content/gdrive')

sess = tf.Session()
graph = tf.get_default_graph()
set_session(sess)

""" this is just to load the models, it will not work if you try to run this"""
with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
  fovmodel = tf.keras.models.load_model('/content/gdrive/My Drive/my_model.h5', )
  nofovmodel = tf.keras.models.load_model('/content/gdrive/My Drive/my_model_nofov.h5', )
  
model_input = tf.keras.layers.Input(shape=(64, 64, 3))
model_output = nofovmodel(model_input)                    ###This line indicates that the perturbations are based on the regular model
model = tf.keras.models.Model(inputs=model_input, outputs=model_output)
model.compile(optimizer=tf.keras.optimizers.SGD(), loss='categorical_crossentropy', metrics=['accuracy'])

fmodel = KerasModel2(model, bounds=(0.0, 255.0))
attack = GradientSignAttack(model=fmodel, criterion=TopKMisclassification(5))

train_data, train_labels, val_data, val_labels, test_data, test_labels = get_data(get_id_dictionary())


totalattacks = 0
correct = 0
for idx in range (1000):
  img = val_data[idx]
  realLabel = np.argmax(val_labels[idx])
  print('input: ' + str(realLabel))
  adversarial = attack((img), label = realLabel, epsilons=[0.04])
  if adversarial is not None:
    advpredict = np.argmax(nofovmodel.predict(adversarial[np.newaxis, ...]))
    print('adverserial: ' + str(advpredict))
    fovtop = top_predictor(adversarial, fovmodel)         ###using the foveated model to evaluate predictions
    print(fovtop)
    if fovtop == realLabel:
      correct +=1
    totalattacks += 1
  else:
      print("No attack could be found")

print(correct/totalattacks)
