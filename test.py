import tensorflow as tf
import pandas as pd
import time
import os
import numpy

from tqdm import tqdm, trange
from deepfill import *
from config import *
from utils import *

from utils import ResizedDataReader

generator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)
discriminator_optimizer = tf.keras.optimizers.SGD(learning_rate=1e-4)

FLAGS = Config('./inpaint.yml')

generator = Generator()
discriminator = Discriminator()

def load(img):
    img = tf.io.read_file(img)
    img = tf.image.decode_jpeg(img)
    return tf.cast(img, tf.float32)

def normalize(img):
    return (img / 127.5) - 1.


def load_image_train(dir, img):
    path = img
    img = load(dir + "/" + img)
    img = resize_pipeline(img, IMG_HEIGHT, IMG_WIDTH)
    return {
        "path": path,
        "img": tf.expand_dims(normalize(img), axis=0)
    }

def resize_pipeline(img, height, width):
    return tf.image.resize(img, [height, width],
                           method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

testing_dirs = "./TEST"

# Get file names
test_dataset = [i for i in os.listdir(
    testing_dirs) if i.endswith(".jpg")]

# Get images
test_dataset = [load_image_train(testing_dirs, x)
                for x in test_dataset]

# test_dataset = tf.data.Dataset.list_files("../TEST/*.jpg")
# test_dataset = test_dataset.map(load_image_train)
# test_dataset = test_dataset.batch(FLAGS.batch_size)
# test_dataset = test_dataset.prefetch(buffer_size = tf.data.experimental.AUTOTUNE)

checkpoint_dir = "./training_checkpoints"
checkpoint = tf.train.Checkpoint(step=tf.Variable(0),
                                 generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)
"'need to change this'"
checkpoint.restore(checkpoint_dir+'/'+'ckpt-19')
step = np.int(checkpoint.step)
print("Continue Testing from epoch ", step)

#restore CSV
#df_load = pd.read_csv(f'./CSV_loss/loss_{step}.csv', delimiter=',')
#g_total = df_load['g_total'].values.tolist()
#g_total = CSV_reader(g_total)
#g_hinge = df_load['g_hinge'].values.tolist()
#g_hinge = CSV_reader(g_hinge)
#g_l1 = df_load['g_l1'].values.tolist()
#g_l1 = CSV_reader(g_l1)
#d = df_load['d'].values.tolist()
#d = CSV_reader(d)
#print(f'Loaded CSV for step: {step}')

# for data in test_dataset.take(15):
#   generate_images(data, generator, training=False, num_epoch=step)
reader = ResizedDataReader()
reader.read_all()

index = 0
tmp = 0
for test_number, input in enumerate(test_dataset):
    #print(input)
    input_filename = input["path"]
    # TODO: The create_mask part needs a mask having the same shape as our mask"
    #image_id = input_filename.replace('_surgical.jpg', '')
    image_id = input_filename.replace('_surgical.jpg', '.jpg')
    #print("Image ID: " + str(image_id))
    if reader.get_num_masks(image_id)==False:
        continue
    num_masks = reader.get_num_masks(image_id)
    # mask_num starts at 1 not 0, so offset by 1
    #print("Num Masks: " + str(num_masks))
    index += 1
    for mask_num in range(1, num_masks + 1):
        if reader.get_mask_coords(image_id, num_masks) == False:
            continue
        if reader.get_image_hw(image_id) == False:
            continue
        xmin, ymin, xmax, ymax = reader.get_mask_coords(
            image_id, num_masks)
        oheight, owidth = reader.get_image_hw(image_id)

        #mask = create_mask(FLAGS, xmin, ymin, xmax, ymax)
        mask = create_mask(FLAGS, xmin, ymin, xmax, ymax, oheight, owidth)
        generate_images(test_number, input["img"], generator=generator,
                        num_epoch=step, mask=mask)
    tmp += 1
    if tmp > 2:
        break
#plot_history(g_total, g_hinge, g_l1, d, step, training=False)
