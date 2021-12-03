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

def L1(x, y):
    loss = np.sum(np.abs(x - y))
    return loss

def L2(x, y):
    loss = np.sum(np.square(x - y))
    return loss

def load(img):
    img = tf.io.read_file(img)
    img = tf.image.decode_jpeg(img)
    return tf.cast(img, tf.float32)

def normalize(img):
    return (img / 127.5) - 1.


def load_image_train(dir, img):
    path = img
    #print("path:", dir + "/" +img)
    img = load(dir + "/" + img)
    img = resize_pipeline(img, IMG_HEIGHT, IMG_WIDTH)
    return {
        "path": path,
        "img": tf.expand_dims(normalize(img), axis=0)
    }

def resize_pipeline(img, height, width):
    return tf.image.resize(img, [height, width],
                           method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

def generate_images_test(test_number, input, original_input,loss1,loss2, generator, mask, training=False, url=False, num_epoch=0):
    # input = original
    # batch_incomplete = original+mask
    # stage2 = prediction/inpainted image
    # mask = create_mask(FLAGS)
    # tf.print(mask, summarize=-1)
    #loss1 = []
    #loss2 = []

    batch_incomplete = input*(1.-mask)
    stage1, stage2, offset_flow = generator(
       batch_incomplete, mask, training=training)

    plt.figure(figsize=(30, 30))

    batch_predict = stage2
    batch_complete = batch_predict*mask + batch_incomplete*(1-mask)

    display_list = [input, batch_incomplete[0] ,batch_complete[0], original_input[0]]
    title = ['Original Image', 'Unmasked Image','Inpainted Image', 'Ground Truth Image']
    if not url:
        x = []
        #fig, ax = plt.subplots(1, 3)
        for i in range(5):
            if i == 2 or i ==3:
                
                img = display_list[i] * 0.5 + 0.5
                if img.shape[0] == 1:
                    img = np.squeeze(img, axis=0)

                x.append(np.array(img))

            if i == 4:

                loss_l1 = L1(x[0], x[1])
                loss1.append(loss_l1)

                loss_l2 = L2(x[0], x[1])
                loss2.append((loss_l2))

        return loss1, loss2
    else:
        return batch_incomplete[0], batch_complete[0]

testing_dirs = "./TEST"
original_dirs = "./TRAIN"

# Get file names
test_dataset = [i for i in os.listdir(
    testing_dirs) if i.endswith(".jpg")]
original_dataset = [i for i in os.listdir(
    original_dirs) if i.endswith(".jpg")]
#print("original dataset:", original_dataset)
# Get images
test_dataset = [load_image_train(testing_dirs, x)
                for x in test_dataset]

original_dataset = [load_image_train(original_dirs, x)
                for x in original_dataset]

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

reader = ResizedDataReader()
reader.read_all()

index = 0
tmp = 0
loss1 = []
loss2 = []
for test_number, input in enumerate(test_dataset):
    #print(input)
    input_filename = input["path"]
    # TODO: The create_mask part needs a mask having the same shape as our mask"
    #image_id = input_filename.replace('_surgical.jpg', '')
    image_id = input_filename.replace('_surgical.jpg', '.jpg')

    for original_number, original_input in enumerate(original_dataset):
        if original_input["path"] == image_id:
            original_image = original_input

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
        loss1, loss2 = generate_images_test(test_number, input["img"], original_image["img"],loss1,loss2, generator=generator,
                        num_epoch=step, mask=mask)
    tmp += 1
    if tmp > 2:
        break
loss1 = np.mean(loss1)
loss2 = np.mean(loss2)

print("-----------------------------------------------------------------------")
print("The L1 loss: ", loss1)
print("The L2 loss:", loss2)
print("-----------------------------------------------------------------------")
      
