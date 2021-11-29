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

tf.random.set_seed(20)
tf.config.run_functions_eagerly(True)

FLAGS = Config('./inpaint.yml')
img_shapes = FLAGS.img_shapes

BATCH_SIZE = FLAGS.batch_size
img_shape = FLAGS.img_shapes
IMG_HEIGHT = img_shape[0]
IMG_WIDTH = img_shape[1]

# both are unmasked faces
training_dirs = "./TRAIN"
testing_dirs = "./TEST"

# image pre-processing


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
        "img": normalize(img)
    }


def resize_pipeline(img, height, width):
    return tf.image.resize(img, [height, width],
                           method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)


generator = Generator()
discriminator = Discriminator()

BUFFER_SIZE = 4000

# Get file names
train_dataset = [i for i in os.listdir(
    training_dirs) if i.endswith(".jpg")]

test_dataset = [i for i in os.listdir(
    testing_dirs) if i.endswith(".jpg")]

# Get images
train_dataset = [load_image_train(training_dirs, x)
                 for x in train_dataset]
test_dataset = [load_image_train(testing_dirs, x)
                for x in test_dataset]

# loss


def generator_loss(input, stage1, stage2, neg):
    gen_l1_loss = tf.reduce_mean(tf.abs(input - stage1))
    gen_l1_loss += tf.reduce_mean(tf.abs(input - stage2))
    gen_hinge_loss = -tf.reduce_mean(neg)
    total_gen_loss = gen_hinge_loss + gen_l1_loss
    return total_gen_loss, gen_hinge_loss, gen_l1_loss


def discriminator_loss(pos, neg):
    hinge_pos = tf.reduce_mean(tf.nn.relu(1.0 - pos))
    hinge_neg = tf.reduce_mean(tf.nn.relu(1.0 + neg))
    return tf.add(.5 * hinge_pos, .5 * hinge_neg)


# optimizers
generator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)
discriminator_optimizer = tf.keras.optimizers.SGD(learning_rate=1e-4)

# training step


def train_step(input, mask):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        batch_incomplete = input["img"]*(1.-mask)

        stage1, stage2, _ = generator(batch_incomplete, mask, training=True)

        batch_complete = stage2*mask + batch_incomplete*(1.-mask)
        batch_pos_neg = tf.concat(
            [tf.reshape(input["img"], [1, 64, 64, 3]), batch_complete], axis=0)
        if FLAGS.gan_with_mask:
            batch_pos_neg = tf.concat([batch_pos_neg, tf.tile(
                mask, [FLAGS.batch_size*2, 1, 1, 1])], axis=3)

        pos_neg = discriminator(batch_pos_neg, training=True)
        pos, neg = tf.split(pos_neg, 2)

        total_gen_loss, gen_hinge_loss, gen_l1_loss = generator_loss(
            input["img"], stage1, stage2, neg)
        dis_loss = discriminator_loss(pos, neg)

    generator_gradients = gen_tape.gradient(total_gen_loss,
                                            generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(dis_loss,
                                                 discriminator.trainable_variables)
    generator_optimizer.apply_gradients(zip(generator_gradients,
                                            generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                discriminator.trainable_variables))
    return total_gen_loss, gen_hinge_loss, gen_l1_loss, dis_loss

# fit function


def fit(train_ds, epochs, test_ds):
    reader = ResizedDataReader()
    reader.read_all()

    checkpoint.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        g_total, g_hinge, g_l1, d = [], [], [], []
        print("Restored from {}".format(manager.latest_checkpoint))
        df_load = pd.read_csv(
            f'./CSV_loss/loss_{int(checkpoint.step)}.csv', delimiter=',')

        g_total.extend(CSV_reader(df_load['g_total'].tolist()))
        g_hinge.extend(CSV_reader(df_load['g_hinge'].tolist()))
        g_l1.extend(CSV_reader(df_load['g_l1'].tolist()))
        d.extend(CSV_reader(df_load['d'].tolist()))

        print(f"Loaded CSV from step: {int(checkpoint.step)}")
    else:
        print("Initializing from scratch.")
        g_total, g_hinge, g_l1, d = [], [], [], []

    for ep in trange(epochs):
        start = time.time()

        checkpoint.step.assign_add(1)
        g_total_b, g_hinge_b, g_l1_b, d_b = 0, 0, 0, 0
        count = 0  # len(train_ds)
        # Train
        " the for loop put a mask for each image "
        index = 0
        for input_image in tqdm(train_ds):
            # input_image_filename = train_dataset_filenames[index]
            input_image_filename = input_image["path"]
            # TODO: The create_mask part needs a mask having the same shape as our mask"
            #image_id = input_image_filename.replace('.jpg', '')
            image_id = input_image_filename
            num_masks = reader.get_num_masks(image_id)
            # mask_num starts at 1 not 0, so offset by 1
            for mask_num in range(1, num_masks + 1):
                xmin, ymin, xmax, ymax = reader.get_mask_coords(
                    image_id, num_masks)
                oheight, owidth = reader.get_image_hw(image_id)

                mask = create_mask(FLAGS, xmin, ymin, xmax, ymax, oheight, owidth)

                total_gen_loss, gen_hinge_loss, gen_l1_loss, dis_loss = train_step(
                    input_image, mask)
                g_total_b += total_gen_loss
                g_hinge_b += gen_hinge_loss
                g_l1_b += gen_l1_loss
                d_b += dis_loss
                count += 1  # verify
            index += 1
        g_total.append(g_total_b/count)
        g_hinge.append(g_hinge_b/count)
        g_l1.append(g_l1_b/count)
        d.append(d_b/count)

        check_step = int(checkpoint.step)
        plot_history(g_total, g_hinge, g_l1, d, check_step)

        dict1 = {'g_total': g_total,
                 'g_hinge': g_hinge,
                 'g_l1': g_l1,
                 'd': d}

        gt = pd.DataFrame(dict1)
        gt.to_csv(f'./CSV_loss/loss_{check_step}.csv', index=False)
        index = 0
        for input in test_ds:
            #print(input)
            input_filename = input["path"]
            # TODO: The create_mask part needs a mask having the same shape as our mask"
            #image_id = input_filename.replace('_surgical.jpg', '')
            image_id = input_filename.replace('_surgical.jpg', '.jpg')
            #print("Image ID: " + str(image_id))
            num_masks = reader.get_num_masks(image_id)
            # mask_num starts at 1 not 0, so offset by 1
            #print("Num Masks: " + str(num_masks))
            index += 1
            for mask_num in range(1, num_masks + 1):

                xmin, ymin, xmax, ymax = reader.get_mask_coords(
                    image_id, num_masks)
                oheight, owidth = reader.get_image_hw(image_id)

                #mask = create_mask(FLAGS, xmin, ymin, xmax, ymax)
                mask = create_mask(FLAGS, xmin, ymin, xmax, ymax, oheight, owidth)
                generate_images(input["img"], generator=generator,
                                num_epoch=check_step, mask=mask)

        print("Epoch: ", check_step)

        if check_step % 10 == 0:
            save_path = manager.save()
            print(f"Saved checkpoint for step {check_step}: {save_path}")

        print(
            f'Time taken for epoch {check_step} is {time.time()-start} sec\n')
    manager.save()


# checkpoint
checkpoint_dir = "./training_checkpoints"
checkpoint = tf.train.Checkpoint(step=tf.Variable(0),
                                 generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)

checkpoint.restore(manager.latest_checkpoint)
print("Continue Training from epoch ", np.int(checkpoint.step))


# FIT
#EPOCHS = 200 - np.int(checkpoint.step)
EPOCHS = 20000

# fit(train_dataset.take(20), EPOCHS, test_dataset)
fit(train_dataset, EPOCHS, test_dataset)
