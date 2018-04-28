import os
import skimage.io

# Import Mask RCNN
import mrcnn.model as modellib
import mrcnn.coco as coco

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, 'data', 'model', "mask_rcnn_coco.h5")

IMAGE_DIR = os.path.join(ROOT_DIR, 'data', 'images')

HERE = os.path.dirname(os.path.realpath(__file__))

class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def init(model_dir=MODEL_DIR, model_path=COCO_MODEL_PATH):
    config = InferenceConfig()

    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=model_dir, config=config)

    # Load weights trained on MS-COCO
    model.load_weights(model_path, by_name=True)
    return model


# def get_demo_image(name='cows_800x600.jpg'):
#     # https://stackoverflow.com/questions/33648322/tensorflow-image-reading-display
#     filename_queue = tf.train.string_input_producer([os.path.join(IMAGE_DIR, name)])
#
#     reader = tf.WholeFileReader()
#     key, value = reader.read(filename_queue)
#
#     my_img = tf.image.decode_png(value)  # use png or jpg decoder based on your files.
#
#     init_op = tf.global_variables_initializer()
#     image = None
#     with tf.Session() as sess:
#         sess.run(init_op)
#
#         # Start populating the filename queue.
#
#         coord = tf.train.Coordinator()
#         threads = tf.train.start_queue_runners(coord=coord)
#
#         for i in range(1):  # length of your filename list
#             image = my_img.eval()  # here is your image Tensor :)
#
#         # print(image.shape)
#         # Image.fromarray(np.asarray(image)).show()
#
#         coord.request_stop()
#         coord.join(threads)
#     return image


def demo(name='cows_800x600.jpg', model=None, image_dir=IMAGE_DIR):
    # Create model and run prediction on a pic of cows
    # image = get_demo_image(name)
    image = skimage.io.imread(os.path.join(image_dir, name))

    if model is None:
        model = init()

    results = model.detect([image], verbose=1)
    r = results[0]
    print([class_names[i] for i in r['class_ids']])
    return r


if __name__ == '__main__':
    model = init()
