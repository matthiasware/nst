import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import PIL


#######################################
# parameters to play with
#######################################

# just take arbitrary models, thtat were trained on large image datasets
uri_vgg = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'

# img dimensions
content_size = 512
style_size = 512

# downsampling parameters
kernel_size = 3
strides = 1

# paths
p_style = "imgs/style/ships.jpg"
p_content = "imgs/content/rotwein.png"
p_result = "result.png"


def load_img_tensor(path_to_img: str, max_dim: int):
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img


def tensor_to_image(tensor: tf.Tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


def get_hub_model(uri: str):
    return hub.load(uri)


class Predictor:
    def __init__(self, uri_model: str, kernel_size: int=3, strides: int=1):
        self.model = get_hub_model(uri_model)
        self.kernel_size = kernel_size
        self.strides = strides

    def predict(self, img_content: tf.Tensor, img_style: tf.Tensor):
        img_style = tf.nn.avg_pool(
            img_style,
            ksize=[self.kernel_size, self.kernel_size],
            strides=[self.strides, self.strides],
            padding='SAME')
        outputs = self.model(tf.constant(img_content), tf.constant(img_style))
        return outputs[0]


predictor = Predictor(uri_vgg, kernel_size, strides)
img_content = load_img_tensor(p_content, content_size)
img_style = load_img_tensor(p_style, style_size)
img_prediction = predictor.predict(img_content, img_style)
result_img = tensor_to_image(img_prediction)
result_img.save(p_result)
