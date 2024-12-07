import os
import tensorflow as tf
from azure.storage.blob import BlobServiceClient
# from official.vision.configs import efficientdet as efficientdet_cfg
from official.vision.configs import retinanet as retinanet_cfg
# from official.vision.training import task_factory
from official.core import task_factory
#from official.vision.modeling import factory
import tensorflow_models as tfm

from official.core import exp_factory
from official.core import config_definitions as cfg
from official.vision.serving import export_saved_model_lib
from official.vision.ops.preprocess_ops import normalize_image
from official.vision.ops.preprocess_ops import resize_and_crop_image
from official.vision.utils.object_detection import visualization_utils
from official.vision.dataloaders.tf_example_decoder import TfExampleDecoder


# Verinin indirileceği klasör
download_path = "/home/alphan/Desktop/Vehicle Tracking System/data"

def download_blob_files():
    container_client = blob_service_client.get_container_client(container_name)
    for blob in container_client.list_blobs():
        blob_client = container_client.get_blob_client(blob)
        file_path = os.path.join(download_path, blob.name)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, "wb") as download_file:
            download_file.write(blob_client.download_blob().readall())
        print(f"İndirilen: {blob.name}")

# download_blob_files()
train_data_dir = 'data/train'
test_data_dir  = 'data/test'

IMG_SIZE = 256

# Veriyi yükleme
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_data_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size = 1
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_data_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size = 1
)


# TFRecord dosyasına yazma işlemi
def create_tfrecord(dataset, tfrecord_file):
    with tf.io.TFRecordWriter(tfrecord_file) as writer:
        for image, label in dataset:
            image = tf.squeeze(image)
            image = tf.image.encode_jpeg(tf.cast(image, tf.uint8)).numpy()  # JPEG formatında sıkıştır
            label = label.numpy()  # NumPy formatına dönüştür
            
            feature = {
                "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
    print(f"TFRecord oluşturuldu: {tfrecord_file}")

# Datasetleri TFRecord'a çevirme
train_tfrecord_file = "train.tfrecord"
test_tfrecord_file = "test.tfrecord"

create_tfrecord(train_dataset, train_tfrecord_file)
create_tfrecord(test_dataset, test_tfrecord_file)


exp_config = exp_factory.get_exp_config('retinanet_resnetfpn_coco')


batch_size = 8
num_classes = 3

HEIGHT, WIDTH = 256, 256
IMG_SIZE = [HEIGHT, WIDTH, 3]

# Backbone config.
exp_config.task.freeze_backbone = False
exp_config.task.annotation_file = ''

# Model config.
exp_config.task.model.input_size = IMG_SIZE
exp_config.task.model.num_classes = num_classes + 1
exp_config.task.model.detection_generator.tflite_post_processing.max_classes_per_detection = exp_config.task.model.num_classes

# Training data config.
exp_config.task.train_data.input_path = train_tfrecord_file
exp_config.task.train_data.dtype = 'float32'
exp_config.task.train_data.global_batch_size = batch_size
exp_config.task.train_data.parser.aug_scale_max = 1.0
exp_config.task.train_data.parser.aug_scale_min = 1.0

# Validation data config.
exp_config.task.validation_data.input_path = test_tfrecord_file
exp_config.task.validation_data.dtype = 'float32'
exp_config.task.validation_data.global_batch_size = batch_size


logical_device_names = [logical_device.name for logical_device in tf.config.list_logical_devices()]

if 'GPU' in ''.join(logical_device_names):
  print('This may be broken in Colab.')
  device = 'GPU'
elif 'TPU' in ''.join(logical_device_names):
  print('This may be broken in Colab.')
  device = 'TPU'
else:
  print('Running on CPU is slow, so only train for a few steps.')
  device = 'CPU'


train_steps = 1000
exp_config.trainer.steps_per_loop = 100 # steps_per_loop = num_of_training_examples // train_batch_size

exp_config.trainer.summary_interval = 100
exp_config.trainer.checkpoint_interval = 100
exp_config.trainer.validation_interval = 100
exp_config.trainer.validation_steps =  100 # validation_steps = num_of_validation_examples // eval_batch_size
exp_config.trainer.train_steps = train_steps
exp_config.trainer.optimizer_config.warmup.linear.warmup_steps = 100
exp_config.trainer.optimizer_config.learning_rate.type = 'cosine'
exp_config.trainer.optimizer_config.learning_rate.cosine.decay_steps = train_steps
exp_config.trainer.optimizer_config.learning_rate.cosine.initial_learning_rate = 0.1
exp_config.trainer.optimizer_config.warmup.linear.warmup_learning_rate = 0.05


if exp_config.runtime.mixed_precision_dtype == tf.float16:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

if 'GPU' in ''.join(logical_device_names):
  distribution_strategy = tf.distribute.MirroredStrategy()
elif 'TPU' in ''.join(logical_device_names):
  tf.tpu.experimental.initialize_tpu_system()
  tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='/device:TPU_SYSTEM:0')
  distribution_strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
  print('Warning: this will be really slow.')
  distribution_strategy = tf.distribute.OneDeviceStrategy(logical_device_names[0])

print('Done')

model_dir = './trained_model/'
export_dir ='./exported_model/'

# Dizin mevcut değilse, oluştur
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    print(f"{model_dir} dizini oluşturuldu.")
else:
    print(f"{model_dir} dizini zaten mevcut.")

# Dizin mevcut değilse, oluştur
if not os.path.exists(export_dir):
    os.makedirs(export_dir)
    print(f"{export_dir} dizini oluşturuldu.")
else:
    print(f"{export_dir} dizini zaten mevcut.")


with distribution_strategy.scope():
  task = tfm.core.task_factory.get_task(exp_config.task, logging_dir=model_dir)


model, eval_logs = tfm.core.train_lib.run_experiment(
    distribution_strategy=distribution_strategy,
    task=task,
    mode='train_and_eval',
    params=exp_config,
    model_dir=model_dir,
    run_post_eval=True)


export_saved_model_lib.export_inference_graph(
    input_type='image_tensor',
    batch_size=1,
    input_image_size=[HEIGHT, WIDTH],
    params=exp_config,
    checkpoint_path=tf.train.latest_checkpoint(model_dir),
    export_dir=export_dir)

# # 1. Model Configurations
# def get_model_config():
#     config = retinanet_cfg.R
#     config = efficientdet_cfg.EfficientDetConfig()
#     config.model.name = "efficientdet-d0"
#     config.model.num_classes = 10  # Sınıf sayınızı burada ayarlayın
#     config.train_data.batch_size = 32
#     config.train_data.is_training = True
#     config.eval_data.batch_size = 32
#     config.runtime.mixed_precision_dtype = 'float32'
#     return config

# config = get_model_config()

# # Model oluşturma
# model = task_factory.get_task(config).build_model()

# # 2. TFRecord Okuma ve İşleme
# def parse_tfrecord(serialized_example):
#     feature_description = {
#         "image": tf.io.FixedLenFeature([], tf.string),
#         "label": tf.io.FixedLenFeature([], tf.int64),
#     }
#     parsed_example = tf.io.parse_single_example(serialized_example, feature_description)
#     image = tf.image.decode_jpeg(parsed_example["image"], channels=3)
#     image = tf.image.resize(image, [640, 640]) / 255.0  # Normalizasyon
#     label = parsed_example["label"]
#     return image, label

# # TFRecord'u okuma ve dataset oluşturma
# train_dataset = tf.data.TFRecordDataset("train.tfrecord")
# train_dataset = train_dataset.map(parse_tfrecord).shuffle(1000).batch(32)

# test_dataset = tf.data.TFRecordDataset("test.tfrecord")
# test_dataset = test_dataset.map(parse_tfrecord).batch(32)

# # 3. Modeli Eğitme
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
#     loss="sparse_categorical_crossentropy",
#     metrics=["accuracy"],
# )

# model.fit(
#     train_dataset,
#     validation_data=test_dataset,
#     epochs=10,
# )

# # 4. Modeli Kaydetme
# model.save("saved_model/efficientdet_custom")

