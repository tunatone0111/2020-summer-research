import functools
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

keras = tf.keras
tfd = tfp.distributions
# %%
def tfp_layer_with_scaled_kl(layer_builder, num_train_examples):
    def scaled_kl_fn(q, p, _):
        return tfd.kl_divergence(q, p) / num_train_examples

    return functools.partial(layer_builder, kernel_divergence_fn=scaled_kl_fn)


def _posterior_mean_field(kernel_size, bias_size=0, dtype=None):
  """Posterior function for variational layer."""
  n = kernel_size + bias_size
  c = np.log(np.expm1(1e-5))
  variable_layer = tfp.layers.VariableLayer(
      2 * n, dtype=dtype,
      initializer=tfp.layers.BlockwiseInitializer([
          keras.initializers.TruncatedNormal(mean=0., stddev=.05, seed=None),
          keras.initializers.Constant(np.log(np.expm1(1e-5)))], sizes=[n, n]))

  def distribution_fn(t):
    scale = 1e-5 + tf.nn.softplus(c + t[Ellipsis, n:])
    return tfd.Independent(tfd.Normal(loc=t[Ellipsis, :n], scale=scale),
                           reinterpreted_batch_ndims=1)
  distribution_layer = tfp.layers.DistributionLambda(distribution_fn)
  return tf.keras.Sequential([variable_layer, distribution_layer])

def _make_prior_fn(kernel_size, bias_size=0, dtype=None):
  del dtype  # TODO(yovadia): Figure out what to do with this.
  loc = tf.zeros(kernel_size + bias_size)
  def distribution_fn(_):
    return tfd.Independent(tfd.Normal(loc=loc, scale=1),
                           reinterpreted_batch_ndims=1)
  return distribution_fn


def get_layer_builders(method, num_train_examples):
    """Get method-appropriate functions for building and/or applying Keras layers.
    
    Args:
        method: UQ method (vanilla, svi).
        num_train_examples: Number of training examples. Used to scale KL loss.    
    Returns:
        conv2d, dense_layer
    """
    tfpl = tfp.layers

    conv2d_variational = tfp_layer_with_scaled_kl(tfpl.Convolution2DFlipout,
                                                num_train_examples)
  # Only DenseVariational works in v2 / eager mode.
  # FMI: https://github.com/tensorflow/probability/issues/409
    if tf.executing_eagerly():
        def dense_variational(units, activation):
            return tfpl.DenseVariational(
                units,
                make_posterior_fn=_posterior_mean_field,
                make_prior_fn=_make_prior_fn,
                activation=activation,
                kl_weight=1./num_train_examples)
    else:
        dense_variational = tfp_layer_with_scaled_kl(tfpl.DenseFlipout,
                                                     num_train_examples)
    
    if method == 'svi':
        return conv2d_variational, dense_variational
    else:
        return keras.layers.Conv2D, keras.layers.Dense
# %%
class ModelOptions(object):
  """Parameters for model construction and fitting."""
  train_epochs = attr.ib()
  num_train_examples = attr.ib()
  batch_size = attr.ib()
  learning_rate = attr.ib()
  method = attr.ib()
  architecture = attr.ib()
  mlp_layer_sizes = attr.ib()
  num_examples_for_predict = attr.ib()
  predictions_per_example = attr.ib()

def _build_lenet(opts):
  """Builds a LeNet Keras model."""
  layer_builders = uq_utils.get_layer_builders(opts.method,
                                               opts.num_train_examples)
  conv2d, dense_layer = layer_builders

  inputs = keras.layers.Input(_MNIST_SHAPE)
  net = inputs
  net = conv2d(32, kernel_size=(3, 3),
               activation='relu',
               input_shape=_MNIST_SHAPE)(net)
  net = conv2d(64, (3, 3), activation='relu')(net)
  net = keras.layers.MaxPooling2D(pool_size=(2, 2))(net)
  net = keras.layers.Flatten()(net)
  net = dense_layer(128, activation='relu')(net)
  logits = dense_layer(_NUM_CLASSES)(net)

  return keras.Model(inputs=inputs, outputs=logits)



def build_model(opts):
  """Builds (uncompiled) Keras model from ModelOptions instance."""
  return {'mlp': _build_mlp, 'lenet': _build_lenet}[opts.architecture](opts)

def build_and_train(opts, dataset_train, dataset_eval, output_dir):
  """Returns a trained MNIST model and saves it to output_dir.

  Args:
    opts: ModelOptions
    dataset_train: Pair of images, labels np.ndarrays for training.
    dataset_eval: Pair of images, labels np.ndarrays for continuous eval.
    output_dir: Directory for the saved model and tensorboard events.
  Returns:
    Trained Keras model.
  """
  model = build_model(opts)
  model.compile(
      keras.optimizers.Adam(opts.learning_rate),
      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=['accuracy'],
  )

  tensorboard_cb = keras.callbacks.TensorBoard(
      log_dir=output_dir, write_graph=False)

  train_images, train_labels = dataset_train
  assert len(train_images) == opts.num_train_examples, (
      '%d != %d' % (len(train_images), opts.num_train_examples))
  model.fit(
      train_images, train_labels,
      epochs=opts.train_epochs,
      # NOTE: steps_per_epoch will cause OOM for some reason.
      validation_data=dataset_eval,
      batch_size=opts.batch_size,
      callbacks=[tensorboard_cb],
  )
  return model



def make_predictions(opts, model, dataset):
  """Build a dictionary of model predictions on a given dataset.

  Args:
    opts: ModelOptions.
    model: Trained Keras model.
    dataset: tf.data.Dataset of <image, label> pairs.
  Returns:
    Dictionary containing labels and model logits.
  """
  if opts.num_examples_for_predict:
    dataset = tuple(x[:opts.num_examples_for_predict] for x in dataset)

  batched_dataset = (tf.data.Dataset.from_tensor_slices(dataset)
                     .batch(_BATCH_SIZE_FOR_PREDICT))
  out = collections.defaultdict(list)
  for images, labels in tfds.as_numpy(batched_dataset):
    logits_samples = np.stack(
        [model.predict(images) for _ in range(opts.predictions_per_example)],
        axis=1)  # shape: [batch_size, num_samples, num_classes]
    probs = scipy.special.softmax(logits_samples, axis=-1).mean(-2)
    out['labels'].extend(labels)
    out['logits_samples'].extend(logits_samples)
    out['probs'].extend(probs)
    if len(out['image_examples']) < _NUM_IMAGE_EXAMPLES_TO_RECORD:
      out['image_examples'].extend(images)

  return {k: np.stack(a) for k, a in six.iteritems(out)}



def get_experiment_config(method, architecture,
                          test_level, output_dir=None):
  """Returns model and data configs."""
  data_opts_list = data_lib.DATA_OPTIONS_LIST
  if test_level:
    data_opts_list = data_opts_list[:4]

  model_opts = hparams_lib.get_tuned_model_options(architecture, method,
                                                   fake_data=test_level > 1,
                                                   fake_training=test_level > 0)
  if output_dir:
    experiment_utils.record_config(model_opts, output_dir+'/model_options.json')
  return model_opts, data_opts_list



def run(method, architecture, output_dir, test_level):
  """Trains a model and records its predictions on configured datasets.

  Args:
    method: Name of modeling method (vanilla, dropout, svi, ll_svi).
    architecture: Name of DNN architecture (mlp or dropout).
    output_dir: Directory to record the trained model and output stats.
    test_level: Zero indicates no testing. One indicates testing with real data.
      Two is for testing with fake data.
  """
  fake_data = test_level > 1
  gfile.makedirs(output_dir)
  model_opts, data_opts_list = get_experiment_config(method, architecture,
                                                     test_level=test_level,
                                                     output_dir=output_dir)

  # Separately build dataset[0] with shuffle=True for training.
  dataset_train = data_lib.build_dataset(data_opts_list[0], fake_data=fake_data)
  dataset_eval = data_lib.build_dataset(data_opts_list[1], fake_data=fake_data)
  model = models_lib.build_and_train(model_opts,
                                     dataset_train, dataset_eval, output_dir)
  logging.info('Saving model to output_dir.')
  model.save_weights(output_dir + '/model.ckpt')

  for idx, data_opts in enumerate(data_opts_list):
    dataset = data_lib.build_dataset(data_opts, fake_data=fake_data)
    logging.info('Running predictions for dataset #%d', idx)
    stats = models_lib.make_predictions(model_opts, model, dataset)
    array_utils.write_npz(output_dir, 'stats_%d.npz' % idx, stats)
    del stats['logits_samples']
    array_utils.write_npz(output_dir, 'stats_small_%d.npz' % idx, stats)
