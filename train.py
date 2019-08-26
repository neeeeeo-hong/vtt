import tensorflow as tf
import modeling
import tokenization
import subprocess
import os
import optimization

class NLKExtract:
  def __init__(self):
    #self.FLAGS = FLAGS = tf.flags.FLAGS
    self.FLAGS = tf.flags.FLAGS


  def validate_flags_or_throw(self, bert_config):
    """Validate the input FLAGS or throw an exception."""
    FLAGS = self.FLAGS

    tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                  FLAGS.init_checkpoint)

    if not FLAGS.do_train and not FLAGS.do_predict:
      raise ValueError("At least one of `do_train` or `do_predict` must be True.")

    if FLAGS.do_train:
      if not FLAGS.train_file:
        raise ValueError(
            "If `do_train` is True, then `train_file` must be specified.")
    if FLAGS.do_predict:
      if not FLAGS.predict_file:
        raise ValueError(
            "If `do_predict` is True, then `predict_file` must be specified.")

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
      raise ValueError(
          "Cannot use sequence length %d because the BERT model "
          "was only trained up to sequence length %d" %
          (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    if FLAGS.max_seq_length <= FLAGS.max_query_length + 3:
      raise ValueError(
          "The max_seq_length (%d) must be greater than max_query_length "
          "(%d) + 3" % (FLAGS.max_seq_length, FLAGS.max_query_length))


  def create_model(self, bert_config, is_training, input_ids, input_mask, segment_ids,
                 params, use_one_hot_embeddings):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    final_hidden = model.get_sequence_output()

    final_hidden_shape = modeling.get_shape_list(final_hidden, expected_rank=3)
    batch_size = final_hidden_shape[0]
    seq_length = final_hidden_shape[1]
    hidden_size = final_hidden_shape[2]

    output_weights = tf.get_variable(
        "cls/squad/output_weights", [3, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "cls/squad/output_bias", [3], initializer=tf.zeros_initializer())

    final_hidden_matrix = tf.reshape(final_hidden,
                                   [batch_size * seq_length, hidden_size])


    resource_weights = tf.get_variable(
        "NLK/resource_weights", [params["res_length"], seq_length],#12435
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    resource_bias = tf.get_variable(
        "NLK/resource_bias", [params["res_length"]], initializer=tf.zeros_initializer())

    ontology_weights = tf.get_variable(
        "NLK/ontology_weights", [params["ont_length"], seq_length],#324
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    ontology_bias = tf.get_variable(
        "NLK/ontology_bias", [params["ont_length"]], initializer=tf.zeros_initializer())


    logits = tf.matmul(final_hidden_matrix, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)


    logits = tf.reshape(logits, [batch_size, seq_length, 3])
    logits = tf.transpose(logits, [2, 0, 1])


    unstacked_logits = tf.unstack(logits, axis=0)

    (subject_logits, property_logits, value_logits) = (unstacked_logits[0], unstacked_logits[1], unstacked_logits[2])


    subject_logits = tf.matmul(subject_logits, resource_weights, transpose_b=True)
    subject_logits = tf.nn.bias_add(subject_logits, resource_bias)


    property_logits = tf.matmul(property_logits, ontology_weights, transpose_b=True)
    property_logits = tf.nn.bias_add(property_logits, ontology_bias)


    value_logits = tf.matmul(value_logits, resource_weights, transpose_b=True)
    value_logits = tf.nn.bias_add(value_logits, resource_bias)

    return (subject_logits, property_logits, value_logits)



  def model_fn_builder(self, bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
      """The `model_fn` for TPUEstimator."""

      tf.logging.info("*** Features ***")
      for name in sorted(features.keys()):
        tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

      unique_ids = features["unique_ids"]
      input_ids = features["input_ids"]
      input_mask = features["input_mask"]
      segment_ids = features["segment_ids"]

      is_training = (mode == tf.estimator.ModeKeys.TRAIN)

      (subject_logits, property_logits, value_logits) = self.create_model(
          bert_config=bert_config,
          is_training=is_training,
          input_ids=input_ids,
          input_mask=input_mask,
          segment_ids=segment_ids,
          params=params,
          use_one_hot_embeddings=use_one_hot_embeddings)

      tvars = tf.trainable_variables()

      initialized_variable_names = {}
      scaffold_fn = None
      if init_checkpoint:
        (assignment_map, initialized_variable_names
        ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
        if use_tpu:

          def tpu_scaffold():
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
            return tf.train.Scaffold()

          scaffold_fn = tpu_scaffold
        else:
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

      tf.logging.info("**** Trainable Variables ****")
      for var in tvars:
        init_string = ""
        if var.name in initialized_variable_names:
          init_string = ", *INIT_FROM_CKPT*"
        tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                        init_string)

      output_spec = None
      if mode == tf.estimator.ModeKeys.TRAIN:
        seq_length = modeling.get_shape_list(input_ids)[1]

        def compute_loss(logits, positions, depth):
          one_hot_positions = tf.one_hot(
              positions, depth=depth, dtype=tf.float32)
          log_probs = tf.nn.log_softmax(logits, axis=-1)
          loss = -tf.reduce_mean(
              tf.reduce_sum(one_hot_positions * log_probs, axis=-1))
          return loss

        # subject, property, value로 나오도록
        subject_label = features["subject"]
        property_label = features["property"]
        value_label = features["value"]
        res_length = params["res_length"]
        ont_length = params["ont_length"]

        subject_loss = compute_loss(subject_logits, subject_label, res_length)
        property_loss = compute_loss(property_logits, property_label, ont_length)
        value_loss = compute_loss(value_logits, value_label, res_length)

        total_loss = (subject_loss + property_loss + value_loss) / 3.0

        train_op = optimization.create_optimizer(
            total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

        output_spec = tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode,
            loss=total_loss,
            train_op=train_op,
            scaffold_fn=scaffold_fn)
      elif mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            "unique_ids": unique_ids,
            "subject_logits": subject_logits,
            "property_logits": property_logits,
            "value_logits": value_logits,
        }
        output_spec = tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
      else:
        raise ValueError(
            "Only TRAIN and PREDICT modes are supported: %s" % (mode))

      return output_spec

    return model_fn



  def input_fn_builder(self, input_file, seq_length, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "unique_ids": tf.FixedLenFeature([], tf.int64),
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
    }

    if is_training:
      name_to_features["subject"] = tf.FixedLenFeature([], tf.int64)
      name_to_features["property"] = tf.FixedLenFeature([], tf.int64)
      name_to_features["value"] = tf.FixedLenFeature([], tf.int64)

    def _decode_record(record, name_to_features):
      """Decodes a record to a TensorFlow example."""
      example = tf.parse_single_example(record, name_to_features)

      # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
      # So cast all int64 to int32.
      for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
          t = tf.to_int32(t)
        example[name] = t

      return example

    def input_fn(params):
      """The actual input function."""
      batch_size = params["batch_size"]

      # For training, we want a lot of parallel reading and shuffling.
      # For eval, we want no shuffling and parallel reading doesn't matter.
      d = tf.data.TFRecordDataset(input_file)
      if is_training:
        d = d.repeat()
        d = d.shuffle(buffer_size=100)

      d = d.apply(
          tf.contrib.data.map_and_batch(
              lambda record: _decode_record(record, name_to_features),
              batch_size=batch_size,
              drop_remainder=drop_remainder))

      return d

    return input_fn



  def train(self):
    FLAGS = self.FLAGS

    tf.logging.set_verbosity(tf.logging.INFO)

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    self.validate_flags_or_throw(bert_config)

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
      tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
          FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    train_file_length = int(subprocess.check_output(["wc", "-l", FLAGS.train_file], 
        universal_newlines=True).split(' ')[0])
    #print(train_file_length)

    num_train_steps = int(
        train_file_length / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    params = {}
    params["res_length"] = int(subprocess.check_output(["wc", "-l", FLAGS.res_vocab_file],
         universal_newlines=True).split(' ')[0])
    params["ont_length"] = int(subprocess.check_output(["wc", "-l", FLAGS.ont_vocab_file],
         universal_newlines=True).split(' ')[0])

    #print(params["res_length"])

    model_fn = self.model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)


    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        params=params,
        train_batch_size=FLAGS.train_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)


    train_input_fn = self.input_fn_builder(
        input_file=os.path.join(FLAGS.output_dir, "train.tf_record"),
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=True)
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
