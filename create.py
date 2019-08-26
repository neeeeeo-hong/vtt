import tensorflow as tf
import collections
import tokenization
import os
import six
import random

class NLKnowledgeExample(object):
  def __init__(self, sentence, doc_tokens, subject, _property, value):
    self.sentence = sentence
    self.doc_tokens = doc_tokens
    self.subject = subject
    self.property = _property
    self.value = value

  def __str__(self):
    return self.__repr__()

  def __repr__(self):
    s = ""
    s += "sentence: %s" % (tokenization.printable_text(self.sentence))
    s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
    s += ", subject: %s" % (tokenization.printable_text(self.subject))
    s += ", property: %s" % (tokenization.printable_text(self.property))
    s += ", value: %s" % (tokenization.printable_text(self.value))
    return s


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               unique_id,
               example_index,
               doc_span_index,
               tokens,
               token_to_orig_map,
               token_is_max_context,
               input_ids,
               input_mask,
               segment_ids,
               subject=None,
               _property=None,
               value=None):
    self.unique_id = unique_id
    self.example_index = example_index
    self.doc_span_index = doc_span_index
    self.tokens = tokens
    self.token_to_orig_map = token_to_orig_map
    self.token_is_max_context = token_is_max_context
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.subject = subject
    self.property = _property
    self.value = value


class FeatureWriter(object):
  """Writes InputFeature to TF example file."""

  def __init__(self, filename, is_training):
    self.filename = filename
    self.is_training = is_training
    self.num_features = 0
    self._writer = tf.python_io.TFRecordWriter(filename)

  def process_feature(self, feature):
    """Write a InputFeature to the TFRecordWriter as a tf.train.Example."""
    self.num_features += 1

    def create_int_feature(values):
      feature = tf.train.Feature(
          int64_list=tf.train.Int64List(value=list(values)))
      return feature

    features = collections.OrderedDict()
    features["unique_ids"] = create_int_feature([feature.unique_id])
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)


    if self.is_training:
      features["subject"] = create_int_feature([feature.subject])
      features["property"] = create_int_feature([feature.property])
      features["value"] = create_int_feature([feature.value])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    self._writer.write(tf_example.SerializeToString())

  def close(self):
    self._writer.close()



class Create:
  def __init__(self):
    #self.FLAGS = FLAGS = tf.flags.FLAGS
    self.FLAGS = tf.flags.FLAGS

  
  def make_vocab_files(self, input_file, res_vocab_file, ont_vocab_file):
    with tf.gfile.GFile(input_file, "r") as reader:
      input_data = reader.readlines()

    resource = []
    ontology = []

    for entry in input_data:
      _result = entry.split('\t')[1].strip()
      _subject = _result.split(' ')[0]
      _property = _result.split(' ')[1]
      _value = _result.split(' ')[2]

      resource.append(_subject)
      ontology.append(_property)
      resource.append(_value)

      resource = list(set(resource))
      with tf.gfile.GFile(res_vocab_file, "w") as res:
        res.write('\n'.join(resource))

      ontology = list(set(ontology))
      with tf.gfile.GFile(ont_vocab_file, "w") as ont:
        ont.write('\n'.join(ontology))


  def read_NLK_examples(self, input_file):
    with tf.gfile.GFile(input_file, "r") as reader:
      input_data = reader.readlines()

    def is_whitespace(c):
      if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
      return False

    examples = []
    for entry in input_data:
      _sentence = entry.split('\t')[0].strip()
      _result = entry.split('\t')[1].strip()
      _subject = _result.split(' ')[0]
      _property = _result.split(' ')[1]
      _value = _result.split(' ')[2]

      doc_tokens = []
      prev_is_whitespace = True
      for c in _sentence:
        if is_whitespace(c):
          prev_is_whitespace = True
        else:
          if prev_is_whitespace:
            doc_tokens.append(c)
          else:
            doc_tokens[-1] += c
          prev_is_whitespace = False

      example = NLKnowledgeExample(
                sentence=_sentence,
                doc_tokens=doc_tokens,
                subject=_subject,
                _property=_property,
                value=_value)
      examples.append(example)

    return examples


  def convert_examples_to_features(self, examples, resource, ontology, tokenizer, max_seq_length,
                                   doc_stride, max_query_length, is_training, output_fn):

    unique_id = 1000000000

    for (example_index, example) in enumerate(examples):
      query_tokens = tokenizer.tokenize(example.sentence)

      if len(query_tokens) > max_query_length:
        query_tokens = query_tokens[0:max_query_length]

      tok_to_orig_index = []
      orig_to_tok_index = []
      all_doc_tokens = []
      for (i, token) in enumerate(example.doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
          tok_to_orig_index.append(i)
          all_doc_tokens.append(sub_token)

      subject = resource.index(example.subject)
      _property = ontology.index(example.property)
      value = resource.index(example.value)

      # The -2 accounts for [CLS], [SEP]
      max_tokens_for_doc = max_seq_length - len(query_tokens) - 2

      # We can have documents that are longer than the maximum sequence length.
      # To deal with this we do a sliding window approach, where we take chunks
      # of the up to our max length with a stride of `doc_stride`.
      _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
          "DocSpan", ["start", "length"])
      doc_spans = []
      start_offset = 0
      while start_offset < len(all_doc_tokens):
        length = len(all_doc_tokens) - start_offset
        if length > max_tokens_for_doc:
          length = max_tokens_for_doc
        doc_spans.append(_DocSpan(start=start_offset, length=length))
        if start_offset + length == len(all_doc_tokens):
          break
        start_offset += min(length, doc_stride)

      for (doc_span_index, doc_span) in enumerate(doc_spans):
        tokens = []
        token_to_orig_map = {}
        token_is_max_context = {}
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in query_tokens:
          tokens.append(token)
          segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
          input_ids.append(0)
          input_mask.append(0)
          segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length


        if example_index % 200 == 0:
          tf.logging.info("*** Example ***")
          tf.logging.info("unique_id: %s" % (unique_id))
          tf.logging.info("example_index: %s" % (example_index))
          tf.logging.info("doc_span_index: %s" % (doc_span_index))
          tf.logging.info("tokens: %s" % " ".join(
              [tokenization.printable_text(x) for x in tokens]))
          tf.logging.info("token_to_orig_map: %s" % " ".join(
              ["%d:%d" % (x, y) for (x, y) in six.iteritems(token_to_orig_map)]))
          tf.logging.info("token_is_max_context: %s" % " ".join([
              "%d:%s" % (x, y) for (x, y) in six.iteritems(token_is_max_context)
          ]))
          tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
          tf.logging.info(
              "input_mask: %s" % " ".join([str(x) for x in input_mask]))
          tf.logging.info(
              "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
          tf.logging.info("subject: %s" % (subject))
          tf.logging.info("property: %s" % (_property))
          tf.logging.info("value: %s" % (value))


#        with tf.gfile.GFile('./Tokens_ids.txt', "a") as token_file:
          #token_file.write(example.sentence)
#          token_file.write(" ".join([str(x) for x in query_tokens]) + '\n')
#          token_file.write(" ".join([str(x) for x in input_ids]) + '\n')

#        with tf.gfile.GFile('./spv_ids.txt', "a") as spv_file:
#          spv_file.write(str(example.subject) + " " + str(subject) + ' ')
#          spv_file.write(str(example.property) + " " + str(_property) + ' ')
#          spv_file.write(str(example.value) + " " + str(value) + '\n')

        feature = InputFeatures(
            unique_id=unique_id,
            example_index=example_index,
            doc_span_index=doc_span_index,
            tokens=tokens,
            token_to_orig_map=token_to_orig_map,
            token_is_max_context=token_is_max_context,
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            subject=subject,
            _property=_property,
            value=value)

        # Run callback
        output_fn(feature)

        unique_id += 1



  def create(self):
    FLAGS = self.FLAGS

    tf.logging.set_verbosity(tf.logging.INFO)

    tf.gfile.MakeDirs(FLAGS.output_dir)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)


    if not (os.path.exists(FLAGS.res_vocab_file) and os.path.exists(FLAGS.ont_vocab_file)):
      self.make_vocab_files(input_file=FLAGS.train_file, res_vocab_file=FLAGS.res_vocab_file,
          ont_vocab_file=FLAGS.ont_vocab_file)

    
    with tf.gfile.GFile(FLAGS.res_vocab_file, "r") as res:
      resource = res.read().splitlines()

    with tf.gfile.GFile(FLAGS.ont_vocab_file, "r") as ont:
      ontology = ont.read().splitlines()
    


    if FLAGS.do_train:
      train_examples = self.read_NLK_examples(input_file=FLAGS.train_file)

      rng = random.Random(12345)
      rng.shuffle(train_examples)

      train_writer = FeatureWriter(
          filename=os.path.join(FLAGS.output_dir, "train.tf_record"),
          is_training=True)

      self.convert_examples_to_features(
          examples=train_examples,
          resource=resource,
          ontology=ontology,
          tokenizer=tokenizer,
          max_seq_length=FLAGS.max_seq_length,
          doc_stride=FLAGS.doc_stride,
          max_query_length=FLAGS.max_query_length,
          is_training=True,
          output_fn=train_writer.process_feature)
      train_writer.close()



    if FLAGS.do_predict:
      eval_examples = self.read_NLK_examples(input_file=FLAGS.predict_file)
      eval_writer = FeatureWriter(
          filename=os.path.join(FLAGS.output_dir, "eval.tf_record"),
          is_training=False)

      eval_features = []
      def append_feature(feature):
        eval_features.append(feature)
        eval_writer.process_feature(feature)

      self.convert_examples_to_features(
          examples=eval_examples,
          resource=resource,
          ontology=ontology,
          tokenizer=tokenizer,
          max_seq_length=FLAGS.max_seq_length,
          doc_stride=FLAGS.doc_stride,
          max_query_length=FLAGS.max_query_length,
          is_training=False,
          output_fn=append_feature)
      eval_writer.close()


