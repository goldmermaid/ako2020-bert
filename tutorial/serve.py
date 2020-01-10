import json, logging, warnings
import gluonnlp as nlp
import mxnet as mx


def model_fn(model_dir):
    """
    Load the gluon model. Called once when hosting service starts.
    :param: model_dir The directory where model files are stored.
    :return: a Gluon model, and the vocabulary
    """
    prefix = 'checkpoint'
    net = mx.gluon.nn.SymbolBlock.imports(prefix + '-symbol.json',
                                          ['data0', 'data1', 'data2'],
                                          prefix + '-0000.params')
    net.load_parameters('%s/' % model_dir + prefix + '-0000.params',
                        ctx=mx.cpu())
    vocab_json = open('%s/vocab.json' % model_dir).read()
    vocab = nlp.Vocab.from_json(vocab_json)
    tokenizer = nlp.data.BERTTokenizer(vocab)
    transform = nlp.data.BERTSentenceTransform(tokenizer, max_seq_length=128,
                                               pair=False, pad=False)
    return net, vocab, transform


def transform_fn(model, data, input_content_type, output_content_type):
    """
    Transform a request using the Gluon model. Called once per request.
    :param model: The Gluon model and the vocab
    :param data: The request payload.
    :param input_content_type: The request content type.
    :param output_content_type: The (desired) response content type.
    :return: response payload and content type.
    """
    # we can use content types to vary input/output handling, but
    # here we just assume json for both
    net, vocabulary, transform = model
    sentence = json.loads(data)
    result = predict_sentiment(net, mx.cpu(), transform, sentence)
    response_body = json.dumps(result)
    return response_body, output_content_type


def predict_sentiment(net, ctx, transform, sentence):
    ctx = ctx[0] if isinstance(ctx, list) else ctx
    inputs, seq_len, token_types = transform([sentence])
    inputs = mx.nd.array([inputs], ctx=ctx)
    token_types = mx.nd.array([token_types], ctx=ctx)
    seq_len = mx.nd.array([seq_len], ctx=ctx)
    out = net(inputs, token_types, seq_len)
    label = mx.nd.argmax(out, axis=1)
    return 'positive' if label.asscalar() == 1 else 'negative'