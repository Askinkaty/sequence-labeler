import sys
import collections
import numpy
import random
import math
import os
import gc
import codecs
import json
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score

try:
    import ConfigParser as configparser
except:
    import configparser


from labeler import SequenceLabeler
from evaluator import SequenceLabelingEvaluator
from embedder import Model, get_token_embeddings


def filter_sentences(line_parts):
    result = []
    word = line_parts[0]
    label = line_parts[1]
    if len(word):
        word = word.replace('...', '.').replace('..', '.').replace('""""', '"').replace('"""', '"').replace('\xad', '')
        word = word.replace('_', '')
        if '-' in word and len(word) > 1 and not word.endswith('-') and not word.startswith('-'):  # here we split hyphenated tokens by hyphen
            word_list = word.split('-') #  hyphen can get incorrect label as well here
            if len(word_list) == 2:
                result.append([word_list[0].strip(), label])
                result.append(['-', label])
                result.append([word_list[1].strip(), label])
        elif len(word) > 1 and (word.endswith('-') or word.startswith('-')):
            word = word.replace('-', '')
            result.append([word, label])
        else:
            if len(word):
                result.append([word, label])
    return result


def read_input_files(file_paths, max_sentence_length=-1):
    """
    Reads input files in whitespace-separated format.
    Will split file_paths on comma, reading from multiple files.
    The format assumes the first column is the word, the last column is the label.
    """
    sentences = []
    line_length = None
    for file_path in file_paths.strip().split(","):
        with codecs.open(file_path, "r", encoding='utf-8') as f:
            sentence = []
            for line in f:
                line = line.strip()
                if len(line) > 0:
                    line_parts = line.split()
                    try:
                        assert(len(line_parts) >= 2)
                    except:
                        print(line_parts)
                    try:
                        assert(len(line_parts) == line_length or line_length == None)
                    except:
                        print(line_parts, line_length)
                    line_parts = [el for el in line_parts if len(el)]
                    line_length = len(line_parts)
                    filtered_parts = filter_sentences(line_parts)
                    for fp in filtered_parts:
                        sentence.append(fp)
                elif len(line) == 0 and len(sentence) > 0:
                    if max_sentence_length <= 0 or len(sentence) <= max_sentence_length:
                        sentences.append(sentence)
                    sentence = []
            if len(sentence) > 0:
                if max_sentence_length <= 0 or len(sentence) <= max_sentence_length:
                    sentences.append(sentence)
    return sentences

def parse_config(config_section, config_path):
    """
    Reads configuration from the file and returns a dictionary.
    Tries to guess the correct datatype for each of the config values.
    """
    config_parser = configparser.SafeConfigParser(allow_no_value=True)
    config_parser.read(config_path)
    config = collections.OrderedDict()
    for key, value in config_parser.items(config_section):
        if value is None or len(value.strip()) == 0:
            config[key] = None
        elif value.lower() in ["true", "false"]:
            config[key] = config_parser.getboolean(config_section, key)
        elif value.isdigit():
            config[key] = config_parser.getint(config_section, key)
        elif is_float(value):
            config[key] = config_parser.getfloat(config_section, key)
        else:
            config[key] = config_parser.get(config_section, key)
    return config


def is_float(value):
    """
    Check in value is of type float()
    """
    try:
        float(value)
        return True
    except ValueError:
        return False


def create_batches_of_sentence_ids(sentences, batch_equal_size, max_batch_size):
    """
    Groups together sentences into batches
    If batch_equal_size is True, make all sentences in a batch be equal length.
    If max_batch_size is positive, this value determines the maximum number of sentences in each batch.
    If max_batch_size has a negative value, the function dynamically creates the batches such that each batch contains abs(max_batch_size) words.
    Returns a list of lists with sentences ids.
    """
    batches_of_sentence_ids = []
    if batch_equal_size == True:
        sentence_ids_by_length = collections.OrderedDict()
        sentence_length_sum = 0.0
        for i in range(len(sentences)):
            length = len(sentences[i])
            if length not in sentence_ids_by_length:
                sentence_ids_by_length[length] = []
            sentence_ids_by_length[length].append(i)

        for sentence_length in sentence_ids_by_length:
            if max_batch_size > 0:
                batch_size = max_batch_size
            else:
                batch_size = int((-1.0 * max_batch_size) / sentence_length)

            for i in range(0, len(sentence_ids_by_length[sentence_length]), batch_size):
                batches_of_sentence_ids.append(sentence_ids_by_length[sentence_length][i:i + batch_size])
    else:
        current_batch = []
        max_sentence_length = 0
        for i in range(len(sentences)):
            current_batch.append(i)
            if len(sentences[i]) > max_sentence_length:
                max_sentence_length = len(sentences[i])
            if (max_batch_size > 0 and len(current_batch) >= max_batch_size) \
              or (max_batch_size <= 0 and len(current_batch)*max_sentence_length >= (-1 * max_batch_size)):
                batches_of_sentence_ids.append(current_batch)
                current_batch = []
                max_sentence_length = 0
        if len(current_batch) > 0:
            batches_of_sentence_ids.append(current_batch)
    return batches_of_sentence_ids


def get_vectors(config, mode):
    path_to_embeddings = config['emb_path']
    file_name = mode + '.jsonl'
    token_file_name = mode + '_tokens.jsonl'
    embedding_list = []
    token_list = []
    with open(os.path.join(path_to_embeddings, file_name), 'r') as f:
        for line in f:
            sent = json.loads(line)
            sent = [numpy.asarray(el, dtype=numpy.float32) for el in sent]
            embedding_list.append(sent)
    with open(os.path.join(path_to_embeddings, token_file_name), 'r') as tf:
        for line in tf:
            sent = json.loads(line)
            token_list.append(sent)
    return embedding_list, token_list


def process_sentences(data, labeler, is_training, learningrate, config, name):
    """
    Process all the sentences with the labeler, return evaluation metrics.
    """
    evaluator = SequenceLabelingEvaluator(config["main_label"], labeler.label2id, config["conll_eval"])
    batches_of_sentence_ids = create_batches_of_sentence_ids(data, config["batch_equal_size"], config["max_batch_size"])
    embeddings, token_list = get_vectors(config, name)
    print(len(data))
    print(len(embeddings))
    assert len(embeddings) == len(data)
    if is_training is True:
        random.shuffle(batches_of_sentence_ids)

    for sentence_ids_in_batch in batches_of_sentence_ids:
        batch = [data[i] for i in sentence_ids_in_batch]
        cost, predicted_labels, predicted_probs = labeler.process_batch(batch, sentence_ids_in_batch,
                                                                        is_training,
                                                                        learningrate,
                                                                        embeddings,
                                                                        token_list)

        true, predicted = evaluator.append_data(cost, batch, predicted_labels)

        word_ids, char_ids, char_mask, label_ids = None, None, None, None
        while config["garbage_collection"] is True and gc.collect() > 0:
            pass

    results, conll_format_preds = evaluator.get_results(name)

    for key in results:
        print(key + ": " + str(results[key]))

    return results, conll_format_preds, true, predicted


def write_batch(model, batch, file, token_file):
    out_tokens, out_vertors = model.get_features(batch)
    batch_tokens, batch_tokens_embeddings = get_token_embeddings(out_tokens, out_vertors)
    assert len(batch_tokens_embeddings) == len(batch) == len(batch_tokens)
    for k, sent in enumerate(batch_tokens_embeddings):
        file.write(json.dumps([e.tolist() for e in sent], ensure_ascii=False))
        file.write('\n')
        token_file.write(json.dumps(batch_tokens[k], ensure_ascii=False))
        token_file.write('\n')


def get_and_save_bert_embeddings(sentences, out_path, model, mode):
    sent_batch = []
    n = 32
    out_file = mode + '.jsonl'
    out_token_file = mode + '_tokens.jsonl'
    c = 0
    with codecs.open(os.path.join(out_path, out_file), 'w', encoding='utf-8') as f:
        with codecs.open(os.path.join(out_path, out_token_file), 'w', encoding='utf-8') as tok_f:
            for i, sentence in enumerate(sentences):
                if len(sent_batch) < n:
                    sentence = ' '.join([el[0] for el in sentence]).strip()
                    sent_batch.append(sentence)
                    c += 1
                if len(sent_batch) == n or i == (len(sentences) - 1):
                    write_batch(model, sent_batch, f, tok_f)
                    sent_batch = []
    print('sentences: ', len(sentences))
    assert len(sentences) == c


def combine_train(cv_path, train_files, i):
    with codecs.open(os.path.join(cv_path, 'train' + str(i) + '.csv'), "w") as f_out:
        for t_f in train_files:
            with codecs.open(os.path.join(cv_path, t_f), "r", encoding='utf-8') as f:
                for line in f:
                    f_out.write(line)


def get_train_test_dev(data_path, path_train, path_dev, path_test, config, bertModel):
    data_train = read_input_files(path_train, config["max_train_sent_length"])
    # if not os.path.exists(os.path.join(data_path, 'train.jsonl')):
    random.shuffle(data_train)
    get_and_save_bert_embeddings(data_train, config['emb_path'], bertModel, 'train')
    data_dev = read_input_files(path_dev)
    # if not os.path.exists(os.path.join(data_path, 'dev.jsonl')):
    get_and_save_bert_embeddings(data_dev, config['emb_path'], bertModel, 'dev')
    data_test = read_input_files(path_test)
    # if not os.path.exists(os.path.join(data_path, 'test.jsonl')):
    get_and_save_bert_embeddings(data_test, config['emb_path'], bertModel, 'test')
    return data_train, data_dev, data_test


def prepare_folds(fold_files, i, cv_path):
    dev_fold = fold_files[i]
    test_fold = fold_files[-1 + i]
    train_folds = []
    for j in range(len(fold_files)):
        if fold_files[j] != test_fold and fold_files[j] != dev_fold:
            train_folds.append(fold_files[j])
    print(train_folds)
    print(len(train_folds))
    print(len(fold_files))
    assert len(train_folds) == len(fold_files) - 2
    combine_train(cv_path, train_folds, i)
    print(f'Created fold {i}')
    print(dev_fold)
    print(test_fold)
    return dev_fold, test_fold


def save_results(config, results, true, predicted, i):
    precision, recall = get_macro_precision_recall(true, predicted)
    results['precision_macro'] = precision
    results['recall_macro'] = recall
    with codecs.open(os.path.join(config['cv_path'], 'result' + str(i) + '.json'), 'w') as out:
        out.write(json.dumps(results, ensure_ascii=False))


def save_test_fold_preds(config, conll_data, i):
    with codecs.open(os.path.join(config['cv_path'], 'conll_test_out_' + str(i) + '.txt'), 'w') as out:
        for el in conll_data:
            out.write(el + '\n')
            out.write('\n')


def remove_ebm_files(config):
    to_remove = ['train', 'train_tokens', 'dev', 'dev_tokens', 'test', 'test_tokens']
    for f in to_remove:
        os.remove(os.path.join(config['cv_path'], f + '.jsonl'))


def run_cv(config, config_path, bertModel):
    temp_model_path = config_path + ".model"
    cv_path = config['cv_path']
    fold_files = os.listdir(cv_path)
    all_results = []
    for i in range(len(fold_files)):
        tf.reset_default_graph()
        dev_file, test_file = prepare_folds(fold_files, i, cv_path)
        data_train, data_dev, data_test = get_train_test_dev(cv_path,
                                                             os.path.join(cv_path, 'train' + str(i) + '.csv'),
                                                             os.path.join(cv_path, dev_file),
                                                             os.path.join(cv_path, test_file), config, bertModel)
        labeler = load_model(config, data_train, data_dev, data_test)
        results_train, results_dev, results_test, conll_format_preds, true, predicted = interate_epochs(config,
                                                                                                        labeler,
                                                                                                        data_train,
                                                                                                        data_dev,
                                                                                                        data_test,
                                                                                                        temp_model_path)
        save_results(config, results_test, true, predicted, i)
        save_test_fold_preds(config, conll_format_preds, i)
        all_results.append((results_train, results_dev, results_test))
        remove_ebm_files(config)
        print(f'Done with fold: {i}')
    main_correct_counts = 0
    main_predicted_counts = 0
    main_total_counts = 0
    correct_sum = 0
    token_count = 0
    for f in all_results:
        test_result = f[2]
        main_correct_counts += test_result.get('test_main_correct_count')
        main_predicted_counts += test_result.get('test_main_predicted_count')
        main_total_counts += test_result.get('test_main_total_count')
        correct_sum += test_result.get('test_correct_sum')
        token_count += test_result.get('test_token_count')
    # calculate average results for all folds here
    total_p = (float(main_correct_counts) / float(main_predicted_counts)) if (main_predicted_counts > 0) else 0.0
    total_r = (float(main_correct_counts) / float(main_total_counts)) if (main_total_counts > 0) else 0.0
    f = (2.0 * total_p * total_r / (total_p + total_r)) if (total_p + total_r > 0.0) else 0.0
    f05 = ((1.0 + 0.5 * 0.5) * total_p * total_r / ((0.5 * 0.5 * total_p) + total_r)) if (total_p + total_r > 0.0) else 0.0
    accuracy = correct_sum / float(token_count)
    with codecs.open(os.path.join(config['cv_path'], 'cv_result.txt'), 'w') as final:
        final.write('CV Precision: ' + str(total_p) + '\n')
        final.write('CV Recall: ' + str(total_r) + '\n')
        final.write('CV F0.5: ' + str(f05) + '\n')
        final.write('F-measure: ' + str(f) + '\n')
        final.write('CV ACC: ' + str(accuracy) + '\n')
    print('CV Precision: ', total_p)
    print('CV Recall: ', total_r)
    print('CV F0.5: ', f05)
    print('CV F-measure: ', f)
    print('CV Accuracy: ', accuracy)


def load_model(config, data_train, data_dev, data_test):
    if config["load"] != None and len(config["load"]) > 0:
        labeler = SequenceLabeler.load(config["load"])
    else:
        labeler = SequenceLabeler(config)
        labeler.build_vocabs(data_train, data_dev, data_test, config["preload_vectors"])
        labeler.construct_network()
        labeler.initialize_session()
        if config["preload_vectors"] is not None:
            labeler.preload_word_embeddings(config["preload_vectors"])
    print("parameter_count: " + str(labeler.get_parameter_count()))
    print("parameter_count_without_word_embeddings: " + str(labeler.get_parameter_count_without_word_embeddings()))
    return labeler


def interate_epochs(config, labeler, data_train, data_dev, data_test, temp_model_path):
    if data_train is not None:
        model_selector = config["model_selector"].split(":")[0]
        model_selector_type = config["model_selector"].split(":")[1]
        best_selector_value = 0.0
        best_epoch = -1
        learningrate = config["learningrate"]
        for epoch in range(config["epochs"]):
            print("EPOCH: " + str(epoch))
            print("current_learningrate: " + str(learningrate))
            results_train, _, _, _ = process_sentences(data_train, labeler, is_training=True,
                                              learningrate=learningrate,
                                              config=config, name="train")

            if data_dev is not None:
                results_dev, _, _, _ = process_sentences(data_dev, labeler, is_training=False,
                                                learningrate=0.0,
                                                config=config, name="dev")

                if math.isnan(results_dev["dev_cost_sum"]) or math.isinf(results_dev["dev_cost_sum"]):
                    sys.stderr.write("ERROR: Cost is NaN or Inf. Exiting.\n")
                    break

                if (epoch == 0 or (model_selector_type == "high" and results_dev[model_selector] > best_selector_value)
                               or (model_selector_type == "low" and results_dev[model_selector] < best_selector_value)):
                    best_epoch = epoch
                    best_selector_value = results_dev[model_selector]
                    labeler.saver.save(labeler.session, temp_model_path,
                                       latest_filename=os.path.basename(temp_model_path)+".checkpoint")
                print("best_epoch: " + str(best_epoch))

                if config["stop_if_no_improvement_for_epochs"] > 0 \
                        and (epoch - best_epoch) >= config["stop_if_no_improvement_for_epochs"]:
                    break

                if (epoch - best_epoch) > 3:
                    learningrate *= config["learningrate_decay"]

            while config["garbage_collection"] == True and gc.collect() > 0:
                pass

        if data_dev is not None and best_epoch >= 0:
            # loading the best model so far
            labeler.saver.restore(labeler.session, temp_model_path)

            os.remove(temp_model_path+".checkpoint")
            os.remove(temp_model_path+".data-00000-of-00001")
            os.remove(temp_model_path+".index")
            os.remove(temp_model_path+".meta")

    if config["save"] is not None and len(config["save"]) > 0:
        labeler.save(config["save"])

    if data_test is not None:
        results_test, conll_format_preds, true, predicted = process_sentences(data_test, labeler, is_training=False,
                                         learningrate=0.0, config=config, name="test")
    return results_train, results_dev, results_test, conll_format_preds, true, predicted


def get_macro_precision_recall(true, predicted):
    p_macro = precision_score(true, predicted, average='macro')
    r_macro = recall_score(true, predicted, average='macro')
    return p_macro, r_macro

def run(config, config_path, bertModel):
    temp_model_path = config_path + ".model"
    data_train, data_dev, data_test = get_train_test_dev(config['emb_path'],
                                                         config['path_train'],
                                                         config['path_dev'],
                                                         config['path_test'], config, bertModel)

    labeler = load_model(config, data_train, data_dev, data_test)
    retults_train, results_dev, results_test, conll_format_preds, true, predicted = interate_epochs(config,
                                                                                                    labeler,
                                                                                                    data_train,
                                                                                                    data_dev,
                                                                                                    data_test,
                                                                                                    temp_model_path)
    save_results(config, results_test, true, predicted, 'test')
    save_test_fold_preds(config, conll_format_preds, 'test')
    remove_ebm_files(config)



def run_experiment_new(config_path):
    config = parse_config("config", config_path)
    if "random_seed" in config:
        random.seed(config["random_seed"])
        numpy.random.seed(config["random_seed"])

    print("Initializing BERT model for contextual embeddings... ")
    bertModel = Model()

    for key, val in config.items():
        print(str(key) + ": " + str(val))

    if config['cv']:
        run_cv(config, config_path, bertModel)
    else:
        run(config, config_path, bertModel)


# So I have no answer to the question why I think you're good. I'm just sure you're, that's it.
# Yep, apparently cannot mentalize everything.

def run_experiment(config_path):
    config = parse_config("config", config_path)
    if "random_seed" in config:
        random.seed(config["random_seed"])
        numpy.random.seed(config["random_seed"])

    print("Initializing BERT model for contextual embeddings... ")
    bertModel = Model()

    for key, val in config.items():
        print(str(key) + ": " + str(val))

    data_train, data_dev, data_test = None, None, None

    if config["path_train"] != None and len(config["path_train"]) > 0:
        data_train = read_input_files(config["path_train"], config["max_train_sent_length"])
        if not os.path.exists(os.path.join(config['path_train'], 'train.jsonl')):
            random.shuffle(data_train)
            get_and_save_bert_embeddings(data_train, config['emb_path'], bertModel, 'train')
    if config["path_dev"] != None and len(config["path_dev"]) > 0:
        data_dev = read_input_files(config["path_dev"])
        if not os.path.exists(os.path.join(config['path_dev'], 'dev.jsonl')):
            get_and_save_bert_embeddings(data_dev, config['emb_path'], bertModel, 'dev')
    if config["path_test"] != None and len(config["path_test"]) > 0:
        data_test = []
        for path_test in config["path_test"].strip().split(":"):
            data_test += read_input_files(path_test)
            if not os.path.exists(os.path.join(config['path_test'], 'test.jsonl')):
                get_and_save_bert_embeddings(data_test, config['emb_path'], bertModel, 'test')
    if config["load"] != None and len(config["load"]) > 0:
        labeler = SequenceLabeler.load(config["load"])
    else:
        labeler = SequenceLabeler(config)
        labeler.build_vocabs(data_train, data_dev, data_test, config["preload_vectors"])
        labeler.construct_network()
        labeler.initialize_session()
        if config["preload_vectors"] is not None:
            labeler.preload_word_embeddings(config["preload_vectors"])

    print("parameter_count: " + str(labeler.get_parameter_count()))
    print("parameter_count_without_word_embeddings: " + str(labeler.get_parameter_count_without_word_embeddings()))

    if data_train is not None:
        model_selector = config["model_selector"].split(":")[0]
        model_selector_type = config["model_selector"].split(":")[1]
        best_selector_value = 0.0
        best_epoch = -1
        learningrate = config["learningrate"]
        for epoch in range(config["epochs"]):
            print("EPOCH: " + str(epoch))
            print("current_learningrate: " + str(learningrate))
            results_train = process_sentences(data_train, labeler, is_training=True,
                                              learningrate=learningrate,
                                              config=config, name="train")

            if data_dev != None:
                results_dev = process_sentences(data_dev, labeler, is_training=False,
                                                learningrate=0.0,
                                                config=config, name="dev")

                if math.isnan(results_dev["dev_cost_sum"]) or math.isinf(results_dev["dev_cost_sum"]):
                    sys.stderr.write("ERROR: Cost is NaN or Inf. Exiting.\n")
                    break

                if (epoch == 0 or (model_selector_type == "high" and results_dev[model_selector] > best_selector_value) 
                               or (model_selector_type == "low" and results_dev[model_selector] < best_selector_value)):
                    best_epoch = epoch
                    best_selector_value = results_dev[model_selector]
                    labeler.saver.save(labeler.session, temp_model_path, latest_filename=os.path.basename(temp_model_path)+".checkpoint")
                print("best_epoch: " + str(best_epoch))

                if config["stop_if_no_improvement_for_epochs"] > 0 and (epoch - best_epoch) >= config["stop_if_no_improvement_for_epochs"]:
                    break

                if (epoch - best_epoch) > 3:
                    learningrate *= config["learningrate_decay"]

            while config["garbage_collection"] == True and gc.collect() > 0:
                pass

        if data_dev != None and best_epoch >= 0:
            # loading the best model so far
            labeler.saver.restore(labeler.session, temp_model_path)

            os.remove(temp_model_path+".checkpoint")
            os.remove(temp_model_path+".data-00000-of-00001")
            os.remove(temp_model_path+".index")
            os.remove(temp_model_path+".meta")

    if config["save"] is not None and len(config["save"]) > 0:
        labeler.save(config["save"])

    if config["path_test"] is not None:
        i = 0
        for path_test in config["path_test"].strip().split(":"):
            data_test = read_input_files(path_test)
            results_test = process_sentences(data_test, labeler, is_training=False,
                                             learningrate=0.0, config=config, name="test"+str(i))
            i += 1


if __name__ == "__main__":
    run_experiment_new(sys.argv[1])

