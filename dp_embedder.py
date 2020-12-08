from deeppavlov.core.common.file import read_json
from deeppavlov import build_model, configs

bert_config = read_json(configs.embedder.bert_embedder)
bert_config['metadata']['variables']['BERT_PATH'] = '/projappl/project_2002016/gramcor/bert-pretraned/rubert_cased_L-12_H-768_A-12_v2'

m = build_model(bert_config)

texts = ['Скоро рождество!', 'А это значит, что все будет хорошо.']
tokens, token_embs, subtokens, subtoken_embs, sent_max_embs, sent_mean_embs, bert_pooler_outputs = m(texts)
print(token_embs.shape)