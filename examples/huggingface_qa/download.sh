if [ ! -d bert-base-uncased ]; then
  mkdir bert-base-uncased
fi
wget https://s3plus.sankuai.com/lqy-data/model/BERT_quant/bert-base-uncased/README.md -P bert-base-uncased
wget https://s3plus.sankuai.com/lqy-data/model/BERT_quant/bert-base-uncased/config.json -P bert-base-uncased
wget https://s3plus.sankuai.com/lqy-data/model/BERT_quant/bert-base-uncased/pytorch_model.bin -P bert-base-uncased
wget https://s3plus.sankuai.com/lqy-data/model/BERT_quant/bert-base-uncased/tokenizer.json -P bert-base-uncased
wget https://s3plus.sankuai.com/lqy-data/model/BERT_quant/bert-base-uncased/tokenizer_config.json -P bert-base-uncased
wget https://s3plus.sankuai.com/lqy-data/model/BERT_quant/bert-base-uncased/vocab.txt -P bert-base-uncased
