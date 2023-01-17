if [ ! -d bert-base-uncased ]; then
  mkdir bert-base-uncased
fi
wget https://s3plus.sankuai.com/lqy-data/model/BERT_quant/bert-base-uncased/README.md -P bert-base-uncased
wget https://s3plus.sankuai.com/lqy-data/model/BERT_quant/bert-base-uncased/config.json -P bert-base-uncased
wget https://s3plus.sankuai.com/lqy-data/model/BERT_quant/bert-base-uncased/pytorch_model.bin -P bert-base-uncased
wget https://s3plus.sankuai.com/lqy-data/model/BERT_quant/bert-base-uncased/tokenizer.json -P bert-base-uncased
wget https://s3plus.sankuai.com/lqy-data/model/BERT_quant/bert-base-uncased/tokenizer_config.json -P bert-base-uncased
wget https://s3plus.sankuai.com/lqy-data/model/BERT_quant/bert-base-uncased/vocab.txt -P bert-base-uncased

if [ ! -d squad/SQuAD1.1 ]; then
  mkdir -p squad/SQuAD1.1
fi
wget https://s3plus.sankuai.com/lqy-data/model/BERT_quant/SQuAD1.1/train-v1.1.json -P squad/SQuAD1.1
wget https://s3plus.sankuai.com/lqy-data/model/BERT_quant/SQuAD1.1/dev-v1.1.json -P squad/SQuAD1.1
cp squad.py squad
