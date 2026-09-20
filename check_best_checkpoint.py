from config.global_config import CONFIG
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.finbert_finetune import tokenize_dataset, evaluate_finbert
import pandas as pd

tokenizer = AutoTokenizer.from_pretrained(CONFIG['finbert_base_checkpoint'])
model = AutoModelForSequenceClassification.from_pretrained(CONFIG['checkpoint_dir'])

test_df = pd.read_parquet(CONFIG['processed_dir'] + '/test.parquet')
test_ds = tokenize_dataset(test_df['sentence'].tolist(), tokenizer, test_df['label'].tolist(), CONFIG['max_seq_len'])
metrics = evaluate_finbert(model, test_ds)
print(metrics)
