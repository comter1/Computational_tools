from transformers import AutoTokenizer, AutoModelForSequenceClassification

# RoBERTa Sentiment
print("Downloading & saving RoBERTa Twitter Sentiment model...")

sent_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"

sent_tokenizer = AutoTokenizer.from_pretrained(sent_name)
sent_model = AutoModelForSequenceClassification.from_pretrained(sent_name)

sent_tokenizer.save_pretrained("./local_models/roberta_sentiment")
sent_model.save_pretrained("./local_models/roberta_sentiment")


# GoEmotions
print("Downloading & saving GoEmotions model...")

go_name = "SamLowe/roberta-base-go_emotions"

go_tokenizer = AutoTokenizer.from_pretrained(go_name)
go_model = AutoModelForSequenceClassification.from_pretrained(go_name)

go_tokenizer.save_pretrained("./local_models/go_emotions")
go_model.save_pretrained("./local_models/go_emotions")


print("\n All models saved to ./local_models/")
