from transformers import RobertaTokenizer, RobertaForMaskedLM, LineByLineTextDataset, DataCollatorForLanguageModeling,  Trainer, TrainingArguments
import os

#Load the necessary tokenizer and the model we want to fine tune
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
lm_model = RobertaForMaskedLM.from_pretrained('roberta-base')

#Load the dataset stored in the line-by-line format.
dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="/home/jvladika/lex-sub/data/SMS_data.txt",
    block_size=128,
)

#Load the data collator that randomly masks some percentage of words in the dataset
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

#Set all the necessary training arguments 
training_args = TrainingArguments(
    output_dir="./roberta-retrained",
    overwrite_output_dir=True,
    num_train_epochs=25,
    per_device_train_batch_size=4,
    save_steps=500,
    save_total_limit=2,
    seed=1
)

#Load the trainer
trainer = Trainer(
    model=lm_model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset
)

#Set the GPU settings
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#Traub the model and save it.
trainer.train()
trainer.save_model("./roberta-retrained")