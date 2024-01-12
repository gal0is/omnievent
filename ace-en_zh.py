from OmniEvent.arguments import DataArguments, ModelArguments, TrainingArguments, ArgumentParser
from OmniEvent.input_engineering.seq2seq_processor import type_start, type_end
parser = ArgumentParser((ModelArguments, DataArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_yaml_file(yaml_file="config/all-datasets/ed/s2s/ace-en.yaml")
training_args.output_dir = 'output/ACE2005-EN/ED/seq2seq/t5-base/'
data_args.markers = ["<event>", "</event>", type_start, type_end]

from OmniEvent.backbone.backbone import get_backbone
from OmniEvent.model.model import get_model
backbone, tokenizer, config = get_backbone(model_type=model_args.model_type, 
                       		       model_name_or_path=model_args.model_name_or_path, 
                       		       tokenizer_name=model_args.model_name_or_path, 
                       		       markers=data_args.markers,
                       		       new_tokens=data_args.markers)
model = get_model(model_args, backbone)

from OmniEvent.input_engineering.seq2seq_processor import EDSeq2SeqProcessor
from OmniEvent.evaluation.metric import compute_seq_F1
train_dataset = EDSeq2SeqProcessor(data_args, tokenizer, data_args.train_file)
eval_dataset = EDSeq2SeqProcessor(data_args, tokenizer, data_args.validation_file)
metric_fn = compute_seq_F1

from OmniEvent.trainer_seq2seq import Seq2SeqTrainer
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=metric_fn,
    data_collator=train_dataset.collate_fn,
    tokenizer=tokenizer,
)
trainer.train()

from OmniEvent.evaluation.utils import predict, get_pred_s2s
from OmniEvent.evaluation.convert_format import get_trigger_detection_s2s
logits, labels, metrics, test_dataset = predict(trainer=trainer, tokenizer=tokenizer, data_class=EDSeq2SeqProcessor,
                                                data_args=data_args, data_file=data_args.test_file,
                                                training_args=training_args)
# paradigm-dependent metrics
print("{} test performance before converting: {}".format(test_dataset.config.dataset_name, metrics["test_micro_f1"]))  
preds = get_pred_s2s(logits, tokenizer)
# convert to the unified prediction and evaluate
pred_labels = get_trigger_detection_s2s(preds, labels, data_args.test_file, data_args, None)
