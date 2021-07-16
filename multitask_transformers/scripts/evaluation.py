from transformers import AutoConfig, EvalPrediction, AutoTokenizer,BertTokenizer
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from transformers import DistilBertConfig
from torch.utils.data import Dataset
from dataclasses import dataclass, field
from typing import Dict, Optional
from multitask_transformers.scripts.utils import InputFeaturesMultitask, f1, DataTrainingArguments, GamesarArguments
from multitask_transformers.scripts.modeling_auto import AutoModelForMultitaskSequenceClassification,\
                            AutoModelForADTaskClassification, AutoModelForSNSTaskClassification
from multitask_transformers.scripts.evaluator import Evaluator
import numpy as np
from multitask_transformers.scripts.utils import store_preds

import torch
import os 
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

label_dict = {'sarc': 1, 'notsarc': 0, 'agree': 0, 'disagree': 1, 'neutral': 0}

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


class SarcArgDataset(Dataset):
    def __init__(self, data, tokenizer,use_neutral):
        self.data = data
        self.tokenizer = tokenizer

        batch_encoding = self.tokenizer.batch_encode_plus(
        [(example.split('\t')[0], example.split('\t')[1]) for example in self.data], add_special_tokens=True, max_length=512, pad_to_max_length=True,
        )

        self.features = []
        for i in range(len(self.data)):
            inputs = {k: batch_encoding[k][i] for k in batch_encoding}
            arglab, sarclab = self.data[i].split('\t')[2:4]   
            if not sarclab or not arglab:
                continue
            if use_neutral:
                if arglab.strip('\t').strip('\n') == 'neutral':  # filter out Neutral samples 
                    continue
            feature = InputFeaturesMultitask(  ## token_ids ; attention_mask; token_type_ids; labels
                **inputs,  
                labels_t1=label_dict[sarclab.strip('\t').strip('\n')],
                labels_t2=label_dict[arglab.strip('\t').strip('\n')])
            self.features.append(feature)
    

        for i, example in enumerate(self.data[:3]):
            logger.info("*** Example ***")
            logger.info("features: %s" % self.features[i])

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]

class NonSarcArgDataset(Dataset):
    def __init__(self, data, tokenizer,use_neutral):
        self.data = data
        self.tokenizer = tokenizer

        batch_encoding = self.tokenizer.batch_encode_plus(
        [(example.split('\t')[0], example.split('\t')[1]) for example in self.data], add_special_tokens=True, max_length=512, pad_to_max_length=True,
        )

        self.features = []
        for i in range(len(self.data)):
            inputs = {k: batch_encoding[k][i] for k in batch_encoding}
            arglab, sarclab = self.data[i].split('\t')[2:4]  
            if not sarclab or not arglab:
                continue
            if use_neutral:
                if arglab.strip('\t').strip('\n') == 'neutral' or sarclab.strip('\t').strip('\n') =='sarc':  # filter out Neutral samples 
                    continue
            else:
                if sarclab.strip('\t').strip('\n') =='sarc':  # filter out Neutral samples 
                    continue
            feature = InputFeaturesMultitask(  ## token_ids ; attention_mask; token_type_ids; labels
                **inputs,  
                labels_t1=label_dict[sarclab.strip('\t').strip('\n')],
                labels_t2=label_dict[arglab.strip('\t').strip('\n')])
            self.features.append(feature)
    

        for i, example in enumerate(self.data[:3]):
            logger.info("*** Example ***")
            logger.info("features: %s" % self.features[i])

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]

class SarcOnlyArgDataset(Dataset):
    def __init__(self, data, tokenizer,use_neutral):
        self.data = data
        self.tokenizer = tokenizer

        batch_encoding = self.tokenizer.batch_encode_plus(
        [(example.split('\t')[0], example.split('\t')[1]) for example in self.data], add_special_tokens=True, max_length=512, pad_to_max_length=True,
        )

        self.features = []
        for i in range(len(self.data)):
            inputs = {k: batch_encoding[k][i] for k in batch_encoding}
            arglab, sarclab = self.data[i].split('\t')[2:4]  
            if not sarclab or not arglab:
                continue
            if use_neutral:
                if arglab.strip('\t').strip('\n') == 'neutral' or sarclab.strip('\t').strip('\n') =='notsarc':  # filter out Neutral samples 
                    continue
            else: 
                if sarclab.strip('\t').strip('\n') =='notsarc':  # filter out Neutral samples 
                    continue
            feature = InputFeaturesMultitask(  ## token_ids ; attention_mask; token_type_ids; labels
                **inputs, 
                labels_t1=label_dict[sarclab.strip('\t').strip('\n')],
                labels_t2=label_dict[arglab.strip('\t').strip('\n')])
            self.features.append(feature)
    

        for i, example in enumerate(self.data[:3]):
            logger.info("*** Example ***")
            logger.info("features: %s" % self.features[i])

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]

def _use_cuda():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True


def _load_data(dargs, evaluate=False):
    dtype = dargs.eval_file if evaluate else dargs.train_file
    with open(os.path.join(dargs.data_dir, dtype)) as f:
        data = f.readlines()
    return data


def main():

    #_use_cuda()
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments,GamesarArguments))
    model_args, data_args, training_args, gamesar_args = parser.parse_args_into_dataclasses()
    # import pdb;pdb.set_trace()

    ADTASK = 'ADtask'
    MULTITASK = 'Multitask'
    SNSTASK = 'SNStask'

    SARC = "sarc"
    NONSARC ='notsarc'
    BOTH = 'both'

    task = gamesar_args.task
    use_data = gamesar_args.eval_on
    sep_pt_ct = False
    

    logger.info("doing %s " % task)

    # Set seed
    set_seed(training_args.seed)

    config = AutoConfig.from_pretrained(
    model_args.config_name if model_args.config_name else model_args.model_name_or_path,
    num_labels=2,
    )
    model = AutoModelForMultitaskSequenceClassification.from_pretrained(  
            model_args.model_name_or_path,
            config=config,
        )
    # tokenizer = AutoTokenizer.from_pretrained(
    #     model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
    # )
    tokenizer = BertTokenizer.from_pretrained(model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path)

    # print(model.state_dict())
    # Fetch Datasets
    if use_data == BOTH :
        eval_dataset = SarcArgDataset(_load_data(data_args, evaluate=True), tokenizer, gamesar_args.use_neutral) if training_args.do_eval else None
    elif use_data ==NONSARC:
        eval_dataset = NonSarcArgDataset(_load_data(data_args, evaluate=True), tokenizer,gamesar_args.use_neutral) if training_args.do_eval else None
    elif use_data ==SARC:
        eval_dataset = SarcOnlyArgDataset(_load_data(data_args, evaluate=True), tokenizer,gamesar_args.use_neutral) if training_args.do_eval else None

    def compute_metrics(p: EvalPrediction) -> Dict:
        preds = np.argmax(p.predictions, axis=1)
        return f1(preds, p.label_ids)

    evaluator = Evaluator(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        task = task
    )
    
    # Evaluation
    results = {}
    if training_args.do_eval and training_args.local_rank in [-1, 0]:
        logger.info("*** Evaluate ***")

        eval_datasets = [eval_dataset]
        for eval_dataset in eval_datasets:
            result_set = evaluator.evaluate(eval_dataset=eval_dataset) 
            if task and (task ==ADTASK or task ==SNSTASK):
                result = result_set.metrics
            else :
                result = result_set[0].metrics

            output_eval_file = os.path.join(
                training_args.output_dir, f"eval_results_.txt"
            )
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in result.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))

            results.update(result)

            if not (task==ADTASK or task ==SNSTASK):
                preds_t1, label_ids_t1 = result_set[0].predictions, result_set[0].label_ids
                preds_t2, label_ids_t2 = result_set[1].predictions, result_set[1].label_ids
                preds_t1, labels_t1 = store_preds(EvalPrediction(predictions=preds_t1, label_ids=label_ids_t1))
                preds_t2, labels_t2 = store_preds(EvalPrediction(predictions=preds_t2, label_ids=label_ids_t2))
            elif task ==ADTASK:
                preds_t2, label_ids_t2 = result_set.predictions, result_set.label_ids
                preds_t2, labels_t2 = store_preds(EvalPrediction(predictions=preds_t2, label_ids=label_ids_t2))
            elif task ==SNSTASK:
                preds_t1, label_ids_t1 = result_set.predictions, result_set.label_ids
                preds_t1, labels_t1 = store_preds(EvalPrediction(predictions=preds_t1, label_ids=label_ids_t1))

            data = _load_data(data_args, evaluate=True)
            context, reply = [], []
            for example in data:
                ctx, rpl,arglab,sarclab = example.split('\t')[0:4]
                if arglab.strip('\t').strip('\n') == 'neutral':
                    continue
                else :
                    if use_data == NONSARC:
                        if sarclab.strip('\t').strip('\n') =='sarc':  
                            continue
                    elif use_data == SARC:
                        if sarclab.strip('\t').strip('\n') =='notsarc':
                            continue
                    else : pass
                context.append(ctx)
                reply.append(rpl)

            output_score_file_t1 = os.path.join(
                training_args.output_dir, f"eval_preds_t1.txt"
            )

            output_score_file_t2 = os.path.join(
                training_args.output_dir, f"eval_preds_t2.txt"
            )

            if not task == ADTASK:
                with open(output_score_file_t1, "w") as writer:
                    for i in range(len(context)):
                        writer.write("%s\t%s\t%s\t%s\n" % (context[i], reply[i], labels_t1[i], preds_t1[i]))

            if not task == SNSTASK:
                with open(output_score_file_t2, "w") as writer:
                    for i in range(len(context)):
                        writer.write("%s\t%s\t%s\t%s\n" % (context[i], reply[i], labels_t2[i], preds_t2[i]))

    return results


if __name__ == "__main__":
    main()
