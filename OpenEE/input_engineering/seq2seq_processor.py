import re
import json
import logging
from typing import List, Union, Tuple, Optional

from tqdm import tqdm
from collections import defaultdict
from .input_utils import get_words, get_plain_label
from .base_processor import (
    EDInputExample,
    EDDataProcessor,
    EDInputFeatures,
    EAEDataProcessor,
    EAEInputExample,
    EAEInputFeatures,
)

type_start = "<"
type_end = ">"
split_word = ":"

logger = logging.getLogger(__name__)


def extract_argument(raw_text: str,
                     instance_id: Union[int, str],
                     event_type: str,
                     template=re.compile(f"{type_start}|{type_end}")) -> List[Tuple]:
    """Extracts the arguments from the raw text.

    Args:
        raw_text (`str`):
            A string indicating the input raw text.
        instance_id (`Union[int, str]`):
            The id of the input example.
        event_type (`str`):
            A string indicating the type of the event.
        template (`optional`, defaults to `re.compile(f"[{type_start}{type_end}]")`):
            The template of the event argument extraction.
    """
    arguments = []
    for span in template.split(raw_text):
        if span.strip() == "":
            continue
        words = span.strip().split(split_word)
        if len(words) != 2:
            continue
        role = words[0].strip().replace(" ", "")
        value = words[1].strip().replace(" ", "")
        if role != "" and value != "":
            arguments.append((instance_id, event_type, role, value))
    arguments = list(set(arguments))
    return arguments


class EDSeq2SeqProcessor(EDDataProcessor):
    """Data processor for Sequence-to-Sequence (Seq2Seq) for event detection.

    Data processor for Sequence-to-Sequence (Seq2Seq) for event detection. The class is inherited from the
    `EDDataProcessor` class, in which the undefined functions, including `read_examples()` and
    `convert_examples_to_features()` are  implemented; the rest of the attributes and functions are multiplexed from the
    `EDDataProcessor` class.
    """

    def __init__(self,
                 config,
                 tokenizer,
                 input_file: str) -> None:
        """Constructs a `EDSeq2SeqProcessor`."""
        super().__init__(config, tokenizer)
        self.read_examples(input_file)
        self.convert_examples_to_features()
    
    def read_examples(self,
                      input_file: str) -> None:
        """Obtains a collection of `EDInputExample`s for the dataset."""
        self.examples = []
        with open(input_file, "r", encoding="utf-8") as f:
            for line in tqdm(f.readlines(), desc="Reading from %s" % input_file):
                item = json.loads(line.strip())
<<<<<<< HEAD
                if self.config.language == "English":
                    words = item["text"].split()
                elif self.config.language == "Chinese":
                    words = list(item["text"])
                else:
                    raise NotImplementedError
=======
                if "source" in item:
                    kwargs = {"source": [item["source"]]}
                    if item["source"] in ["<duee>", "<fewfc>", "<leven>"]:
                        self.config.language = "Chinese"
                    else:
                        self.config.language = "English"
                else:
                    kwargs = {"source": []}

                words = get_words(text=item["text"], language=self.config.language)
>>>>>>> dev
                # training and valid set
                if "events" in item:
                    labels = []
                    for event in item["events"]:
                        type = get_plain_label(event["type"])
                        for trigger in event["triggers"]:
                            labels.append(f"{type_start} {type}{split_word} {trigger['trigger_word']} {type_end}")
                    if len(labels) != 0:
                        labels = "".join(labels)
                    else:  # no arguments for the trigger
                        labels = ""

                    example = EDInputExample(
                        example_id=item["id"],
                        text=words,
<<<<<<< HEAD
                        labels=labels
=======
                        labels=labels,
                        **kwargs,
>>>>>>> dev
                    )
                    self.examples.append(example)
                else:
                    example = EDInputExample(example_id=idx, text=words, labels="", **kwargs)
                    self.examples.append(example)

    def convert_examples_to_features(self) -> None:
        """Converts the `EDInputExample`s into `EDInputFeatures`s."""
        self.input_features = []
        for example in tqdm(self.examples, desc="Processing features for Seq2Seq"):
            # context 
            input_context = self.tokenizer(example.text,
                                           truncation=True,
                                           padding="max_length",
                                           max_length=self.config.max_seq_length,
                                           is_split_into_words=True)
            # output labels
            label_outputs = self.tokenizer(example.labels.split(),
                                           truncation=True,
                                           padding="max_length",
                                           max_length=self.config.max_out_length,
                                           is_split_into_words=True)
            # set -100 to unused token 
            for i, flag in enumerate(label_outputs["attention_mask"]):
                if flag == 0:
                    label_outputs["input_ids"][i] = -100
            features = EDInputFeatures(
                example_id=example.example_id,
                input_ids=input_context["input_ids"],
                attention_mask=input_context["attention_mask"],
                labels=label_outputs["input_ids"],
            )
            self.input_features.append(features)


class EAESeq2SeqProcessor(EAEDataProcessor):
    """Data processor for sequence to sequence for event argument extraction.

    Data processor for token classification for event argument extraction. The class is inherited from the
    `EAEDataProcessor` class, in which the undefined functions, including `read_examples()` and
    `convert_examples_to_features()` are  implemented; a new function entitled `insert_marker()` is defined, and
    the rest of the attributes and functions are multiplexed from the `EAEDataProcessor` class.
    """

    def __init__(self,
                 config,
                 tokenizer: str,
                 input_file: str,
                 pred_file: str,
                 is_training: Optional[bool] = False) -> None:
        """Constructs a `EAESeq2SeqProcessor`."""
        super().__init__(config, tokenizer, pred_file, is_training)
        self.read_examples(input_file)
        self.convert_examples_to_features()
    
    def read_examples(self,
                      input_file: str) -> None:
        """Obtains a collection of `EAEInputExample`s for the dataset."""
        self.examples = []
        self.data_for_evaluation["golden_arguments"] = []
        self.data_for_evaluation["roles"] = []
        language = self.config.language
        trigger_idx = 0
        with open(input_file, "r", encoding="utf-8") as f:
            for line in tqdm(f.readlines(), desc="Reading from %s" % input_file):
                item = json.loads(line.strip())
<<<<<<< HEAD
                prefix = []
                if self.config.language == "English":
                    words = prefix + item["text"].split()
                    whitespace = " "
                elif self.config.language == "Chinese":
                    words = prefix + list(item["text"])
                    whitespace = ""
                else:
                    raise NotImplementedError
=======
                if "source" in item:
                    kwargs = {"source": [item["source"]]}
                    if item["source"] in ["<duee>", "<fewfc>", "<leven>"]:
                        self.config.language = "Chinese"
                    else:
                        self.config.language = "English"
                else:
                    kwargs = {"source": []}

                text = item["text"]
                words = get_words(text=text, language=language)
                whitespace = " " if language == "English" else ""

>>>>>>> dev
                if "events" in item:
                    for event in item["events"]:
                        for trigger in event["triggers"]:
                            if self.is_training or self.config.golden_trigger or self.event_preds is None:
                                pred_event_type = event["type"]
                            else:
                                pred_event_type = self.event_preds[trigger_idx]
                            trigger_idx += 1

                            # Evaluation mode for EAE
                            # If the predicted event type is NA, We don't consider the trigger
                            if self.config.eae_eval_mode in ["default", "loose"] and pred_event_type == "NA":
                                continue

                            labels = []
                            arguments_per_trigger = defaultdict(list)
                            for argument in trigger["arguments"]:
                                role = argument["role"]
                                for mention in argument["mentions"]:
                                    arguments_per_trigger[argument["role"]].append(mention["mention"])
                                    labels.append(
                                        f"{type_start}{whitespace}{role}{split_word}{whitespace}{mention['mention']}{whitespace}{type_end}")
                            if len(labels) != 0:
                                labels = "".join(labels)
                            else:  # no arguments for the trigger
                                labels = ""
                            self.data_for_evaluation["golden_arguments"].append(dict(arguments_per_trigger))
                            example = EAEInputExample(
<<<<<<< HEAD
                                example_id=trigger["id"],
=======
                                example_id=trigger_idx - 1,
>>>>>>> dev
                                text=words,
                                pred_type=pred_event_type,
                                true_type=event["type"],
                                trigger_left=trigger["position"][0],
                                trigger_right=trigger["position"][1],
<<<<<<< HEAD
                                labels=labels
=======
                                labels=labels,
                                **kwargs,
>>>>>>> dev
                            )
                            self.examples.append(example)
                    # negative triggers 
                    for neg_trigger in item["negative_triggers"]:
                        if self.is_training or self.config.golden_trigger or self.event_preds is None:
                            pred_event_type = event["type"]
                        else:
                            pred_event_type = self.event_preds[trigger_idx]
                        trigger_idx += 1

                        if self.config.eae_eval_mode == "loose":
                            continue
                        elif self.config.eae_eval_mode in ["default", "strict"]:
                            if pred_event_type != "NA":
                                labels = ""
                                arguments_per_trigger = {}
                                self.data_for_evaluation["golden_arguments"].append(dict(arguments_per_trigger))
                                example = EAEInputExample(
                                    example_id=trigger_idx - 1,
                                    text=words,
                                    pred_type=pred_event_type,
                                    true_type="NA",
                                    trigger_left=neg_trigger["position"][0],
                                    trigger_right=neg_trigger["position"][1],
<<<<<<< HEAD
                                    labels=labels
=======
                                    labels=labels,
                                    **kwargs,
>>>>>>> dev
                                )
                                self.examples.append(example)
                        else:
                            raise ValueError("Invaild eac_eval_mode: %s" % self.config.eae_eval_mode)
                else:
                    for candi in item["candidates"]:
                        pred_event_type = self.event_preds[trigger_idx]
                        trigger_idx += 1
                        if pred_event_type != "NA":
                            labels = ""
                            # labels = f"{type_start}{type_end}"
                            arguments_per_trigger = {}
                            self.data_for_evaluation["golden_arguments"].append(dict(arguments_per_trigger))
                            example = EAEInputExample(
<<<<<<< HEAD
                                example_id=item["id"],
=======
                                example_id=trigger_idx - 1,
>>>>>>> dev
                                text=words,
                                pred_type=pred_event_type,
                                true_type="NA",  # true type not given, set to NA.
                                trigger_left=candi["position"][0],
                                trigger_right=candi["position"][1],
                                labels=labels,
<<<<<<< HEAD
=======
                                **kwargs,
>>>>>>> dev
                            )
                            self.examples.append(example)
            if self.event_preds is not None:
                assert trigger_idx == len(self.event_preds)


    def insert_marker(self,
                      tokens: List[str],
                      trigger_pos: List[int],
                      markers: List,
                      whitespace: Optional[bool] = True) -> List[str]:
        """Adds a marker at the start and end position of event triggers and argument mentions."""
        space = " " if whitespace else ""
        marked_words = []
        char_pos = 0
        for i, token in enumerate(tokens):
            if char_pos == trigger_pos[0]:
                marked_words.append(markers[0])
            char_pos += len(token) + len(space)
            marked_words.append(token)
            if char_pos == trigger_pos[1] + len(space):
                marked_words.append(markers[1])
        return marked_words

    def convert_examples_to_features(self) -> None:
        """Converts the `EAEInputExample`s into `EAEInputFeatures`s."""
        self.input_features = []
        whitespace = True if self.config.language == "English" else False
        for example in tqdm(self.examples, desc="Processing features for Seq2Seq"):
            # context 
<<<<<<< HEAD
            words = self.insert_marker(example.text, 
                                      [example.trigger_left, example.trigger_right],
                                      self.config.markers,
                                      whitespace)
            input_context = self.tokenizer(words,
=======
            words = self.insert_marker(example.text,
                                       [example.trigger_left, example.trigger_right],
                                       self.config.markers,
                                       whitespace)
            input_context = self.tokenizer(example.kwargs["source"] + words,
>>>>>>> dev
                                           truncation=True,
                                           padding="max_length",
                                           max_length=self.config.max_seq_length,
                                           is_split_into_words=True)
            # output labels
            label_outputs = self.tokenizer(example.labels.split(),
                                           padding="max_length",
                                           truncation=True,
                                           max_length=self.config.max_out_length,
                                           is_split_into_words=True)
<<<<<<< HEAD
            # set -100 to unused token 
=======
            # set -100 to unused token
>>>>>>> dev
            for i, flag in enumerate(label_outputs["attention_mask"]):
                if flag == 0:
                    label_outputs["input_ids"][i] = -100
            features = EAEInputFeatures(
                example_id=example.example_id,
                input_ids=input_context["input_ids"],
                attention_mask=input_context["attention_mask"],
                labels=label_outputs["input_ids"],
            )
            self.input_features.append(features)