# coding=utf-8
# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Runs inference with a MetricX model."""


import json
import argparse
import sys

import datasets
from pathlib import Path
from typing import Iterator
from metricx import models
import torch
import transformers
from functools import partial


DEF_TOKENIZER = "google/mt5-xl"
DEF_WIDTH = 5
DEF_BATCH_SIZE = 1
MAX_LEN_METRICX23 = 1024
MAX_LEN_METRICX24 = 1536
TSV_FIELDS = ["source", "hypothesis", "reference"]

# print this in help for easy lookup
KNOWN_MODELS_TXT = '''
google/metricx-24-hybrid-large-v2p6
google/metricx-24-hybrid-xl-v2p6
google/metricx-24-hybrid-xxl-v2p6
google/metricx-24-hybrid-large-v2p6-bfloat16
google/metricx-24-hybrid-xl-v2p6-bfloat16
google/metricx-24-hybrid-xxl-v2p6-bfloat16
google/metricx-23-qe-large-v2p0
google/metricx-23-qe-xl-v2p0
google/metricx-23-qe-xxl-v2p0
google/metricx-23-large-v2p0
google/metricx-23-xl-v2p0
google/metricx-23-xxl-v2p0
'''
KNOWN_MODELS = KNOWN_MODELS_TXT.strip().split()

def make_input_23(example, is_qe):
    if is_qe:
      example["input"] = (
          "candidate: "
          + example["hypothesis"]
          + " source: "
          + example["source"]
      )
    else:
      example["input"] = (
          "candidate: "
          + example["hypothesis"]
          + " reference: "
          + example["reference"]
      )
    return example

def make_input_24(example, is_qe):
  if is_qe:
    example["input"] = (
        "source: "
        + example["source"]
        + " candidate: "
        + example["hypothesis"]
    )
  else:
    example["input"] = (
        "source: "
        + example["source"]
        + " candidate: "
        + example["hypothesis"]
        + " reference: "
        + example["reference"]
    )
  return example

def load_data_file(input_file: str, is_tsv: bool) -> Iterator[dict]:
  try:
    if not input_file or input_file == "-":
      inp = sys.stdin
    else:
      inp = open(input_file, "r", encoding="utf-8")

    for line in inp:
      if is_tsv:
        parts = line.split("\t")
        example = {k:v for k, v in zip(TSV_FIELDS, parts)}
      else:
        example = json.loads(line)
      yield example
  finally:
    if inp != sys.stdin:
      inp.close()


def get_dataset(
    input_file: str, tokenizer, max_input_length: int, device, is_qe: bool,
    model_id:str, is_tsv: bool = False,
):
  """Gets the test dataset for prediction.

  If `is_qe` is true, the input data must have "hypothesis" and "source" fields.
  If it is false, there must be "hypothesis" and "reference" fields.

  Args:
    input_file: The path to the jsonl input file.
    tokenizer: The tokenizer to use.
    max_input_length: The maximum input sequence length.
    device: The ID of the device to put the PyTorch tensors on.
    is_qe: Indicates whether the metric is a QE metric or not.
    model_id: The model identifier.
    is_tsv: Indicates whether the input file is a TSV file or not.

  Returns:
    The dataset.
  """

  def _tokenize(example):
    return tokenizer(
        example["input"],
        max_length=max_input_length,
        truncation=True,
        padding=False,
    )

  def _remove_eos(example):
    example["input_ids"] = example["input_ids"][:-1]
    example["attention_mask"] = example["attention_mask"][:-1]
    return example

  def _map_tsv_to_dict(example):
    parts = example.pop("text").split("\t")
    for k, v in zip(TSV_FIELDS, parts):
      example[k] = v
    return example

  if input_file == "-":
    ds = datasets.Dataset.from_generator(
        load_data_file, streaming=True,
        gen_kwargs={"input_file": input_file, "is_tsv": is_tsv})
  else:
    ds = datasets.load_dataset(is_tsv and "text" or "json", data_files={"test": input_file})
    ds = ds["test"]
    ds._output_all_columns = True
    if is_tsv:
      ds = ds.map(_map_tsv_to_dict)

  make_input = make_input_23
  if "metricx-24-" in model_id.lower():
    make_input = make_input_24

  ds = ds.map(make_input, fn_kwargs={"is_qe": is_qe})
  ds = ds.map(_tokenize)
  ds = ds.map(_remove_eos)
  ds = ds.with_format("torch")
  return ds

def data_collator(items, pad_id, device) -> dict:
  max_len = max(len(it["input_ids"]) for it in items)
  input_ids = torch.full((len(items), max_len), pad_id, dtype=torch.long, device=device)
  attention_mask = torch.zeros((len(items), max_len), dtype=torch.uint8, device=device)
  for idx, it in enumerate(items):
    input_ids[idx, :len(it["input_ids"])] = it["input_ids"]
    attention_mask[idx, :len(it["attention_mask"])] = it["attention_mask"]
  return dict(input_ids=input_ids, attention_mask=attention_mask)


def parse_args():
  # originally written using transformers.HfArgumentParser and a dataclass
  # but we are using argparse as it is a matured lib and more common in the Python community
  # and has more features relevant for CLI parsing (e.g. short forms)
  # and more *nix frindly e.g. dashes instead of underscores, STDIN/STDOUT as default files
  epilog = "Knwon models are:\n" + KNOWN_MODELS_TXT.strip() + "\n\nThe above list maybe incomplete. See https://huggingface.co/models?search=metricx for the latest list."

  parser = argparse.ArgumentParser(
    description="Runs inference with a MetricX model.",
    formatter_class=argparse.RawDescriptionHelpFormatter, epilog=epilog)
  parser.add_argument('-t', "--tokenizer", type=str, default=DEF_TOKENIZER, metavar='MODEL_ID',
                      help="The name of the tokenizer.")
  parser.add_argument('-m', "--model_name_or_path",  '--model', metavar='MODEL', type=str, required=True,
                      help="Path to pretrained model or model identifier from huggingface.co/models")
  parser.add_argument( '-x', "--max_input_length",
                       metavar='INT', type=int, default=-1,
                      help=f"The maximum allowable input sequence length, default=-1 => infer: e.g. {MAX_LEN_METRICX23} for metricx23 and {MAX_LEN_METRICX24} for metricx24.")
  parser.add_argument('-b', "--batch_size", type=int, default=DEF_BATCH_SIZE,
                      metavar='INT', help="The global prediction batch size.")
  parser.add_argument('-i', "--input_file", type=str, default='-',
                      metavar='FILE', help="The input file.")
  parser.add_argument('-o', "--output_file", type=str, default='-',
                      metavar='FILE', help="The output file with predictions .")
  parser.add_argument('-qe', "--qe", action="store_true", default=False,
                      help="Indicates the metric is a QE metric.")
  parser.add_argument('-tsv', "--tsv", action="store_true", default=False,
                      help="Input_file is a TSV of [source, hypothesis, reference] fields order. \
                        When --qe is set. the last column i.e. reference is optional. Also, produces TSV output.")
  parser.add_argument('--debug', action="store_true", default=False,
                      help="Print debug information.")
  parser.add_argument('-w', "--width", type=int, default=5,
                      help="The width score i.e. number of decimal points.")
  args = parser.parse_args()

  if args.max_input_length == -1:
    if "metricx-24-" in args.model_name_or_path.lower():
      args.max_input_length = MAX_LEN_METRICX24
    else:
      args.max_input_length = MAX_LEN_METRICX23
  return args

def main() -> None:
  args = parse_args()
  output_dir = str(Path(args.output_file).absolute().parent)

  if torch.cuda.is_available():
    device = torch.device("cuda")
    per_device_batch_size = max(1, args.batch_size // torch.cuda.device_count())
  else:
    device = torch.device("cpu")
    per_device_batch_size = args.batch_size
  assert per_device_batch_size > 0, "Batch size must be greater than 0."
  tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer)

  ds = get_dataset(
      args.input_file,
      tokenizer,
      args.max_input_length,
      device,
      args.qe,
      model_id=args.model_name_or_path,
      is_tsv=args.tsv
  )

  model = models.MT5ForRegression.from_pretrained(args.model_name_or_path)
  model.to(device)
  model.eval()

  training_args = transformers.TrainingArguments(
      output_dir=output_dir,
      per_device_eval_batch_size=per_device_batch_size,
      dataloader_pin_memory=False,
  )
  trainer = transformers.Trainer(
      model=model,
      args=training_args,
      data_collator=partial(data_collator, pad_id=tokenizer.pad_token_id, device=device),
  )
  predictions, _, _ = trainer.predict(test_dataset=ds)

  try:
    if args.output_file == "-":
      out = sys.stdout
    else:
      out = open(args.output_file, "w", encoding="utf-8")

    if args.input_file == "-":
      # streaming mode: not possible to read input again
      for pred in predictions:
        out.write(f'{pred:.{args.width}f}\n')
    else:
      # inputs are available in the dataset
      for pred, example in zip(predictions, ds):
        example["prediction"] = float(pred)
        if args.tsv:
          line = f'{example["prediction"]:.{args.width}f}'
          if args.debug:
            line += f'\t{example["input"]}'
          out.write(line + "\n")
        else:
          del example["input"]
          del example["input_ids"]
          del example["attention_mask"]
          out.write(json.dumps(example) + "\n")
  finally:
    if out != sys.stdout:
      out.close()


if __name__ == "__main__":
  main()
