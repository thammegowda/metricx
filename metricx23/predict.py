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

import dataclasses
import json
import os

import datasets
from pathlib import Path
from metricx23 import models
import torch
import transformers


@dataclasses.dataclass
class Arguments:
  """Prediction command-line arguments."""

  tokenizer: str = dataclasses.field(
      metadata={"help": "The name of the tokenizer"},
  )

  model_name_or_path: str = dataclasses.field(
      metadata={
          "help": (
              "Path to pretrained model or model identifier from"
              " huggingface.co/models"
          )
      },
  )

  max_input_length: int = dataclasses.field(
      metadata={"help": "The maximum allowable input sequence length."},
  )

  batch_size: int = dataclasses.field(
      metadata={"help": "The global prediction batch size."},
  )

  input_file: str = dataclasses.field(metadata={"help": "The input file."})

  output_file: str = dataclasses.field(
      metadata={"help": "The output file with predictions."},
  )

  qe: bool = dataclasses.field(
      metadata={"help": "Indicates the metric is a QE metric."},
      default=False,
  )

  tsv: bool = dataclasses.field(
      metadata={"help": "input_file is a TSV of [source, hypothesis, reference] \
                fields order. When --qe is set. the last column i.e. reference is optional."},
      default=False,
  )


def get_dataset(
    input_file: str, tokenizer, max_input_length: int, device, is_qe: bool,
    is_tsv: bool = False,
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

  Returns:
    The dataset.
  """

  def _make_input(example):
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

  def _make_example_from_tsv(example):
    row = example['text'].split("\t")
    example = {}
    if is_qe:
      assert len(row) >= 2, "TSV file must have at least two columns: [source, candidate] for QE metric"
      # reference is optional at index=2 and ignored
      example["input"] = f"candidate: {row[1]} source: {row[0]}"
    else:
      assert len(row) >= 3, "TSV file must have at least three columns [source, candidate, reference] for MT metric"
      # source is ignored but assumed to exist in the input to make it consistent with the QE metric
      example["input"] = f"candidate: {row[1]} reference: {row[2]}"
    return example

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

  if is_tsv:
    ds = datasets.load_dataset("text", data_files={"test": input_file})
    ds = ds.map(_make_example_from_tsv)
  else:
    ds = datasets.load_dataset("json", data_files={"test": input_file})
    ds = ds.map(_make_input)
  ds = ds.map(_tokenize)
  ds = ds.map(_remove_eos)
  ds.set_format(
      type="torch",
      columns=["input_ids", "attention_mask"],
      device=device,
      output_all_columns=True,
  )
  return ds


def main() -> None:
  parser = transformers.HfArgumentParser(Arguments)
  (args,) = parser.parse_args_into_dataclasses()

  if torch.cuda.is_available():
    device = torch.device("cuda")
    per_device_batch_size = args.batch_size // torch.cuda.device_count()
  else:
    device = torch.device("cpu")
    per_device_batch_size = args.batch_size
  assert per_device_batch_size > 0, "Batch size must be greater than 0."
  tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer)

  model = models.MT5ForRegression.from_pretrained(args.model_name_or_path)
  model.to(device)
  model.eval()

  ds = get_dataset(
      args.input_file,
      tokenizer,
      args.max_input_length,
      device,
      args.qe,
      is_tsv=args.tsv
  )

  training_args = transformers.TrainingArguments(
      output_dir=str(Path(args.output_file).absolute().parent),
      per_device_eval_batch_size=per_device_batch_size,
      dataloader_pin_memory=False,
  )
  trainer = transformers.Trainer(
      model=model,
      args=training_args,
  )
  predictions, _, _ = trainer.predict(test_dataset=ds["test"])

  dirname = os.path.dirname(args.output_file)
  if dirname:
    os.makedirs(dirname, exist_ok=True)

  with open(args.output_file, "w") as out:
    for pred, example in zip(predictions, ds["test"]):
      example["prediction"] = float(pred)
      del example["input"]
      del example["input_ids"]
      del example["attention_mask"]
      out.write(json.dumps(example) + "\n")


if __name__ == "__main__":
  main()
