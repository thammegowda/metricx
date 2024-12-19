# MetricX

*This is not an officially supported Google product.*

This repository contains the code for running inference on MetricX-23 and
MetricX-24 models, a family of models for automatic evaluation of translations
that were proposed in the WMT Metrics Shared Task submissions
[MetricX-23: The Google Submission to the WMT 2023 Metrics Shared Task](https://aclanthology.org/2023.wmt-1.63/)
and
[MetricX-24: The Google Submission to the WMT 2024 Metrics Shared Task](https://aclanthology.org/2024.wmt-1.35/).
The models were trained in [T5X](https://github.com/google-research/t5x) and
then converted for use in PyTorch.


## Available Models

There are 3 MetricX-24 models and 6 MetricX-23 models available on Hugging Face
that vary in the number of parameters. The MetricX-23 models further vary in
whether or not the model is reference-based or reference-free (also known as
quality estimation, or QE), whereas the MetricX-24 models are all hybrid models
that can do both reference-based and QE inference. Finally, we also offer
lower-precision (bfloat16) MetricX-24 model variants with a 50% lower memory
footprint.

**MetricX-24:**

* [MetricX-24-Hybrid-XXL](https://huggingface.co/google/metricx-24-hybrid-xxl-v2p6)
* [MetricX-24-Hybrid-XL](https://huggingface.co/google/metricx-24-hybrid-xl-v2p6)
* [MetricX-24-Hybrid-Large](https://huggingface.co/google/metricx-24-hybrid-large-v2p6)

**MetricX-24 (bfloat16):**

* [MetricX-24-Hybrid-XXL](https://huggingface.co/google/metricx-24-hybrid-xxl-v2p6-bfloat16)
* [MetricX-24-Hybrid-XL](https://huggingface.co/google/metricx-24-hybrid-xl-v2p6-bfloat16)
* [MetricX-24-Hybrid-Large](https://huggingface.co/google/metricx-24-hybrid-large-v2p6-bfloat16)

**MetricX-23:**

* [MetricX-23-XXL](https://huggingface.co/google/metricx-23-xxl-v2p0)
* [MetricX-23-XL](https://huggingface.co/google/metricx-23-xl-v2p0)
* [MetricX-23-Large](https://huggingface.co/google/metricx-23-large-v2p0)
* [MetricX-23-QE-XXL](https://huggingface.co/google/metricx-23-qe-xxl-v2p0)
* [MetricX-23-QE-XL](https://huggingface.co/google/metricx-23-qe-xl-v2p0)
* [MetricX-23-QE-Large](https://huggingface.co/google/metricx-23-qe-large-v2p0)

We recommend using the XXL model versions for the best agreement with human
judgments of translation quality, the Large versions for best speed, and the
XL for an intermediate use case.


## Changes to the WMT'24 Submission

The MetricX-24 models available here are most similar to the primary submission
to the WMT'24 Metrics Shared Task. They are initialized with
[mT5](https://aclanthology.org/2021.naacl-main.41/)
then fine-tuned on a combination of direct assessment and MQM data from
WMT'15-'22. However, we made a couple of small changes that make these models
different from the WMT'24 submissions.

First, the metric scores get automatically clipped at 0 and 25, to ensure they
are strictly in the [0, 25] range, as due to the nature of regression models,
the scores could otherwise sometimes fall outside the range. Note that this
clipping is also present in the MetricX-23 models in this repository, but it
wasn't in any of the official submissions to the WMT Metrics shared tasks.

Second, we included one additional type of synthetic training examples that
weren't ready in time for the official submission. These are examples of perfect
translations of multi-sentence segments, generated from the MQM data from
WMT'20-'22. The purpose of this category of synthetic data is to reduce the
model's bias against longer translations when the source segment and/or
reference are also long.


## Changes to the WMT'23 Submission

Similarly to the MetricX-24 models in this repository, we made some changes to
the MetricX-23 models as well, which make these models different from the WMT'23
submissions.

First, the models are trained to regress the actual MQM score rather than a
normalized score between 0 and 1. That means the output from the MetricX-23
models is a score in the range [0, 25] where lower is better (i.e., it predicts
an error score).

Second, these models were trained with a larger variety of synthetic data that
makes them more robust to translation edge cases like over- and
undertranslation, described in more detail in the following section. Note that
an updated version of the synthetic data was later used for training MetricX-24
and is described in detail in the
[MetricX-24 paper](https://aclanthology.org/2024.wmt-1.35/).

### Synthetic Data

In order for our MetricX models to learn to identify certain types of bad
translations that are not sufficiently (or at all) represented in the regular
training data, we created synthetic examples and mixed them in during training.
The synthetic training data was generated from the DA datasets ranging from
WMT15 to WMT21 (~ 43 language pairs). In most cases, the synthetic examples have
the candidate translation manipulated so as to turn it into a bad translation
with a specific issue commonly unrecognized by learned metrics.

The table below provides an overview of the various failure modes that we
considered, including brief descriptions of how we prepared the synthetic data
to address them.

| Failure mode | Synthetic example description |
| ----------- | ----------- |
| Undertranslation | Candidate translation with an arbitrary sentence removed (if multi-sentence); alternatively, candidate with a certain proportion of words removed from the end. |
| Overtranslation | Candidate translation duplicated (with space in between). |
| Fluent but unrelated translation | Arbitrary reference of a similar length from the dataset. |
| Gibberish | Text of a similar length as the reference, generated by sampling words from the reference translation vocabulary (built from all references in the data). |
| Missing punctuation | Reference translation with the end punctuation removed (11 punctuation symbols considered). |
| Latin instead of Chinese/Japanese or Hindi/Bengali punctuation | Candidate translation with the language-specific punctuation symbol at the end replaced with the Latin equivalent (e.g., "." instead of "。" or "।"); alternatively, the punctuation symbol is replaced with the Latin equivalent in the reference, keeping the correct one in the candidate. |
| Reference-matching translation | Reference translation copied as the candidate translation (unlike the rest of the synthetic data, these examples are meant to train the metric to predict a perfect score for candidates matching the reference). |

Examples from the first 4 categories were assigned a label corresponding to the
worst score on the given rating scale (e.g., 25 when mixed with MQM training
data), whereas the reference-matching translation examples are assigned the best
score (e.g., 0 when used with MQM data). The missing/incorrect punctuation
examples were labeled with a score slightly worse than perfect.

Note that some of the synthetic datasets are only meaningful in the
reference-based scenario, and we thus excluded them when training a QE variant
of MetricX. These are the Latin-vs-special punctuation and the
reference-matching translation examples.

Most of the synthetic training sets were created using stratified sampling
across target languages, taking 500 examples per target language. One exception
is the missing punctuation set, which used a stratified sample across different
punctuation symbols instead.

When training MetricX, a small proportion of the synthetic examples was mixed
with the regular training examples. During the first-stage fine-tuning on DA
data, each synthetic training set constituted between 0.1% and 1% of all
training examples, whereas in the second-stage fine-tuning on MQM data we used
an even smaller proportion, around 0.05%.

As for evaluating the effect of the synthetic training data on the model's
performance, the DEMETR challenge set - which we originally used to evaluate the
models submitted to the WMT23 Metrics Shared Task - was not adequate anymore. We
therefore created a new DEMETR-style test set based on the WMT22 DA data, with
examples constructed analogically to the synthetic training examples, as
described above. This test set helped us determine the right proportions of
synthetic data for fine-tuning in order to make MetricX robust for the failure
modes in consideration, without sacrificing the system- and segment-level
correlations with human ratings.



## Usage

The `metricx/predict.py` script contains examples
for how to run inference on the models.

```bash
# install
pip install .
```

This installs CLI tool named `metricx-predict` to your $PATH. Alternatively, you may also invoke `python -m metricx.predict -h`.

```bash
metricx-predict -h
usage: metricx-predict [-h] [-t MODEL_ID] -m MODEL [-x INT] [-b INT] [-i FILE] [-o FILE] [-qe] [-tsv]
                       [--debug] [-w WIDTH]

Runs inference with a MetricX model.

options:
  -h, --help            show this help message and exit
  -t MODEL_ID, --tokenizer MODEL_ID
                        The name of the tokenizer.
  -m MODEL, --model_name_or_path MODEL, --model MODEL
                        Path to pretrained model or model identifier from huggingface.co/models
  -x INT, --max_input_length INT
                        The maximum allowable input sequence length, default=-1 => infer: e.g. 1024 for
                        metricx23 and 1536 for metricx24.
  -b INT, --batch_size INT
                        The global prediction batch size.
  -i FILE, --input_file FILE
                        The input file.
  -o FILE, --output_file FILE
                        The output file with predictions .
  -qe, --qe             Indicates the metric is a QE metric.
  -tsv, --tsv           Input_file is a TSV of [source, hypothesis, reference] fields order. When --qe is
                        set. the last column i.e. reference is optional. Also, produces TSV output.
  --debug               Print debug information.
  -w WIDTH, --width WIDTH
                        The width score i.e. number of decimal points.

Knwon models are:
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

The above list maybe incomplete. Search at huggingface.co/models for the latest list.

```


### Reference-Based

Example usage for a reference-based MetricX-24 model:

```bash
python -m metricx.predict \
  --tokenizer google/mt5-xl \
  --model_name_or_path google/metricx-24-hybrid-xl-v2p6 \
  --max_input_length 1536 \
  --batch_size 1 \
  --input_file input.jsonl \
  --output_file output.jsonl
```

`input.jsonl` is expected to have 1 serialized JSON object per line with
`"source"`, `"hypothesis"` and `"reference"` fields. If source segments are not
available, the empty string can be passed as the value of `"source"`. The output
jsonl will be parallel to `input.jsonl` but additionally contain a
`"prediction"` field with the predicted score.

Example usage for a reference-based MetricX-23 model:

```bash
python -m metricx.predict \
  --tokenizer google/mt5-xl \
  --model_name_or_path google/metricx-23-xl-v2p0 \
  --max_input_length 1024 \
  --batch_size 1 \
  --input_file input.jsonl \
  --output_file output.jsonl
```

For use with MetricX-23 models, `input.jsonl` is expected to have 1 serialized
JSON object per line with only the `"hypothesis"` and `"reference"` fields (no
`"source"`).

WARNING: The models were trained with a maximum input length of 1536 and 1024
tokens, respectively, so significantly increasing that value may lead to
unpredictable behavior.

### Reference-Free

Example usage for a reference-free MetricX-24 model:

```bash
python -m metricx.predict \
  --tokenizer google/mt5-xl \
  --model_name_or_path google/metricx-24-hybrid-xl-v2p6 \
  --max_input_length 1536 \
  --batch_size 1 \
  --input_file input.jsonl \
  --output_file output.jsonl \
  --qe
```

Since MetricX-24 models are hybrid, for reference-free evaluation they expect
the same input features as for reference-based evaluation, with the difference
that the `"reference"` field should be passed the empty string. `input.jsonl`
is thus expected to have 1 serialized JSON object per line with `"source"`,
`"hypothesis"` and `"reference"` fields. The output jsonl will be parallel to
`input.jsonl` but additionally contain a `"prediction"` field with the predicted
score.

Example usage for a reference-free MetricX-23 model:

```bash
python -m metricx.predict \
  --tokenizer google/mt5-xl \
  --model_name_or_path google/metricx-23-qe-xl-v2p0 \
  --max_input_length 1024 \
  --batch_size 1 \
  --input_file input.jsonl \
  --output_file output.jsonl \
  --qe
```

For use with MetricX-23 models, `input.jsonl` is expected to have 1 serialized
JSON object per line with `"source"` and `"hypothesis"` fields (no
`"reference"`).


## Meta-Evaluation

The `metricx/evaluate_23.py` and `metricx/evaluate_24.py` scripts contain code to
calculate various correlations between the MetricX scores and MQM ratings of
translation quality using the
[MT Metrics Eval](https://github.com/google-research/mt-metrics-eval) library.

Example usage for a MetricX-24 model:

```bash
python -m metricx.evaluate_24 \
  --dataset wmt24 \
  --lp en-es \
  --input_file input.jsonl \
  --output_file output.json
```

Example usage for a MetricX-23 model:

```bash
python -m metricx.evaluate_23 \
  --dataset wmt22 \
  --lp en-de \
  --input_file input.jsonl \
  --output_file output.json
```

`input.jsonl` is expected to have one JSON object serialized per line.
Each JSON object is expected to contain 4 fields:

* `"system_id"`: The name of the system that generated the translation.
* `"segment_id"`: The 0-based index of the corresponding segment in the MT
Metrics Eval data.
* `"label"`: The ground-truth translation quality score (with higher is better).
* `"prediction"`: The model predicted translation quality score (with lower is
better; the script negates the scores so higher is better).

The script will calculate the 4 agreement/correlations that were used in the
WMT'23 Metrics Shared Task, as well as the system-level soft pairwise accuracy
(SPA), newly added in the WMT'24 Metrics Shared Task. Below are the results for
the MetricX-24 models on the WMT'24 Metrics Shared Task data and for the
MetricX-23 models on the WMT'22 Metrics Shared Task data:



MetricX-24 (WMT'24):

| Model | Sys-Level SPA (en-de) | Seg-Level Acc (en-de) | Sys-Level SPA (en-es) | Seg-Level Acc (en-es) | Sys-Level SPA (ja-zh) | Seg-Level Acc (ja-zh) |
| -------------------------- | ----- | ----- | ----- | ----- | ----- | ----- |
| MetricX-24-Hybrid-XXL      | 0.865 | 0.543 | 0.785 | 0.685 | 0.878 | 0.541 |
| MetricX-24-Hybrid-XL       | 0.884 | 0.522 | 0.806 | 0.683 | 0.859 | 0.528 |
| MetricX-24-Hybrid-Large    | 0.879 | 0.511 | 0.795 | 0.686 | 0.845 | 0.514 |
| MetricX-24-Hybrid-QE-XXL   | 0.884 | 0.525 | 0.789 | 0.685 | 0.863 | 0.527 |
| MetricX-24-Hybrid-QE-XL    | 0.879 | 0.502 | 0.774 | 0.683 | 0.849 | 0.509 |
| MetricX-24-Hybrid-QE-Large | 0.809 | 0.490 | 0.762 | 0.684 | 0.847 | 0.508 |

NOTE: Since MetricX-24 models are hybrid models, MetricX-24-\<size\> and
MetricX-24-QE-\<size\> correspond to the same model, evaluated *with* and
*without* the references, respectively.

NOTE: The performance of the bfloat16 model counterparts deviates by less than
0.001 from the above results across both metrics and all language pairs.

MetricX-23 (WMT'22 en-de):

| Model      | Sys-Level Acc | Sys-Level Pearson | Seg-Level Pearson | Seg-Level Acc |
| ----------- | ----------- | ----------- | ----------- | ----------- |
| MetricX-23-XXL      | 0.795       | 0.835       | 0.546       | 0.619       |
| MetricX-23-XL   | 0.756        | 0.813       | 0.540       | 0.605       |
| MetricX-23-Large   | 0.769        | 0.759       | 0.507       | 0.595       |
| MetricX-23-QE-XXL   | 0.769        | 0.830       | 0.490       | 0.606       |
| MetricX-23-QE-XL   | 0.718        | 0.684       | 0.421       | 0.594       |
| MetricX-23-QE-Large   | 0.744        | 0.671       | 0.387       | 0.579       |

MetricX-23 (WMT'22 en-ru):

| Model      | Sys-Level Acc | Sys-Level Pearson | Seg-Level Pearson | Seg-Level Acc |
| ----------- | ----------- | ----------- | ----------- | ----------- |
| MetricX-23-XXL      | 0.905       | 0.943       | 0.477       | 0.609       |
| MetricX-23-XL   | 0.876        | 0.906       | 0.498       | 0.589       |
| MetricX-23-Large   | 0.876        | 0.841       | 0.474       | 0.569       |
| MetricX-23-QE-XXL   | 0.895        | 0.940       | 0.470       | 0.602       |
| MetricX-23-QE-XL   | 0.848        | 0.861       | 0.415       | 0.570       |
| MetricX-23-QE-Large   | 0.819        | 0.778       | 0.411       | 0.551       |

MetricX-23 (WMT'22 zh-en):

| Model      | Sys-Level Acc | Sys-Level Pearson | Seg-Level Pearson | Seg-Level Acc |
| ----------- | ----------- | ----------- | ----------- | ----------- |
| MetricX-23-XXL      | 0.868       | 0.919       | 0.605       | 0.551       |
| MetricX-23-XL   | 0.868        | 0.924       | 0.584       | 0.543       |
| MetricX-23-Large   | 0.857        | 0.919       | 0.555       | 0.539       |
| MetricX-23-QE-XXL   | 0.857        | 0.928       | 0.573       | 0.544       |
| MetricX-23-QE-XL   | 0.802        | 0.879       | 0.546       | 0.529       |
| MetricX-23-QE-Large   | 0.758        | 0.904       | 0.522       | 0.529       |


The `metricx/evaluate_wmt24.py` script re-calculates the average correlation
score for MetricX-24 that was used to rank submissions from the
[WMT'24 Metrics Shared Task](https://www2.statmt.org/wmt24/pdf/2024.wmt-1.2.pdf).
Similarly, the `metricx/evaluate_wmt23.py` script re-calculates the average
correlation score for MetricX-23 that was used to rank submissions from the
[WMT'23 Metrics Shared Task](https://www2.statmt.org/wmt23/pdf/2023.wmt-1.51.pdf).

Example usage for a MetricX-24 model:

```bash
python -m metricx.evaluate_wmt24 \
  --en_de predictions_ende.jsonl \
  --en_es predictions_enes.jsonl \
  --ja_zh predictions_jazh.jsonl \
  --output_file output.json
```

Example usage for a MetricX-23 model:

```bash
python -m metricx.evaluate_wmt23 \
  --en_de predictions_ende.jsonl \
  --he_en predictions_heen.jsonl \
  --zh_en predictions_zhen.jsonl \
  --output_file output.json
```

Each of the 3 input files is expected to be in the same format as described
above. Each file should correspond to running inference on each of the language
pairs from the WMT'24 / WMT'23 dataset. Below are the results for the MetricX-24
models on the WMT'24 Metrics Shared Task data and for the MetricX-23 models on
the WMT'22 Metrics Shared Task data:

MetricX-24 (WMT'24):

| Model | Average Correlation |
| -------------------------- | ----- |
| MetricX-24-Hybrid-XXL      | 0.716 |
| MetricX-24-Hybrid-XL       | 0.714 |
| MetricX-24-Hybrid-Large    | 0.705 |
| MetricX-24-Hybrid-QE-XXL   | 0.712 |
| MetricX-24-Hybrid-QE-XL    | 0.699 |
| MetricX-24-Hybrid-QE-Large | 0.683 |

NOTE: Since MetricX-24 models are hybrid models, MetricX-24-\<size\> and
MetricX-24-QE-\<size\> correspond to the same model, evaluated *with* and
*without* the references, respectively.

MetricX-23 (WMT'23):

| Model | Average Correlation |
| ------------------- | ----- |
| MetricX-23-XXL      | 0.812 |
| MetricX-23-XL       | 0.813 |
| MetricX-23-Large    | 0.794 |
| MetricX-23-QE-XXL   | 0.797 |
| MetricX-23-QE-XL    | 0.767 |
| MetricX-23-QE-Large | 0.762 |


## Citation

If you use MetricX-24 in your research, please cite the following publication:

```bibtex
@inproceedings{juraska-etal-2024-metricx,
    title = "{M}etric{X}-24: The {G}oogle Submission to the {WMT} 2024 Metrics Shared Task",
    author = "Juraska, Juraj  and
      Deutsch, Daniel  and
      Finkelstein, Mara  and
      Freitag, Markus",
    editor = "Haddow, Barry  and
      Kocmi, Tom  and
      Koehn, Philipp  and
      Monz, Christof",
    booktitle = "Proceedings of the Ninth Conference on Machine Translation",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.wmt-1.35",
    pages = "492--504",
}
```

If you use MetricX-23 in your research, please cite the following publication:

```bibtex
@inproceedings{juraska-etal-2023-metricx,
    title = {{MetricX-23: The Google Submission to the WMT 2023 Metrics Shared Task}},
    author = "Juraska, Juraj  and
      Finkelstein, Mara  and
      Deutsch, Daniel  and
      Siddhant, Aditya  and
      Mirzazadeh, Mehdi  and
      Freitag, Markus",
    editor = "Koehn, Philipp  and
      Haddow, Barry  and
      Kocmi, Tom  and
      Monz, Christof",
    booktitle = "Proceedings of the Eighth Conference on Machine Translation",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.wmt-1.63",
    doi = "10.18653/v1/2023.wmt-1.63",
    pages = "756--767",
}
```