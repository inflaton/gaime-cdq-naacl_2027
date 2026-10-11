# Third-party notices and license scope

## Original contributions

The [BSD 3-Clause License](LICENSE) covers the original code, scripts, prompt template,
documentation and author-created analysis artifacts in this repository. It grants only
rights held by the repository contributors and does not relicense third-party content.

## Walmart–Amazon data

The experiments use the raw Walmart–Amazon release linked from the
[DeepMatcher dataset documentation](https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md#walmart-amazon).
That documentation attributes the product data to the University of Wisconsin–Madison
entity-matching data collection, with records from Walmart and Amazon.

* [Original dataset directory](https://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/Walmart-Amazon/)
* [Raw archive](https://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/Walmart-Amazon/walmart_amazon_raw_data.zip)
* Archive SHA-256: `c49a755801896b828ec68438cb39fc9fa1f127bcc716a26aaebbb05308a71c85`

The archive contains `tableA.csv`, `tableB.csv` and `matches.csv`, with no separate license
file. The linked dataset documentation does not state a separate license for these raw
records. [DeepMatcher's BSD 3-Clause software license](https://github.com/anhaidgroup/deepmatcher/blob/master/LICENSE)
is the basis for our choice of software license; it is not treated here as permission to
relicense the source data. Dataset use and redistribution remain subject to the source
rights and any applicable upstream terms.

The original archive is downloaded from its host by `prepare` and is not included in this
repository. Some released results, including `accepted.csv`, `curation.csv` and model
response records, contain source product text or references to source labels. That
third-party content is outside this repository's BSD license, even when embedded in a
result file. Our original analysis and annotations do not change the rights in the
underlying records.

## Models and runtime dependencies

The experiments use `google/gemma-4-31B-it` and `meta-models/Muse-Glimmer-30B`, served with
vLLM. Model weights and runtime dependencies are not distributed in this repository and
retain their respective licenses and provider terms. The repository license does not
grant rights to those materials or to any third-party content reproduced in model outputs.
