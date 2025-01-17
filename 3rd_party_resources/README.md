# ConvAssist 3rd Party Resources

## Datasets

TODO: Explanation of why we need conversation datasets for the predictors

Two datasets are provided by ConvAssist:
* [Human Conversation Training Data](human-conversation/README.md)
* [Personachat](personalchat/README/md)

### Licensing

> [!NOTE]
> Datasets included in the ConvAssist project are subject to the license provided.  We make no
> claim of copyright to the data provided by.  See specific license information for each
> dataset.

### "Bring your own conversation dataset"

If you would like to extend ConvAssist with your own converstaion dataset:
* Conversation data **MUST** be in plain text format.
* Each sentence should be on a separate line.
* ConvAssist expects the conversation data to be in a single file.  You may combine multiple datasets to provide a more robust library to the predictors.

The following Predictors support custom conversation dataset(s)

| Predictor | .ini Setting |
| --- | --- |
| GeneralWordPredictor| aac_dataset |
| SmoothedNGramPredictor* | N/A |
| SentencePredictor | retrieve_database |

>\* NOTE
The SmoothedNGramPredictor uses an AAC like dataset to populate the NGram database used for predictions.  ConvAssist uses the data from [Human Conversation Training Data](human-conversation/README.md) in the [ACATConvAssist](../interfaces/ACAT/acatconvassist/) utility as well as the [Continuous Predictor](../interfaces/Demos/continuous_predict/) Demo application.