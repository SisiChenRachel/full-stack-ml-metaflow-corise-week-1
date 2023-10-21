from metaflow import (
    FlowSpec,
    step,
    Flow,
    current,
    Parameter,
    IncludeFile,
    card,
    current,
)
from metaflow.cards import Table, Markdown, Artifact

# TODO move your labeling function from earlier in the notebook here
labeling_function = lambda row: 1 if row['rating'] >= 4 else 0



class BaselineNLPFlow(FlowSpec):
    # We can define input parameters to a Flow using Parameters
    # More info can be found here https://docs.metaflow.org/metaflow/basics#how-to-define-parameters-for-flows
    split_size = Parameter("split-sz", default=0.2)
    # In order to use a file as an input parameter for a particular Flow we can use IncludeFile
    # More information can be found here https://docs.metaflow.org/api/flowspec#includefile
    data = IncludeFile("data", default="../data/Womens Clothing E-Commerce Reviews.csv")

    @step
    def start(self):
        # Step-level dependencies are loaded within a Step, instead of loading them
        # from the top of the file. This helps us isolate dependencies in a tight scope.
        import pandas as pd
        import io
        from sklearn.model_selection import train_test_split

        # load dataset packaged with the flow.
        # this technique is convenient when working with small datasets that need to move to remove tasks.
        df = pd.read_csv(io.StringIO(self.data))

        # filter down to reviews and labels
        df.columns = ["_".join(name.lower().strip().split()) for name in df.columns]
        df["review_text"] = df["review_text"].astype("str")
        _has_review_df = df[df["review_text"] != "nan"]
        reviews = _has_review_df["review_text"]
        labels = _has_review_df.apply(labeling_function, axis=1)
        # Storing the Dataframe as an instance variable of the class
        # allows us to share it across all Steps
        # self.df is referred to as a Data Artifact now
        # You can read more about it here https://docs.metaflow.org/metaflow/basics#artifacts
        self.df = pd.DataFrame({"label": labels, **_has_review_df})
        del df
        del _has_review_df

        # split the data 80/20, or by using the flow's split-sz CLI argument
        _df = pd.DataFrame({"review": reviews, "label": labels})
        self.traindf, self.valdf = train_test_split(_df, test_size=self.split_size)
        print(f"num of rows in train set: {self.traindf.shape[0]}")
        print(f"num of rows in validation set: {self.valdf.shape[0]}")

        self.next(self.baseline)

    @step
    def baseline(self):
        "Compute the baseline"
        from sklearn.metrics import accuracy_score, roc_auc_score

        ### TODO: Fit and score a baseline model on the data, log the acc and rocauc as artifacts.
        
        self.traindf["model_1"] = 1
        self.base_acc = accuracy_score(self.traindf["label"], self.traindf["model_1"])
        self.base_rocauc = roc_auc_score(self.traindf["label"], self.traindf["model_1"])
        self.next(self.end)

    @card(
        type="corise"
    )  # TODO: after you get the flow working, chain link on the left side nav to open your card!
    @step
    def end(self):
        import pandas as pd
        msg = "Baseline Accuracy: {}\nBaseline AUC: {}"
        print(msg.format(round(self.base_acc, 3), round(self.base_rocauc, 3)))

        current.card.append(Markdown("# Womens Clothing Review Results"))
        current.card.append(Markdown("## Overall Accuracy"))
        current.card.append(Artifact(self.base_acc))

        # TODO: compute the false positive predictions where the baseline is 1 and the valdf label is 0.

                # Initialize counters for false positives
        self.valdf["model_1"] = 1


        data = {
            'predicted': self.valdf["model_1"],
            'true_label':self.valdf["label"]
        }

        df_t1 = pd.DataFrame(data)
             # Filter the DataFrame for false negatives
        false_positives = df_t1[(df_t1['predicted'] == 1) & (df_t1['true_label'] == 0)].sum()
        false_positives_df = df_t1[(df_t1['predicted'] == 1) & (df_t1['true_label'] == 0)]

        # Print the number of false positives
        print("False Positives:", false_positives)
        current.card.append(Markdown("## Examples of False Positives"))
        # TODO: display the false_positives dataframe using metaflow.cards
        # Documentation: https://docs.metaflow.org/api/cards#table
        current.card.append(
            Table.from_dataframe(
                false_positives_df 
                )
            )
        # TODO: compute the false_negatives predictions where the baseline is 0 and the valdf label is 1.e

        false_negatives = df_t1[(df_t1['predicted'] == 0) & (df_t1['true_label'] == 1)].sum()
        false_negatives_df = df_t1[(df_t1['predicted'] == 0) & (df_t1['true_label'] == 1)]

        # Print the number of false positives
        print("False Negatives:", false_negatives)
        current.card.append(Markdown("## Examples of False Negatives"))
        # TODO: display the false_negatives dataframe using metaflow.cards
        current.card.append(
            Table.from_dataframe(
                false_negatives_df 
                )
            )

if __name__ == "__main__":
    BaselineNLPFlow()
