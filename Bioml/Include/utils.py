# TODO needs a database
def pssm_generation(
    input_file,
    output_database="../data/whole_sequence",
    database_input="/home/lavane/sdb/Databases/uniref50.fasta",
    generate_searchdb=True,
    database_output=None,
    pssm_filename="pssm.pssm",
    outputdir="pssm_test",
) -> str:
    from BioML.utilities.utils import MmseqsClustering

    pssm_file = MmseqsClustering.easy_generate_pssm(
        input_file=input_file,
        database_input=database_input,
        output_database=output_database,
        generate_searchdb=generate_searchdb,
        database_output=database_output,
        pssm_filename=pssm_filename,
        output_dir=outputdir,
    )

    return pssm_file


# TODO adaptar a los dos tipos de extraction
def extract_features(
    minimal_sequence_length=100,
    extraction_type="ifeature",
    fasta_file=None,
    ifeature_path="/home/lavane/Repos/iFeature",
    possum_path="/home/lavane/sdb/Programs/POSSUM_Standalone_Toolkit",
):
    from BioML.features.extraction import (
        IfeatureFeatures,
        PossumFeatures,
        read_features,
    )
    from BioML.utilities.utils import clean_fasta

    clean_fasta(
        possum_path,
        fasta_file,
        "cleaned.fasta",
        minimal_sequence_length,
    )
    fasta_file = "cleaned.fasta"

    if extraction_type == "ifeature":
        ifeatures = IfeatureFeatures(ifeature_path)
        extracted_features = ifeatures.extract(fasta_file)

    elif extraction_type == "possum":
        possum = PossumFeatures(
            pssm_dir="pssm",
            output="possum_features",
            program=possum_path,
        )
        extracted_features = possum.extract(fasta_file)

    return extracted_features


def llm_embeddings(
    fasta_file=None,
    model_name="facebook/esm2_t6_8M_UR50D",
    option="mean",
    save_path="embeddings.csv",
    mode="write",
):
    from BioML.deep import embeddings

    embeddings.generate_embeddings(
        fasta_file,
        model_name=model_name,
        option=option,
        save_path=save_path,
        mode=mode,
    )

    return save_path


def feature_selection(
    fasta_file,
    cluster_at_sequence_identity=0.3,
    labels_csv="../data/esterase_labels.csv",
    file_splits=1,
    variance_thres=0.005,
    problem="classification",
    num_threads=2,
    seed=10,
    scaler="robust",
    features_file="classification_results/filtered_features.xlsx",
    extraction_type="ifeature",
    cluster_info="cluster.tsv",
    num_splits=5,
    random_state=100,
    stratified=True,
    num_features_min=20,
    num_features_max=60,
    step_range=10,
):
    import pandas as pd
    from BioML.features import selection
    from BioML.features.extraction import read_features
    from BioML.utilities.split_methods import ClusterSpliter
    from BioML.utilities.utils import MmseqsClustering

    feat = []

    if "ifeature" in extraction_type:
        ifeat = read_features(
            "ifeature", ifeature_out="ifeature_features", file_splits=file_splits
        )
        feat.append(ifeat)
    elif "possum" in extraction_type:
        possum_feat = read_features(
            "possum",
            possum_out="possum_features",
            file_splits=file_splits,
            index=ifeat.index,
        )
        feat.append(possum_feat)
    elif "llm" in extraction_type:
        emb = pd.read_csv("embeddings.csv", index_col=0)
        feat.append(emb)

    features = selection.DataReader(labels_csv, feat, variance_thres=variance_thres)

    #! Why not used the cluster variable?
    # esto genera el "cluster_info"
    _ = MmseqsClustering.easy_cluster(
        fasta_file, cluster_at_sequence_identity=cluster_at_sequence_identity
    )

    split = ClusterSpliter(
        cluster_info,
        num_splits=num_splits,
        random_state=random_state,
        stratified=stratified,
    )
    #! Why not used the y_test variable?
    X_train, X_test, y_train, y_test = split.train_test_split(
        features.features, features.label
    )

    feature_range = selection.get_range_features(
        features.features,
        num_features_min=num_features_min,
        num_features_max=num_features_max,
        step_range=step_range,
    )

    if problem == "classification":
        filters = selection.FeatureClassification()
    elif problem == "regression":
        filters = selection.FeatureRegression()

    select = selection.FeatureSelection(
        features_file,
        filters,
        num_thread=num_threads,
        seed=seed,
        scaler=scaler,
    )

    select.construct_features(
        features.features, X_train, X_test, y_train, feature_range
    )

    return features_file


def outlier_detection(
    features_file="classification_results/filtered_features.xlsx",
    output="classification_results/outliers.csv",
    num_thread=4,
):
    from BioML.utilities.outlier import OutlierDetection

    detection = OutlierDetection(
        features_file,
        output=output,
        num_thread=num_thread,
    )
    outliers = detection.run()

    # outliers dict = outliers
    return output


def model_training(
    problem="classification",
    features="../data/esterase_features.xlsx",
    labels="../data/esterase_labels.csv",
    sheets="ch2_20",
    cluster_info="cluster.tsv",
    num_splits=5,
    random_state=100,
    stratified=True,
    seed=250,
    budget_time=20,
    best_model=3,
    output_path="classification_results",
    optimize="MCC",
    test_size=0.2,
    num_iter=50,
    cross_validation=True,
):
    from BioML.models.base import DataParser, PycaretInterface, Trainer
    from BioML.models.classification import Classifier
    from BioML.models.regression import Regressor
    from BioML.utilities.split_methods import ClusterSpliter
    from BioML.utilities.utils import write_results
    from sklearn.linear_model import PassiveAggressiveClassifier

    data = DataParser(
        features=features,
        label=labels,
        sheets=sheets,
    )
    split = ClusterSpliter(
        cluster_info,
        num_splits=num_splits,
        random_state=random_state,
        stratified=stratified,
    )
    X_train, X_test, _, _ = split.train_test_split(
        data.features, data.features[data.label]
    )

    #! Seems that there are varius models trainings

    experiment = PycaretInterface(
        problem,
        seed,
        budget_time=budget_time,
        best_model=best_model,
        output_path=output_path,
        optimize=optimize,
    )
    if problem == "classification":
        args = Classifier(
            optimize=optimize,
            drop=(),
            selected=(),
            add=(),
            plot=("learning", "confusion_matrix", "class_report"),
        )
    elif problem == "regression":
        args = Regressor(
            optimize=optimize,
            drop=(),
            selected=(),
            add=(),
            plot=("learning", "residuals", "error"),
        )

    training = Trainer(
        experiment,
        args,
        num_splits=num_splits,
        test_size=test_size,
        num_iter=num_iter,
        cross_validation=cross_validation,
    )
    results, models = training.generate_training_results(
        X_train, data.label, tune=True, test_data=X_test, fold_strategy=split
    )

    test_set_predictions = training.generate_holdout_prediction(models)

    training_output = "classification_results"
    l = []
    for tune_status, result_dict in results.items():
        for key, value in result_dict.items():
            write_results(f"{training_output}/{tune_status}", *value, sheet_name=key)
        write_results(
            f"{training_output}/{tune_status}",
            test_set_predictions[tune_status],
            sheet_name=f"test_results",
        )

    from BioML.models import save_model

    generate = save_model.GenerateModel(training)
    for status, model in models.items():
        for key, value in model.items():
            if key == "holdout":
                for num, (name, mod) in enumerate(value.items()):
                    if num > training.experiment.best_model - 1:
                        break
                    final_model = generate.finalize_model(value, num)
                    generate.save_model(
                        final_model,
                        f"classification_results/saved_models/{status}_{name}",
                    )
            else:
                final_model = generate.finalize_model(value)
                generate.save_model(
                    final_model, f"classification_results/saved_models/{key}_{status}"
                )


def prediction(
    training_features="classification_results/filtered_features.xlsx",
    label="../data/esterase_labels.csv",
    outlier_train=(),
    outlier_test=(),
    sheet_name="chi2_60",
    problem="classification",
    model_path="classification_results/saved_models/tuned_rbfsvm",
    scaler="robust",
    fasta="../data/whole_sequence.fasta",
    res_dir="prediction_results_domain",
):

    from BioML.models import predict

    feature = predict.DataParser(
        training_features, label, outliers=outlier_train, sheets=sheet_name
    )
    test_features = feature.remove_outliers(
        feature.read_features(training_features, sheet_name), outlier_test
    )
    predictions = predict.predict(test_features, model_path, problem)

    predictions.index = [f"sample_{x}" for x, _ in enumerate(predictions.index)]
    col_name = ["prediction_score", "prediction_label", "AD_number"]
    predictions = predictions.loc[
        :, predictions.columns.str.contains("|".join(col_name))
    ]

    extractor = predict.FastaExtractor(fasta, res_dir)
    positive, negative = extractor.separate_negative_positive(predictions)
    extractor.extract(
        positive,
        negative,
        positive_fasta="classification_results/positive.fasta",
        negative_fasta=f"classification_results/negative.fasta",
    )
