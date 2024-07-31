def pssm_generation(
    fasta_file,
    output_database="../data/whole_sequence",
    generate_searchdb=True,
    pssm_filename="pssm.pssm",
    outputdir="pssm_test",
) -> str:
    from BioML.utilities.utils import MmseqsClustering

    pssm_file = MmseqsClustering.easy_generate_pssm(
        input_file=fasta_file,
        database_input=fasta_file,
        output_database=output_database,
        generate_searchdb=generate_searchdb,
        pssm_filename=pssm_filename,
        output_dir=outputdir,
    )

    return pssm_file


def extract_features(
    minimal_sequence_length=100, extraction_type="ifeature", fasta_file=None, ifeature_path=None
):
    from BioML.features.extraction import IfeatureFeatures, PossumFeatures, read_features
    from BioML.utilities.utils import clean_fasta

    clean_fasta(
        "/home/ruite/Projects/enzyminer/POSSUM_Toolkit",
        fasta_file,
        "cleaned.fasta",
        minimal_sequence_length,
    )
    fasta_file = "cleaned.fasta"

    if extraction_type == "ifeature":
        ifeatures = IfeatureFeatures(ifeature_path)
        ifeatures.extract(fasta_file)

    elif extraction_type == "possum":
        possum = PossumFeatures(
            pssm_dir="pssm",
            output="possum_features",
            program="/home/ruite/Projects/enzyminer/POSSUM_Toolkit",
        )
        possum.extract(fasta_file)
