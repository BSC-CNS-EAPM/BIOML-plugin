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
