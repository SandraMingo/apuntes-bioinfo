set -e
true
true
/home/sandra/miniconda3/envs/ngs/bin/spades-hammer /media/sf_METAG/unit_3b/virome_1_sc/corrected/configs/config.info
/home/sandra/miniconda3/envs/ngs/bin/python3 /home/sandra/miniconda3/envs/ngs/share/spades/spades_pipeline/scripts/compress_all.py --input_file /media/sf_METAG/unit_3b/virome_1_sc/corrected/corrected.yaml --ext_python_modules_home /home/sandra/miniconda3/envs/ngs/share/spades --max_threads 4 --output_dir /media/sf_METAG/unit_3b/virome_1_sc/corrected --gzip_output
true
true
/home/sandra/miniconda3/envs/ngs/bin/spades-core /media/sf_METAG/unit_3b/virome_1_sc/K21/configs/config.info /media/sf_METAG/unit_3b/virome_1_sc/K21/configs/mda_mode.info
/home/sandra/miniconda3/envs/ngs/bin/spades-core /media/sf_METAG/unit_3b/virome_1_sc/K33/configs/config.info /media/sf_METAG/unit_3b/virome_1_sc/K33/configs/mda_mode.info
/home/sandra/miniconda3/envs/ngs/bin/spades-core /media/sf_METAG/unit_3b/virome_1_sc/K55/configs/config.info /media/sf_METAG/unit_3b/virome_1_sc/K55/configs/mda_mode.info
/home/sandra/miniconda3/envs/ngs/bin/python3 /home/sandra/miniconda3/envs/ngs/share/spades/spades_pipeline/scripts/copy_files.py /media/sf_METAG/unit_3b/virome_1_sc/K55/before_rr.fasta /media/sf_METAG/unit_3b/virome_1_sc/before_rr.fasta /media/sf_METAG/unit_3b/virome_1_sc/K55/assembly_graph_after_simplification.gfa /media/sf_METAG/unit_3b/virome_1_sc/assembly_graph_after_simplification.gfa /media/sf_METAG/unit_3b/virome_1_sc/K55/final_contigs.fasta /media/sf_METAG/unit_3b/virome_1_sc/contigs.fasta /media/sf_METAG/unit_3b/virome_1_sc/K55/first_pe_contigs.fasta /media/sf_METAG/unit_3b/virome_1_sc/first_pe_contigs.fasta /media/sf_METAG/unit_3b/virome_1_sc/K55/strain_graph.gfa /media/sf_METAG/unit_3b/virome_1_sc/strain_graph.gfa /media/sf_METAG/unit_3b/virome_1_sc/K55/scaffolds.fasta /media/sf_METAG/unit_3b/virome_1_sc/scaffolds.fasta /media/sf_METAG/unit_3b/virome_1_sc/K55/scaffolds.paths /media/sf_METAG/unit_3b/virome_1_sc/scaffolds.paths /media/sf_METAG/unit_3b/virome_1_sc/K55/assembly_graph_with_scaffolds.gfa /media/sf_METAG/unit_3b/virome_1_sc/assembly_graph_with_scaffolds.gfa /media/sf_METAG/unit_3b/virome_1_sc/K55/assembly_graph.fastg /media/sf_METAG/unit_3b/virome_1_sc/assembly_graph.fastg /media/sf_METAG/unit_3b/virome_1_sc/K55/final_contigs.paths /media/sf_METAG/unit_3b/virome_1_sc/contigs.paths
true
/home/sandra/miniconda3/envs/ngs/bin/python3 /home/sandra/miniconda3/envs/ngs/share/spades/spades_pipeline/scripts/breaking_scaffolds_script.py --result_scaffolds_filename /media/sf_METAG/unit_3b/virome_1_sc/scaffolds.fasta --misc_dir /media/sf_METAG/unit_3b/virome_1_sc/misc --threshold_for_breaking_scaffolds 3
true
