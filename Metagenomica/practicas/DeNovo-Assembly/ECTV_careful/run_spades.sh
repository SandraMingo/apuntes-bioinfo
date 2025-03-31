set -e
true
true
/home/sandra/miniconda3/envs/ngs/bin/spades-hammer /media/sf_METAG/unit_3/ECTV_careful/corrected/configs/config.info
/home/sandra/miniconda3/envs/ngs/bin/python3 /home/sandra/miniconda3/envs/ngs/share/spades/spades_pipeline/scripts/compress_all.py --input_file /media/sf_METAG/unit_3/ECTV_careful/corrected/corrected.yaml --ext_python_modules_home /home/sandra/miniconda3/envs/ngs/share/spades --max_threads 2 --output_dir /media/sf_METAG/unit_3/ECTV_careful/corrected --gzip_output
true
true
/home/sandra/miniconda3/envs/ngs/bin/spades-core /media/sf_METAG/unit_3/ECTV_careful/K21/configs/config.info /media/sf_METAG/unit_3/ECTV_careful/K21/configs/careful_mode.info
/home/sandra/miniconda3/envs/ngs/bin/spades-core /media/sf_METAG/unit_3/ECTV_careful/K33/configs/config.info /media/sf_METAG/unit_3/ECTV_careful/K33/configs/careful_mode.info
/home/sandra/miniconda3/envs/ngs/bin/spades-core /media/sf_METAG/unit_3/ECTV_careful/K55/configs/config.info /media/sf_METAG/unit_3/ECTV_careful/K55/configs/careful_mode.info
/home/sandra/miniconda3/envs/ngs/bin/python3 /home/sandra/miniconda3/envs/ngs/share/spades/spades_pipeline/scripts/copy_files.py /media/sf_METAG/unit_3/ECTV_careful/K55/before_rr.fasta /media/sf_METAG/unit_3/ECTV_careful/before_rr.fasta /media/sf_METAG/unit_3/ECTV_careful/K55/assembly_graph_after_simplification.gfa /media/sf_METAG/unit_3/ECTV_careful/assembly_graph_after_simplification.gfa /media/sf_METAG/unit_3/ECTV_careful/K55/final_contigs.fasta /media/sf_METAG/unit_3/ECTV_careful/contigs.fasta /media/sf_METAG/unit_3/ECTV_careful/K55/first_pe_contigs.fasta /media/sf_METAG/unit_3/ECTV_careful/first_pe_contigs.fasta /media/sf_METAG/unit_3/ECTV_careful/K55/strain_graph.gfa /media/sf_METAG/unit_3/ECTV_careful/strain_graph.gfa /media/sf_METAG/unit_3/ECTV_careful/K55/scaffolds.fasta /media/sf_METAG/unit_3/ECTV_careful/scaffolds.fasta /media/sf_METAG/unit_3/ECTV_careful/K55/scaffolds.paths /media/sf_METAG/unit_3/ECTV_careful/scaffolds.paths /media/sf_METAG/unit_3/ECTV_careful/K55/assembly_graph_with_scaffolds.gfa /media/sf_METAG/unit_3/ECTV_careful/assembly_graph_with_scaffolds.gfa /media/sf_METAG/unit_3/ECTV_careful/K55/assembly_graph.fastg /media/sf_METAG/unit_3/ECTV_careful/assembly_graph.fastg /media/sf_METAG/unit_3/ECTV_careful/K55/final_contigs.paths /media/sf_METAG/unit_3/ECTV_careful/contigs.paths
true
true
/home/sandra/miniconda3/envs/ngs/bin/python3 /home/sandra/miniconda3/envs/ngs/share/spades/spades_pipeline/scripts/correction_iteration_script.py --corrected /media/sf_METAG/unit_3/ECTV_careful/contigs.fasta --assembled /media/sf_METAG/unit_3/ECTV_careful/misc/assembled_contigs.fasta --assembly_type contigs --output_dir /media/sf_METAG/unit_3/ECTV_careful --bin_home /home/sandra/miniconda3/envs/ngs/bin
/home/sandra/miniconda3/envs/ngs/bin/python3 /home/sandra/miniconda3/envs/ngs/share/spades/spades_pipeline/scripts/correction_iteration_script.py --corrected /media/sf_METAG/unit_3/ECTV_careful/scaffolds.fasta --assembled /media/sf_METAG/unit_3/ECTV_careful/misc/assembled_scaffolds.fasta --assembly_type scaffolds --output_dir /media/sf_METAG/unit_3/ECTV_careful --bin_home /home/sandra/miniconda3/envs/ngs/bin
true
/home/sandra/miniconda3/envs/ngs/bin/python3 /home/sandra/miniconda3/envs/ngs/share/spades/spades_pipeline/scripts/breaking_scaffolds_script.py --result_scaffolds_filename /media/sf_METAG/unit_3/ECTV_careful/scaffolds.fasta --misc_dir /media/sf_METAG/unit_3/ECTV_careful/misc --threshold_for_breaking_scaffolds 3
true
