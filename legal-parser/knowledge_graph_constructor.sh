#!/bin/bash
#SBATCH --job-name=preprocess_voice
#SBATCH --output=out/preprocess_voice_JOB_%j.out
#SBATCH --error=err/preprocess_voice_JOB_%j.err
#SBATCH --nodes=1 # As we have single node it should be always set as 1
#SBATCH --cpus-per-task=4 # Number of CPUs
#SBATCH --gres=gpu:1g.10gb:1  # Allocate 1 GPU resources with specified configurations
#SBATCH --mem=32G  # Specify the total amount of memory
#SBATCH --time=72:00:00  # Set the time limit to 1 minute
#SBATCH --partition=debugging
#SBATCH --qos=debugging
#SBATCH --account=debugging


# python knowledge_graph_constructor.py \
#     --parsed_main "../data/stvo/parsed_main_content_Straßenverkehrs_Ordnung_de.json" \
#     --parsed_table "../data/stvo/parsed_table_content_Straßenverkehrs_Ordnung_de.json" \
#     --main_graph "main_content_Straßenverkehrs_Ordnung_graph_de" \
#     --table_graph "table_content_Straßenverkehrs_Ordnung_graph_de" \
#     --integrated_graph "integrated_Straßenverkehrs_Ordnung_graph_de" \
#     --out_dir "../data/stvo/de"


python knowledge_graph_constructor.py \
    --parsed_main "../data/stvo/m2m100_418M_translated_main_content_Straßenverkehrs_Ordnung.json" \
    --parsed_table "../data/stvo/m2m100_418M_translated_table_content_Straßenverkehrs_Ordnung.json" \
    --main_graph "m2m100_418M_translated_main_content_Straßenverkehrs_Ordnung_graph" \
    --table_graph "m2m100_418M_translated_table_content_Straßenverkehrs_Ordnung_graph" \
    --integrated_graph "m2m100_418M_translated_integrated_Straßenverkehrs_Ordnung_graph"\
    --out_dir "../data/stvo"



# python knowledge_graph_constructor.py \
#     --parsed_main "../data/stvo/opus-mt-de-en_translated_main_content_Straßenverkehrs_Ordnung.json" \
#     --parsed_table "../data/stvo/opus-mt-de-en_translated_table_content_Straßenverkehrs_Ordnung.json" \
#     --main_graph "opus-mt-de-en_translated_main_content_Straßenverkehrs_Ordnung_graph" \
#     --table_graph "opus-mt-de-en_translated_table_content_Straßenverkehrs_Ordnung_graph" \
#     --integrated_graph "opus-mt-de-en_translated_integrated_Straßenverkehrs_Ordnung_graph" \
#     --out_dir "../data/stvo"





# # opus-mt-de-en_translated_main_content_BGB
# python knowledge_graph_constructor.py \
#     --parsed_main "../data/BGB/opus-mt-de-en_translated_main_content_BGB.json" \
#     --parsed_table "../data/BGB/opus-mt-de-en_translated_table_content_BGB.json" \
#     --main_graph "opus-mt-de-en_translated_main_content_BGB_graph" \
#     --integrated_graph "opus-mt-de-en_translated_BGB_graph" \
#     --out_dir "../data/BGB"


# # m2m100_418M_translated_main_content_BGB
# python knowledge_graph_constructor.py \
#     --parsed_main "../data/BGB/m2m100_418M_translated_main_content_BGB.json" \
#     --parsed_table "../data/BGB/m2m100_418M_translated_table_content_BGB.json" \
#     --main_graph "m2m100_418M_translated_main_content_BGB_graph" \
#     --integrated_graph "m2m100_418M_translated_BGB_graph" \
#     --out_dir "../data/BGB"







