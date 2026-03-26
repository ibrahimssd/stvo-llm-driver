# Straßenverkehrs-Ordnung (StVO)
python parser_eval.py  --url 'https://www.gesetze-im-internet.de/stvo_2013/BJNR036710013.html' \
                        --base_url 'https://www.gesetze-im-internet.de' \
                        --parsed_main "parsed_main_content_Straßenverkehrs_Ordnung.json" \
                        --parsed_table "parsed_table_content_Straßenverkehrs_Ordnung.json" \
                        --save_original "Straßenverkehrs_Ordnung_orignal.txt" \
                        --save_reconstructed "Straßenverkehrs_Ordnung_reconstructed.txt"\
                        --save_scores "Straßenverkehrs_Ordnung.png"
                        

############################################################################################################################################################################


# Straßenverkehrs-Zulassungs-Ordnung (StVZO)
# Main content 

# https://www.gesetze-im-internet.de/stvzo_2012/BJNR067910012.html

python parser_eval.py  --url 'https://www.gesetze-im-internet.de/stvzo_2012/BJNR067910012.html' \
                        --base_url 'https://www.gesetze-im-internet.de' \
                        --parsed_main "parsed_main_content_Straßenverkehrs_Zulassungs_Ordnung.json" \
                        --parsed_table "parsed_table_content_Straßenverkehrs_Zulassungs_Ordnung.json" \
                        --save_original "Straßenverkehrs_Zulassungs_Ordnung_orignal.txt" \
                        --save_reconstructed "Straßenverkehrs_Zulassungs_Ordnung_reconstructed.txt"\
                        --save_scores "Straßenverkehrs_Zulassungs_Ordnung.png"



#  Verordnung über die Zulassung von Fahrzeugen zum Straßenverkehr (Fahrzeug-Zulassungsverordnung - FZV)
# https://www.gesetze-im-internet.de/fzv_2023/BJNR0C70B0023.html

python parser_eval.py --url 'https://www.gesetze-im-internet.de/fzv_2023/BJNR0C70B0023.html' \
                        --base_url 'https://www.gesetze-im-internet.de' \
                        --parsed_main "parsed_main_content_Fahrzeug_Zulassungsverordnung.json" \
                        --parsed_table "parsed_table_content_Fahrzeug_Zulassungsverordnung.json" \
                        --save_original "Fahrzeug_Zulassungsverordnung_orignal.txt" \
                        --save_reconstructed "Fahrzeug_Zulassungsverordnung_reconstructed.txt"\
                        --save_scores "Fahrzeug_Zulassungsverordnung.png"