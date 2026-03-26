# Straßenverkehrs-Ordnung (StVO)

# # Main content
python main_content_extractor.py --url https://www.gesetze-im-internet.de/stvo_2013/BJNR036710013.html \
                                     --base_url https://www.gesetze-im-internet.de \
                                     --save_image_dir "./legal-images/main_content_Straßenverkehrs_Ordnung_images"\
                                      --save_data_dir "parsed_main_content_Straßenverkehrs_Ordnung.json"

# table content 
python table_content_extractor.py --url https://www.gesetze-im-internet.de/stvo_2013/BJNR036710013.html \
                                     --base_url https://www.gesetze-im-internet.de \
                                     --save_image_dir "./legal-images/table_content_Straßenverkehrs_Ordnung_images"\
                                      --save_data_dir "parsed_table_content_Straßenverkehrs_Ordnung.json"



# https://www.gesetze-im-internet.de/bgb/BJNR001950896.html
python main_content_extractor.py --url https://www.gesetze-im-internet.de/bgb/BJNR001950896.html \
                                     --base_url https://www.gesetze-im-internet.de \
                                     --save_image_dir "./legal-images/main_content_BGB_images"\
                                     --save_data_dir "parsed_main_content_BGB.json"

# python table_content_extractor.py --url https://www.gesetze-im-internet.de/bgb/BJNR001950896.html \
#                                      --base_url https://www.gesetze-im-internet.de \
#                                      --save_image_dir "./legal-images/table_content_BGB_images"\
#                                      --save_data_dir "./nmt/parsed_table_content_BGB.json"

############################################################################################################


# # Straßenverkehrs-Zulassungs-Ordnung (StVZO)
# # Main content 
# python main_content_extractor.py --url https://www.gesetze-im-internet.de/stvzo_2012/BJNR067910012.html\
#                                         --base_url https://www.gesetze-im-internet.de \
#                                         --save_image_dir main_content_Straßenverkehrs_Zulassungs_Ordnung_images\
#                                         --save_data_dir parsed_main_content_Straßenverkehrs_Zulassungs_Ordnung.json

# # table content
# python table_content_extractor.py --url https://www.gesetze-im-internet.de/stvzo_2012/BJNR067910012.html\
#                                         --base_url https://www.gesetze-im-internet.de \
#                                         --save_image_dir table_content_Straßenverkehrs_Zulassungs_Ordnung_images\
#                                         --save_data_dir parsed_table_content_Straßenverkehrs_Zulassungs_Ordnung.json





# #  Verordnung über die Zulassung von Fahrzeugen zum Straßenverkehr (Fahrzeug-Zulassungsverordnung - FZV)
# # Main content
# python main_content_extractor.py --url https://www.gesetze-im-internet.de/fzv_2023/BJNR0C70B0023.html\
#                                         --base_url https://www.gesetze-im-internet.de \
#                                         --save_image_dir main_content_Fahrzeug_Zulassungsverordnung_images\
#                                         --save_data_dir parsed_main_content_Fahrzeug_Zulassungsverordnung.json

# # table content
# python table_content_extractor.py --url https://www.gesetze-im-internet.de/fzv_2023/BJNR0C70B0023.html\
#                                         --base_url https://www.gesetze-im-internet.de \
#                                         --save_image_dir table_content_Fahrzeug_Zulassungsverordnung_images\
#                                         --save_data_dir parsed_table_content_Fahrzeug_Zulassungsverordnung.json