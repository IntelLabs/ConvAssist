import os
from configparser import ConfigParser
from ACAT_ConvAssist_Interface.ConvAssistCPApp.ConvAssistCPApp import ConvAssistVariables


def createPredictorConfig(ca_vars:ConvAssistVariables, config_file_name) -> ConfigParser | None:
    # check if all required params are set
    if not ca_vars.path or not ca_vars.pathstatic or not ca_vars.pathpersonalized:
        ca_vars.logger.debug("Path not set")
        return None
    
    # Check if config file exists
    config_file = os.path.join(ca_vars.path, config_file_name)
    if not os.path.exists(config_file):
        ca_vars.logger.debug("Config file not found")
        return None
    
    # Create a ConfigParser object and read the config file
    config_parser = ConfigParser()
    config_parser.read(config_file)

    # Check if the config file is read successfully
    if not config_parser:
        ca_vars.logger.debug("Config file not read successfully")
        return None
    
    # make additional customizations to the ConfigParser object
    for section in config_parser.sections():
        for key in config_parser[section]:
            if key == "static_resources_path":
                config_parser[section][key] = ca_vars.pathstatic
            elif key == "personalized_resources_path":
                config_parser[section][key] = ca_vars.pathpersonalized
            elif key == "suggestions":
                config_parser[section][key] = str(ca_vars.suggestions)
            elif key == "test_generalSentencePrediction":
                config_parser[section][key] = str(ca_vars.testgensentencepred)
            elif key == "retrieveAAC":
                config_parser[section][key] = str(ca_vars.retrieveaac)
            elif key == "log_location":
                config_parser[section][key] = ca_vars.pathlog
            elif key == "log_level":
                config_parser[section][key] = ca_vars.loglevel

    # Write the config file
    with open(config_file, 'w') as configfile:
        config_parser.write(configfile)

    #return the ConfigParser object
    return config_parser