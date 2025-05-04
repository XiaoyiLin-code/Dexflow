# config.py
import os

class Path_config:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 设置各个路径的相对路径
    DATA_PATH = os.path.join(BASE_DIR, 'data')
    SCRIPTS = os.path.join(BASE_DIR, 'scritps')
    OUTPUT_PATH = os.path.join(BASE_DIR, 'output')
    THIRD_PARTY_PATH = os.path.join(BASE_DIR, 'third_party')
    DATA_UTILS_PATH = os.path.join(BASE_DIR, 'data_utils')
    RETARGET_UTILS_PATH = os.path.join(BASE_DIR, 'retarget_utils')


    @staticmethod
    def get_data_path():
        return Path_config.DATA_PATH

    @staticmethod
    def get_scripts_path():
        return Path_config.SCRIPTS

    @staticmethod
    def get_output_path():
        return Path_config.OUTPUT_PATH
    
    @staticmethod
    def get_third_party_path():
        return Path_config.THIRD_PARTY_PATH

if __name__ == '__main__':
    path_config=Path_config()
    print(path_config.get_data_path())
    print(path_config.get_scripts_path())
    print(path_config.get_output_path())
    print(path_config.get_third_party_path())