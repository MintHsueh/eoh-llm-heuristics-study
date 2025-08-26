from eoh import eoh                     # 載入主框架
from eoh.utils.getParas import Paras    # 載入參數管理

# Parameter initilization # 創建參數物件
paras = Paras() 

# Set parameters # 設定參數
paras.set_paras(method = "eoh",    # ['ael','eoh']
                problem = "tsp_construct", #['tsp_construct','bp_online']
                llm_api_endpoint = "api.deepseek.com", # set your LLM endpoint
                llm_api_key = "sk-xxxxxxxxxxxxxxxxxxxx",   # set your key
                llm_model = "deepseek-chat",
                ec_pop_size = 5, # number of samples in each population
                ec_n_pop = 4,  # number of populations
                exp_n_proc = 4,  # multi-core parallel
                exp_debug_mode = False)
                

# initilization # 創建演化物件
evolution = eoh.EVOL(paras)

# run 開始執行
evolution.run()