class GetPrompts():
    def __init__(self):
#         self.prompt_task = "I need help designing a novel score function that scoring a set of bins to assign an item. \
# In each step, the item will be assigned to the bin with the maximum score. If the rest capacity of a bin equals the maximum capacity, it will not be used. The final goal is to minimize the number of used bins."
        # chatgpt
        self.prompt_task = "I need help designing a novel score function that scores a set of bins to assign an item. \
In each step, the item will be assigned to the bin with the maximum score. If the rest capacity of a bin equals the maximum capacity, it will not be used. The final goal is to minimize the number of used bins. \
You must wrap your algorithm description in curly braces {}. Then, implement a Python function named 'score'."
        self.prompt_func_name = "score"
        self.prompt_func_inputs = ['item', 'bins']
        self.prompt_func_outputs = ['scores']
        self.prompt_inout_inf = "'item' and 'bins' are the size of current item and the rest capacities of feasible bins, which are larger than the item size. \
The output named 'scores' is the scores for the bins for assignment. "
        # self.prompt_other_inf = "Note that 'item' is of type int, while 'bins' and 'scores' are both Numpy arrays. The novel function should be sufficiently complex in order to achieve better performance. It is important to ensure self-consistency."

#         # claude
#         self.prompt_other_inf = "Note that 'item' is of type int, while 'bins' and 'scores' are both Numpy arrays. The novel function should be sufficiently complex in order to achieve better performance. It is important to ensure self-consistency.\
# # Do not use markdown formatting. Do not use code blocks. Output the description and code directly without any formatting."
        
        # chatgpt
        self.prompt_other_inf = "Note that 'item' is of type int, while 'bins' and 'scores' are both Numpy arrays. \
The novel function should be sufficiently complex in order to achieve better performance. It is important to ensure self-consistency. \
Do not explain. Do not use markdown. Output Python code only, with no additional commentary."

#Include the following imports at the beginning of the code: 'import numpy as np', and 'from numba import jit'. Place '@jit(nopython=True)' just above the 'priority' function definition."

    def get_task(self):
        return self.prompt_task
    
    def get_func_name(self):
        return self.prompt_func_name
    
    def get_func_inputs(self):
        return self.prompt_func_inputs
    
    def get_func_outputs(self):
        return self.prompt_func_outputs
    
    def get_inout_inf(self):
        return self.prompt_inout_inf

    def get_other_inf(self):
        return self.prompt_other_inf

