class GetPrompts():
    def __init__(self):
        self.prompt_task = "I need help designing a novel score function that scores a set of bins to assign an item. \
In each step, the item will be assigned to the bin with the maximum score. If the rest capacity of a bin equals the maximum capacity, it will not be used. \
The final goal is to minimize the number of used bins. You must wrap your algorithm description in curly braces {}. \
Then, implement it in Python as a function named 'score'."

        self.prompt_func_name = "score"
        self.prompt_func_inputs = ['item', 'bins']
        self.prompt_func_outputs = ['scores']

        self.prompt_inout_inf = "'item' is of type int, while 'bins' and 'scores' are Numpy arrays. \
The function should return an array of scores for assigning the item to each bin."

        self.prompt_other_inf = "Do not explain anything. Do not include markdown or code block. \
Only output the description in curly braces {}, followed by the function code in plain Python."

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
