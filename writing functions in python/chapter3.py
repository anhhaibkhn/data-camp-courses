""" Chapter 3: More on Decorators
Now that you understand how decorators work under the hood, this chapter gives you 
a bunch of real-world examples of when and how you would write decorators in your own code. 
You will also learn advanced decorator concepts like how to preserve the metadata of
your decorated functions and how to write decorators that take arguments."""

import time, os

class Decorators():

    def __init__(self):
        pass

    def exercise_1(self,mean, std, minimum, maximum, load_data, get_user_input):
        """ Building a command line data app """
        # Add the missing function references to the function map
        function_map = {
        'mean': mean,
        'std': std,
        'minimum': minimum,
        'maximum': maximum
        }

        data = load_data()
        print(data)

        func_name = get_user_input()

        # Call the chosen function and pass "data" as an argument
        function_map[func_name](data)
        '''
            <script.py> output:
            height  weight
            0    72.1     198
            1    69.8     204
            2    63.2     164
            3    64.7     238
            Type a command: 
            > minimum
            height     63.2
            weight    164.0
            dtype: float64
        '''

    def exercise_2(self):
        """ Returning functions for a math game """

        def create_math_function(func_name):
            if func_name == 'add':
                def add(a, b):
                    return a + b
                return add
            elif func_name == 'subtract':
                # Define the subtract() function
                def subtract(a,b):
                    return a-b
                return subtract
            else:
                print("I don't know that one")
                
        add = create_math_function('add')
        print('5 + 2 = {}'.format(add(5, 2)))

        subtract = create_math_function('subtract')
        print('5 - 2 = {}'.format(subtract(5, 2)))

        
    





def main():
    chap3 = Decorators()
    chap3.exercise_1()


if __name__ == "__main__":
    main()
