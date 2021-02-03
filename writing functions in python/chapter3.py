""" Chapter 3:Decorators
Decorators are an extremely powerful concept in Python. They allow you to modify the behavior 
of a function without changing the code of the function itself. This chapter will lay the foundational concepts 
needed to thoroughly understand decorators (functions as objects, scope, and closures), 
and give you a good introduction into how decorators are used and defined. 
This deep dive into Python internals will set you up to be a superstar Pythonista.
"""

import time, os, random

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

    def exercise_3(self, load_and_plot_data):
        """ review your co-worker's code """

        def has_docstring(func):
            """Check to see if the function 
            `func` has a docstring.

            Args:
                func (callable): A function.

            Returns:
                bool
            """
            return func.__doc__ is not None
        
        # Call has_docstring() on the load_and_plot_data() function
        ok = has_docstring(load_and_plot_data)

        if not ok:
            print("load_and_plot_data() doesn't have a docstring!")
        else:
            print("load_and_plot_data() looks ok")

    def exercise_4(self):
        """ Modifying variables outside local scope """
        # instruction 1: add keyword to update call_count from inside the function 
        call_count = 0

        def my_function():
            # Use a keyword that lets us update call_count (ans: global)
            global call_count
            call_count += 1
            
            print("You've called my_function() {} times!".format(
                call_count
            ))
        
        for _ in range(20):
            my_function()
        
        # instruction 2: Add a keyword that lets us modify file_contents from inside save_contents().
        def read_files():
            file_contents = None
            
            def save_contents(filename):
                # Add a keyword that lets us modify file_contents (ans: nonlocal)
                nonlocal file_contents
                if file_contents is None:
                    file_contents = []
                with open(filename) as fin:
                    file_contents.append(fin.read())
                
            for filename in ['1984.txt', 'MobyDick.txt', 'CatsEye.txt']:
                save_contents(filename)
                
            return file_contents

        print('\n'.join(read_files()))
    
        # instruction 3: Add a keyword to done in check_is_done() so that wait_until_done() eventually stops looping.
        def wait_until_done():
            def check_is_done():
                # Add a keyword so that wait_until_done()  (ans: global done)
                # doesn't run forever
                global done
                if random.random() < 0.1:
                    done = True
                
            while not done:
                check_is_done()

            done = False
            wait_until_done()

            print('Work done? {}'.format(done))
        

    def exercise_5(self):
        """ Checking for closure """
        # 1 Use an attribute of the my_func() function to show that it has a closure that is not None
        def return_a_func(arg1, arg2):
            def new_func():
                print('arg1 was {}'.format(arg1))
                print('arg2 was {}'.format(arg2))
            return new_func
                
        my_func = return_a_func(2, 17)

        # Show that my_func()'s closure is not None
        print(my_func.__closure__ is not None)
        # Show that there are two variables in the closure
        print(len(my_func.__closure__) == 2)

        # Get the values of the variables in the closure
        closure_values = [
        my_func.__closure__[i].cell_contents for i in range(2)
        ]
        print(closure_values == [2, 17])

    def excercise_6(self):
        
            




def main():
    chap3 = Decorators()
    chap3.exercise_5()


if __name__ == "__main__":
    main()
