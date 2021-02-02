""" Chapter 2: Context Managers

If you've ever seen the "with" keyword in Python and wondered what its deal was, 
then this is the chapter for you! Context managers are a convenient way to provide connections in Python and 
guarantee that those connections get cleaned up when you are done using them.
This chapter will show you how to use context managers, as well as how to write your own.

"""
import time, os

class Context_Managers():

    def __init__(self):
        pass

    def exercise_1(self):
        """ The timer() context manager """
        # Add a decorator that will make timer() a context manager
        @contextlib.contextmanager
        def timer():
            """Time the execution of a context block.
            Yields:
                None
            """
            start = time.time()
            # Send control back to the context block
            yield
            end = time.time()
            print('Elapsed: {:.2f}s'.format(end - start))

        with timer():
            print('This should take approximately 0.25 seconds')
            time.sleep(0.25)
            time.sleep(0.5)

    def exercise_2(self):
        """ A read-only open() context manager """
        @contextlib.contextmanager
        def open_read_only(filename):
            """Open a file in read-only mode.

            Args:
                filename (str): The location of the file to read

            Yields:
                file object
            """
            read_only_file = open(filename, mode='r')
            # Yield read_only_file so it can be assigned to my_file
            yield read_only_file
            # Close read_only_file
            read_only_file.close()

        with open_read_only('my_file.txt') as my_file:
            print(my_file.read())

    def exercise_3(self, stock):
        """ Scraping the NASDAQ """
        # Use the "stock('NVDA')" context manager
        # and assign the result to the variable "nvda"
        with stock('NVDA') as nvda:
            # Open "NVDA.txt" for writing as f_out
            with open("NVDA.txt", 'w') as f_out:
                for _ in range(10):
                    value = nvda.price()
                    print('Logging ${:.2f} for NVDA'.format(value))
                    f_out.write('{:.2f}\n'.format(value))


    def exercise_4(self):
        """ Changing the working directory """
        def in_dir(directory):
            """ Change current working directory to `directory`,
            allow the user to run some code, and change back.

            Args:
                directory (str): The path to a directory to work in.
            """
            current_dir = os.getcwd()
            os.chdir(directory)

            # Add code that lets you handle errors
            try:
                yield
            # Ensure the directory is reset,
            # whether there was an error or not
            finally:
                os.chdir(current_dir)



def main():
    chap2 = Context_Managers()
    chap2.exercise_1()


if __name__ == "__main__":
    main()
