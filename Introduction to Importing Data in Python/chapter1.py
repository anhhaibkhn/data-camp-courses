'''
Chapter1 : Introduction and flat files
In this chapter, you'll learn how to import data into Python from all types
 of flat files, which are a simple and prevalent form of data storage.
 You've previously learned how to use NumPy and pandas
 you will learn how to use these packages to import flat files and customize your imports.
'''
# Import package
import numpy as np
import matplotlib.pyplot as plt

class Chapter_one_flat_flies():

    def __init__(self):
        pass


    def exercise_1(self, file_path=None):
        '''Importing entire text files
        '''
        # Open a file: file
        if file_path is not None:
            file = open(file_path, mode='r')

            # Print its first 500 chars
            print(file.read()[:500])

            # Check whether file is closed
            print(file.closed)

            # Close file
            file.close()

            # Check whether file is closed
            print(file.closed)

    def exercise_2(self):
        # Read & print the first 3 lines
        with open('moby_dick.txt') as file:
            print(file.readline())
            print(file.readline())
            print(file.readline())

    def exercise_3(self):
        ''' Using NumPy to import flat files '''

        # Assign filename to variable: file
        file = 'digits.csv'

        # Load file as array: digits
        digits = np.loadtxt(file, delimiter=',')

        # Print datatype of digits
        print(type(digits))

        # Select and reshape a row
        im = digits[21, 1:]
        im_sq = np.reshape(im, (28, 28))

        # Plot reshaped data (matplotlib.pyplot already loaded as plt)
        plt.imshow(im_sq, cmap='Greys', interpolation='nearest')
        plt.show()

    def exercise_4(self):
        ''' Customizing your NumPy import '''
        # Import numpy
        import numpy as np

        # Assign the filename: file
        file = 'digits_header.txt'

        # Load the data: data
        data = np.loadtxt(file, delimiter='\t', skiprows=1, usecols=[0,2])

        # Print data
        print(data)
    
    def exercise_5(self):
        '''Importing different datatypes'''
        # Assign filename: file
        file = 'seaslug.txt'

        # Import file: data
        data = np.loadtxt(file, delimiter='\t', dtype=str)

        # Print the first element of data
        print(data[0])

        # Import data as floats and skip the first row: data_float
        data_float = np.loadtxt(file, delimiter='\t', dtype=float, skiprows=1)

        # Print the 10th element of data_float
        print(data_float[9])

        # Plot a scatterplot of the data
        plt.scatter(data_float[:, 0], data_float[:, 1])
        plt.xlabel('time (min.)')
        plt.ylabel('percentage of larvae')
        plt.show()













def main():
    chap1 = Chapter_one_flat_flies()
    chap1.exercise_1('mobydick.txt')


if __name__ == "__main__":
    main()
