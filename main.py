import os

if __name__ == '__main__':
    print("Goodbye World :/")

    # filepath = ".\\images"
    for file in os.listdir(".\\images"):
        print(file)
