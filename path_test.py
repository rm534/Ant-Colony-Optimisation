from main import *

if __name__ == '__main__':
    optimiser = AntOptimiser(Data.data)
    # print(optimiser.data)
    x = 0
    while x <= 3:
        x+=1
        optimiser.create_path()
        optimiser._reset_path()
        optimiser.update_path_choices()
        print(optimiser.ants["ant1"]["path"])
        print(optimiser.ants["ant1"]["choices"])
