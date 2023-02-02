import numpy as np 
import matplotlib.pyplot as plt
import cv2
#Izgara zemin oluşturur.
class CA_Grid:
    def __init__(self, height=200, width=400, initial_number_of_black_cell=1):
        self.height = height
        self.width = width
        self.initial_number_of_black_cell = initial_number_of_black_cell
        self.grid = None

    def __initialize(self,height, width, initial_number_of_black_cell):
        self.height = height
        self.width = width
        self.grid = None
        self.initial_number_of_black_cell = initial_number_of_black_cell

    def get_grid(self):

        if self.initial_number_of_black_cell==1:
            self.__single_black_cell_grid()
        else:
            self.__multiple_black_cell_grid()

        return self.grid
    #Yükseklik x genişlik ölçülerine göre bir matris oluşturur ve matrisin üst satırında ortadaki hücreye 1 atar.
    def __single_black_cell_grid(self):
        self.grid = np.zeros((self.height, self.width), dtype=np.int32)
        self.grid[0, int(self.width / 2)] = 1

    def __multiple_black_cell_grid(self):
        self.__single_black_cell_grid()
        n=self.initial_number_of_black_cell-1
        for i in range(n):
            random_col = np.random.randint(0, self.width)
            self.grid[0, random_col] = 1

class CA(CA_Grid):

    def __init__(self,grid_apparence="normal",**kwargs):
        super().__init__(**kwargs)
        self.grid_apparence=grid_apparence
        self.transform_vector=np.array([4,2,1])
        self.rule=None
        self.rule_binary=None

    def set_grid_parameters(self,
                            height,
                            width,
                            initial_number_of_black_cell=1,
                            grid_apparence="normal"):
        self.height = height
        self.width = width
        self.initial_number_of_black_cell = initial_number_of_black_cell
        self.grid = None
        self.grid_apparence=grid_apparence

    def __get_rule_binary(self):
        self.rule_binary = np.array([int(b) for b in np.binary_repr(self.rule, 8)], dtype=np.int8)

    def generate(self, rule):

        self.rule=rule
        self.get_grid()
        self.__get_rule_binary()
        for i in range(self.height-1):
            self.grid[i+1,:]=self.step(self.grid[i,:])
        self.grid[self.grid==1]=255
        if self.grid_apparence=="wolfram":
            self.grid=cv2.bitwise_not(self.grid)

        return self.grid

    def generate_all_ca(self):
        all_ca=list()
        for i in range(256):
            self.generate(i)
            all_ca.append(self.grid)

        return all_ca

    def __get_neighborhood_matrix(self, center):
        left=np.roll(center, 1)
        right=np.roll(center, -1)
        neighborhood_matrix=np.vstack((left, center, right)).astype(np.int8)

        return neighborhood_matrix

    def step(self, row):
        neighborhood_matrix=self.__get_neighborhood_matrix(center=row)
        rmts=self.transform_vector.dot(neighborhood_matrix)

        return self.rule_binary[7-rmts].astype(np.int8)

class Demonstrate_CA:
    def __init__(self):
        print("Demonstrate_CA object created")
    
    def show_rule(self, rule, step):
        step=step
        elementary_CA=CA(height=step, width=step*2, grid_apparence="wolfram")
        
        rule=rule
        generated_image=elementary_CA.generate(rule=rule)

        plt.figure(figsize=(15,15))
        plt.imshow(generated_image, cmap="gray")
        plt.xticks([])
        plt.yticks([])
        plt.title("Demonstration of Rule {} for {} Steps".format(rule, step))
        plt.show()
    
demonstrate_ca=Demonstrate_CA()
rules=[120]
step=100
for rule in rules:
    print("Demontration of rule {} for {} step".format(rule, step))
    demonstrate_ca.show_rule(rule=rule, step=step)