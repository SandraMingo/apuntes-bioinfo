## Working with functions

# 1. Write a simple computer program that consists of a single function that takes as input the equilibrium concentration of the reactants of the following reaction, 2NO2 <-> N2O4 and gives as output the K_eq (with the correct units) of the following reaction. 
# Test the function with the following equilibrium concentrations: [NO_2] = 2; [N_2O_4] = 3

def equilibrium_constant(NO2, N2O4):
    '''Calculates the equilibrium constant of the reaction 2NO2 <-> N2O4
    Parameters:
    NO2 (int): Concentration of NO2 in Mol
    N2O4 (int): Concentration of N2O4 in Mol

    Returns:
    k_eq (float): Equilibrium constant of the reaction
    units (str): Units of the equilibrium constant
    '''
    k_eq = N2O4 / (NO2**2)
    units = "M^-1"
    return k_eq, units

NO2 = 2
N2O4 = 3
print(f'The equilibrium constant is {equilibrium_constant(NO2, N2O4)[0]} {equilibrium_constant(NO2, N2O4)[1]}')

# 2. Modify the previous function to calculate the K_eq with the correct units for the following reaction: Na2CO3 + CaCl2 <-> CaCO3 + 2NaCl with equilibrium concentrations: [Na2CO3] = 2; [CaCl2] = 0.5; [CaCO3] = 2; [NaCl] = 1.2

def equilibrium_constant2(reactives, products):
    '''Calculate equilibrium constant of a reaction
    Parameters:
    reactives (dict): Dictionary with the reactives of the reaction and their stoichiometry
    products (dict): Dictionary with the products of the reaction and their stoichiometry

    Returns:
    k_eq (float): Equilibrium constant of the reaction
    units (str): Units of the equilibrium
    '''
    k_eq = products[0] * products[1]**2 / (reactives[0] * reactives[1])
    units = "M^1"
    return k_eq, units

reactives= [2, 0.5]
products = [2, 1.2]
print(f'The equilibrium constant is {equilibrium_constant2(reactives, products)[0]} {equilibrium_constant2(reactives, products)[1]}')


#reactives = {"Na2CO3": (2, 1), "CaCl2": (0.5, 1)}
#products = {"CaCo3": (2, 1), "NaCl": (1.2, 2)}
