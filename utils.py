import numpy as np

def inverse_rigid_trans(Tr):
    """
    :param Tr: rigid body transform matrix (3x4 as [R|t])
    :return: Inverse rigid body transform matrix
    """
    inv_Tr = np.zeros_like(Tr) # 3x4
    inv_Tr[0:3,0:3] = np.transpose(Tr[0:3,0:3]) #inverse R mat
    inv_Tr[0:3,3] = np.dot(-np.transpose(Tr[0:3,0:3]), Tr[0:3,3]) #inverse shift
    return inv_Tr