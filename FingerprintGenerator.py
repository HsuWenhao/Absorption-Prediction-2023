from enum import Enum
from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit import Chem
from rdkit.Chem import MACCSkeys, AllChem
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw
from rdkit.Chem.AtomPairs import Pairs, Torsions
from rdkit.Chem import rdMolDescriptors
from rdkit.Avalon import pyAvalonTools

class Fingerprints(Enum):
    RDKit = 1
    RDKit_linear = 5
    Morgan_Circular = 9
    Morgan_Circular_Feature = 13
    Atom_Pairs = 17
    Topological_Torsions = 18
    MACCS_keys = 19
    AvalonFP2048 = 21
    ALL = 1337

#8000 bit Fingerprints wegen convergenz ersetzt

class FingerprintGenerator(object):
    def __init__(self):
        # print("FingerprintGenerator geladen")
        pass
    def generateArrofFingerprints(self, data):
        if data == 'nan':
            return self.__generateFingerprints_ALL('NA')
        # data = Chem.MolFromSmiles(data)
        return self.__generateFingerprints_ALL(data)

    def __generateFingerprints_ALL(self,data):
        length = 2048
        ret_arr = {}
        ret_arr['RDKit_2']=self.__generateFingerprints_RDKit(data,2,length)
        ret_arr['RDKit_4']=self.__generateFingerprints_RDKit(data,4,length)
        ret_arr['RDKit_6']=self.__generateFingerprints_RDKit(data,6,length)
        ret_arr['RDKit_8']=self.__generateFingerprints_RDKit(data,8,length)

        ret_arr['RDKit_linear_2']=self.__generateFingerprints_RDKitlinear(data,2,length)
        ret_arr['RDKit_linear_4']=self.__generateFingerprints_RDKitlinear(data,4,length)
        ret_arr['RDKit_linear_6']=self.__generateFingerprints_RDKitlinear(data,6,length)
        ret_arr['RDKit_linear_8']=self.__generateFingerprints_RDKitlinear(data,8,length)

        ret_arr['MorganCircle_0']=self.__generateFingerprints_Morgan_Circular(data,0,length)
        ret_arr['MorganCircle_2']=self.__generateFingerprints_Morgan_Circular(data,2,length)
        ret_arr['MorganCircle_4']=self.__generateFingerprints_Morgan_Circular(data,4,length)
        ret_arr['MorganCircle_6']=self.__generateFingerprints_Morgan_Circular(data,6,length)  

        ret_arr['MorganCircle_feature_0']=self.__generateFingerprints_Morgan_Circular_Feature(data,0,length)
        ret_arr['MorganCircle_feature_2']=self.__generateFingerprints_Morgan_Circular_Feature(data,2,length)
        ret_arr['MorganCircle_feature_4']=self.__generateFingerprints_Morgan_Circular_Feature(data,4,length)
        ret_arr['MorganCircle_feature_6']=self.__generateFingerprints_Morgan_Circular_Feature(data,6,length)

        ret_arr['Layerd_2']=self.__generateFingerprints_LayerdFingerprint(data, 2,length)
        ret_arr['Layerd_4']=self.__generateFingerprints_LayerdFingerprint(data, 4,length)
        ret_arr['Layerd_6']=self.__generateFingerprints_LayerdFingerprint(data, 6,length)
        ret_arr['Layerd_8']=self.__generateFingerprints_LayerdFingerprint(data, 8,length)

        ret_arr['Avalon']=self.__generateFingerprints_Avalon(data,length)

        ret_arr['MACCS']=self.__generateFingerprints_MACCS_keys(data)
        ret_arr['AtomPairs']=self.__generateFingerprints_Atom_Pairs(data,length)
        ret_arr['TopologicalTorsions']=self.__generateFingerprints_Topological_Torsions(data,length) 
        return ret_arr

    def __getEmptyBitVector(self, length):
        bitvector = ExplicitBitVect(length)
        return bitvector

    def __generateFingerprints_RDKit(self,data,maxPath,length):
        if data == 'NA':
            return self.__getEmptyBitVector(length)
        fp = Chem.RDKFingerprint(mol=data, maxPath=maxPath, fpSize=length)
        return fp
    
    def __generateFingerprints_RDKitlinear(self,data,maxPath, length):
        if data == 'NA':
            return self.__getEmptyBitVector(length)
        fp = Chem.RDKFingerprint(mol=data, maxPath=maxPath, branchedPaths=False, fpSize=length)
        return fp

    def __generateFingerprints_Atom_Pairs(self,data, length):
        if data == 'NA':
            return self.__getEmptyBitVector(length)
        return rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(data, nBits=length)

    def __generateFingerprints_Topological_Torsions(self,data, length):
        if data == 'NA':
            return self.__getEmptyBitVector(length)
        return rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(data, nBits=length)

    def __generateFingerprints_MACCS_keys(self,data):
        if data == 'NA':
            return self.__getEmptyBitVector(167)
        return MACCSkeys.GenMACCSKeys(data)

    def __generateFingerprints_Morgan_Circular(self,data,r, length):
        if data == 'NA':
            return self.__getEmptyBitVector(length)
        return AllChem.GetMorganFingerprintAsBitVect(data, r, nBits=length)

    def __generateFingerprints_Morgan_Circular_Feature(self,data, r, length):
        if data == 'NA':
            return self.__getEmptyBitVector(length)
        return AllChem.GetMorganFingerprintAsBitVect(data, r, useFeatures=True, nBits=length)

    def __generateFingerprints_Avalon(self,data, bitlength):
        if data == 'NA':
            return self.__getEmptyBitVector(bitlength)
        return pyAvalonTools.GetAvalonFP(data, nBits=bitlength)

    def __generateFingerprints_LayerdFingerprint(self, data, r, bitlength):
        if data == 'NA':
            return self.__getEmptyBitVector(bitlength)
        return Chem.LayeredFingerprint(data, maxPath=r, fpSize=bitlength)