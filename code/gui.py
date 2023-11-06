from PyQt5.QtWidgets import *
from PyQt5 import uic

class MyGUI(QMainWindow):

    # global store acting as a single source of truth to share state across the app
    # and to execute global functionalities
    store = dict({})

    def __init__(self,UI_file_path):
        super(MyGUI,self).__init__()
        uic.loadUi(UI_file_path,self)

        
        self.store = dict({
            # get input fields' values
            "get" : {
                "learningRate" : lambda : float(self.learningRateSpinBox.text()),
                "epocs" : lambda : float(self.epocsSpinBox.text()),
                "mseThreshold" : lambda : float(self.epocsSpinBox.text()),
                "isAddBias" : lambda : bool(self.addBiasCheckBox.isChecked()),
                "isAdeline": lambda : bool(self.adalineRadioButton.isChecked()),
                "isPercetron": lambda : bool(self.perceptronRadioButton.isChecked()),

                # classification classes initials
                "isBSclasses": lambda : bool(self.BSradioButton.isChecked()),
                "isCSclasses": lambda : bool(self.CSradioButton.isChecked()),
                "isBCclasses": lambda : bool(self.BCradioButton.isChecked())
            }
            
        }) 

        #init functions
        self.show()

        #----------------------EVENT HANDLERES---------------------------
        #--------------Buttons
        self.startButton.clicked.connect(self.onStartButtonClick)



    # --------------------- Store Getters
    def getEpocs(self) :
        return self.store["get"]["epocs"]()

    def getLearningRate(self) :
        return self.store["get"]["learningRate"]()
    
    def getMSEThreshold(self) :
        return self.store["get"]["mseThreshold"]()

    def getMSEThreshold(self) :
        return self.store["get"]["mseThreshold"]()

    def getIsAdaline(self) :
        return self.store["get"]["isAdeline"]()

    def getIsBSclasses(self) :
        return self.store["get"]["isBSclasses"]()
    
    def getIsCSclasses(self) :
        return self.store["get"]["isCSclasses"]()
    
    def getIsBCclasses(self) :
        return self.store["get"]["isBCclasses"]()
    #---------------------------------------

    def onStartButtonClick(self) :

        # get input fields values
        learningRate = self.getLearningRate()
        epocs = self.getEpocs()
        mseThreshold = self.getMSEThreshold()
        isAdeline = self.getIsAdaline()
        isPercetron = not isAdeline
        isBS = self.getIsBSclasses()
        isCS = self.getIsCSclasses()
        isBC = self.getIsBCclasses()

        print(learningRate,epocs,mseThreshold,isAdeline,isPercetron,isBS,isCS,isBC)
        
        return