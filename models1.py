def parameterenabling(self):
    i = self.dlg.Method_comboBox.currentIndex()
    print(i)
    if i == 0:
        self.dlg.LearningRate_Filed.setEnabled(True)
        self.dlg.Iteration_comboBox.setEnabled(True)
        self.dlg.HiddenLayer_comboBox.setEnabled(True)
    if i == 1:
        self.dlg.LearningRate_Filed.setEnabled(True)
        self.dlg.Iteration_comboBox.setEnabled(True)
        self.dlg.HiddenLayer_comboBox.setEnabled(True)
    if i == 2:
        self.dlg.LearningRate_Filed.setEnabled(True)
        self.dlg.Iteration_comboBox.setEnabled(True)
        self.dlg.HiddenLayer_comboBox.setEnabled(True)