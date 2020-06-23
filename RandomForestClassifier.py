# -*- coding: utf-8 -*-
"""
/***************************************************************************
 RandomForestClassifier
                                 A QGIS plugin
 -
 Generated by Plugin Builder: http://g-sherman.github.io/Qgis-Plugin-Builder/
                              -------------------
        begin                : 2020-06-01
        git sha              : $Format:%H$
        copyright            : (C) 2020 by -
        email                : -
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""
from qgis.PyQt.QtCore import QSettings, QTranslator, QCoreApplication
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction, QMessageBox
from qgis.core import QgsVectorLayer, QgsProject

# Initialize Qt resources from file resources.py
from .resources import *
# Import the code for the dialog
from .RandomForestClassifier_dialog import RandomForestClassifierDialog
import gdal
import os
import subprocess
import os.path
import processing
import glob

class RandomForestClassifier:
    """QGIS Plugin Implementation."""

    def __init__(self, iface):
        """Constructor.

        :param iface: An interface instance that will be passed to this class
            which provides the hook by which you can manipulate the QGIS
            application at run time.
        :type iface: QgsInterface
        """
        # Save reference to the QGIS interface
        self.iface = iface
        # initialize plugin directory
        self.plugin_dir = os.path.dirname(__file__)
        # initialize locale
        locale = QSettings().value('locale/userLocale')[0:2]
        locale_path = os.path.join(
            self.plugin_dir,
            'i18n',
            'RandomForestClassifier_{}.qm'.format(locale))

        if os.path.exists(locale_path):
            self.translator = QTranslator()
            self.translator.load(locale_path)
            QCoreApplication.installTranslator(self.translator)

        # Declare instance attributes
        self.actions = []
        self.menu = self.tr(u'&RandomForestClassifier')

        # Check if plugin was started the first time in current QGIS session
        # Must be set in initGui() to survive plugin reloads
        self.first_start = None

    # noinspection PyMethodMayBeStatic
    def tr(self, message):
        """Get the translation for a string using Qt translation API.

        We implement this ourselves since we do not inherit QObject.

        :param message: String for translation.
        :type message: str, QString

        :returns: Translated version of message.
        :rtype: QString
        """
        # noinspection PyTypeChecker,PyArgumentList,PyCallByClass
        return QCoreApplication.translate('RandomForestClassifier', message)


    def add_action(
        self,
        icon_path,
        text,
        callback,
        enabled_flag=True,
        add_to_menu=True,
        add_to_toolbar=True,
        status_tip=None,
        whats_this=None,
        parent=None):
        """Add a toolbar icon to the toolbar.

        :param icon_path: Path to the icon for this action. Can be a resource
            path (e.g. ':/plugins/foo/bar.png') or a normal file system path.
        :type icon_path: str

        :param text: Text that should be shown in menu items for this action.
        :type text: str

        :param callback: Function to be called when the action is triggered.
        :type callback: function

        :param enabled_flag: A flag indicating if the action should be enabled
            by default. Defaults to True.
        :type enabled_flag: bool

        :param add_to_menu: Flag indicating whether the action should also
            be added to the menu. Defaults to True.
        :type add_to_menu: bool

        :param add_to_toolbar: Flag indicating whether the action should also
            be added to the toolbar. Defaults to True.
        :type add_to_toolbar: bool

        :param status_tip: Optional text to show in a popup when mouse pointer
            hovers over the action.
        :type status_tip: str

        :param parent: Parent widget for the new action. Defaults None.
        :type parent: QWidget

        :param whats_this: Optional text to show in the status bar when the
            mouse pointer hovers over the action.

        :returns: The action that was created. Note that the action is also
            added to self.actions list.
        :rtype: QAction
        """

        icon = QIcon(icon_path)
        action = QAction(icon, text, parent)
        action.triggered.connect(callback)
        action.setEnabled(enabled_flag)

        if status_tip is not None:
            action.setStatusTip(status_tip)

        if whats_this is not None:
            action.setWhatsThis(whats_this)

        if add_to_toolbar:
            # Adds plugin icon to Plugins toolbar
            self.iface.addToolBarIcon(action)

        if add_to_menu:
            self.iface.addPluginToMenu(
                self.menu,
                action)

        self.actions.append(action)

        return action

    def initGui(self):
        """Create the menu entries and toolbar icons inside the QGIS GUI."""

        icon_path = ':/plugins/RandomForestClassifier/icon.png'
        self.add_action(
            icon_path,
            text=self.tr(u'RandomForestClassifier'),
            callback=self.run,
            parent=self.iface.mainWindow())

        # will be set False in run()
        self.first_start = True


    def unload(self):
        """Removes the plugin menu item and icon from QGIS GUI."""
        for action in self.actions:
            self.iface.removePluginMenu(
                self.tr(u'&RandomForestClassifier'),
                action)
            self.iface.removeToolBarIcon(action)

    # def show_pop(self):
    #     msg = QMessa

    # def saysome(self):
    #     print("Button Clicked")

#------------------------------------------------------------------------------------------------------------
    # def array2raster(newRasterfn, geotrans, proj, array):

    #     cols = array.shape[1]
    #     rows = array.shape[0]
    #     driver = gdal.GetDriverByName('GTiff')
    #     outRaster = driver.Create(newRasterfn, cols, rows, 1, gdal.GDT_Byte)
    #     outRaster.SetGeoTransform(geotrans)
    #     outband = outRaster.GetRasterBand(1)
    #     outband.WriteArray(array)
    #     outRaster.SetProjection(proj)
    #     outband.FlushCache()


#------------------------------------------------TOOLS-----------------------------------------------------------
    
    def vector2raster(self, v_path):
        #v_path = self.dlg.Train_img_label.filePath()
        print(v_path)

        vlayer = QgsVectorLayer(v_path, "Vector layer", "ogr")
        if not vlayer.isValid():
            print("Layer failed to load!")
        else:
            QgsProject.instance().addMapLayer(vlayer)
            print("Done")
            
        ext = vlayer.extent()
        #for feature in layer.getFeatures():
        #   ext = feature.geometry().boundingBox()  # this line is the relevant part
        xmin = ext.xMinimum()
        xmax = ext.xMaximum()
        ymin = ext.yMinimum()
        ymax = ext.yMaximum()
        print(xmin,' ',xmax,' ',ymax,' ',ymin)

        extent_list = str(xmin) + ',' + str(xmax) + ',' + str(ymin) + ',' + str(ymax) + '[]'

        if not os.path.exists('QGIS_Temp'):
            os.makedirs('QGIS_Temp')

        r_path = 'QGIS_Temp/V2R_converted_temp.tif'

        processing.run("gdal:rasterize",
                       {'INPUT': v_path, 'FIELD':'id', 'UNITS': 1, 'WIDTH': 0.001,
                        'HEIGHT': 0.001, 'EXTENT': extent_list , 'NODATA': 0,
                        'OPTIONS': '', 'DATA_TYPE': 5, 'INIT': None, 'INVERT': False, 'EXTRA': '',
                        'OUTPUT': r_path})

        return r_path

    def resampler(self, REF_IMG, IMG_LABEL):

        from osgeo import gdal, gdalconst

        self.dlg.train_progressBar.setValue(5)

        # IMG_ADD = self.dlg.train_img_add.filePath()
        # IMG_LABEL_ADD = self.dlg.train_img_label.filePath()


        input1 = gdal.Open(IMG_LABEL, gdalconst.GA_ReadOnly)
        inputProj = input1.GetProjection()
        inputTrans = input1.GetGeoTransform()


        reference = gdal.Open(REF_IMG, gdalconst.GA_ReadOnly)
        referenceProj = reference.GetProjection()
        referenceTrans = reference.GetGeoTransform()
        bandreference = reference.GetRasterBand(1)    
        x = reference.RasterXSize 
        y = reference.RasterYSize

        RESAMPLED_IMG_LABEL = "Delhi_ROI_resampled2.tif" #Path to output file
        driver= gdal.GetDriverByName('GTiff')
        output = driver.Create(RESAMPLED_IMG_LABEL,x,y,1,bandreference.DataType)
        output.SetGeoTransform(referenceTrans)
        output.SetProjection(referenceProj)

        gdal.ReprojectImage(input1,output,inputProj,referenceProj,gdalconst.GRA_Bilinear)

        del output

        return RESAMPLED_IMG_LABEL

#-------------------------------------------TILES GENERATOR-------------------------------------------------
    def tiles(self):

        in_path = self.dlg.Tiles_Input.filePath()
        
        if(not in_path):
            print("Enter Input")
            QMessageBox.critical(self.dlg, 'No Input', 'Please select the image to be splitted.')
            return

        # in_path = 'C:/forest.tif'         
        out_path = 'C:/Users/Atishay/Desktop/Images/'
        file_name = 'Tile'
        
        if not os.path.exists(out_path):
            os.makedirs(out_path)
             
        tile_size_x = self.dlg.TileSizeX.value()
        tile_size_y = self.dlg.TileSizeY.value()
             
        if(not tile_size_x):
            tile_size_x = int(self.dlg.TileSizeX.defaultValue())
        else:
           tile_size_x = int(tile_size_x)
           
        if(not tile_size_y):
            tile_size_y = int(self.dlg.TileSizeX.defaultValue())
        else:
           tile_size_y = int(tile_size_y)


        ds = gdal.Open(in_path)

        complete = 0
        self.dlg.Tile_progressBar.setValue(complete)

        CREATE_NO_WINDOW = 0x08000000

        #for i in range(5):
        band = ds.GetRasterBand(1)
        xsize = band.XSize
        ysize = band.YSize
                 
        for i in range(0, xsize, tile_size_x):
            for j in range(0, ysize, tile_size_y):
                com_string = "gdal_translate -of GTIFF -srcwin " + str(i)+ ", " + str(j) + ", " + str(tile_size_x) + ", " + str(tile_size_y) + " " + str(in_path) + " " + str(out_path) + str(file_name)  + str(i) + "_" + str(j) + ".tif"
                subprocess.call(com_string, creationflags=CREATE_NO_WINDOW)
                complete = complete + 20
                self.dlg.Tile_progressBar.setValue(complete)


        label_path_v = self.dlg.Tiles_Input_2.filePath()
        label_path_r = self.vector2raster(label_path_v)
        label_path = self.resampler(in_path, label_path_r)

        out_path = 'C:/Users/Atishay/Desktop/Label/'
        file_name = 'Tile'
        
        if not os.path.exists(out_path):
            os.makedirs(out_path)
             
        tile_size_x = self.dlg.TileSizeX.value()
        tile_size_y = self.dlg.TileSizeY.value()
             
        if(not tile_size_x):
            tile_size_x = int(self.dlg.TileSizeX.defaultValue())
        else:
           tile_size_x = int(tile_size_x)
           
        if(not tile_size_y):
            tile_size_y = int(self.dlg.TileSizeX.defaultValue())
        else:
           tile_size_y = int(tile_size_y)


        ds = gdal.Open(label_path)

        complete = 0
        self.dlg.Tile_progressBar.setValue(complete)

        CREATE_NO_WINDOW = 0x08000000

        #for i in range(5):
        band = ds.GetRasterBand(1)
        xsize = band.XSize
        ysize = band.YSize
                 
        for i in range(0, xsize, tile_size_x):
            for j in range(0, ysize, tile_size_y):
                com_string = "gdal_translate -of GTIFF -srcwin " + str(i)+ ", " + str(j) + ", " + str(tile_size_x) + ", " + str(tile_size_y) + " " + str(label_path) + " " + str(out_path) + str(file_name)  + str(i) + "_" + str(j) + ".tif"
                subprocess.call(com_string, creationflags=CREATE_NO_WINDOW)
                complete = complete + 20
                self.dlg.Tile_progressBar.setValue(complete)

#-------------------------------------------CLASSIFIERS---------------------------------------------------------

    def randomForest(self):
    #import statements
    
        from osgeo import gdal, gdal_array
        import numpy as np

        try:
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import confusion_matrix,classification_report
        except ImportError:
            print("scikit-learn package not present\nInstalling...")
            import pip
            pip.main(["install", "--user", "scikit-learn"])

            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import confusion_matrix,classification_report

        try:
            import pickle
        except ImportError:
            import pip
            pip.main(["install", "--user", "pickle"])
            import pickle


        self.dlg.Clfr_progressBar.setValue(30)
            

        IMAGE_ADD = self.dlg.input_img_box.filePath()
        MODEL_ADD = self.dlg.input_img_box_2.filePath()
        #OUTPUT_ADD = fp3

        # #To open the image:
        img_ds = gdal.Open(IMAGE_ADD, gdal.GA_ReadOnly)

        img = np.zeros((img_ds.RasterYSize, img_ds.RasterXSize, img_ds.RasterCount),
                       gdal_array.GDALTypeCodeToNumericTypeCode(img_ds.GetRasterBand(1).DataType))

        for b in range(img.shape[2]):
            img[:, :, b] = img_ds.GetRasterBand(b + 1).ReadAsArray()

        print(img.shape)

        with open(MODEL_ADD, 'rb') as f:
            rf = pickle.load(f)

            img_as_array = img.reshape(-1, img.shape[2])
            print('Reshaped from {n} to {o}'.format(o=img.shape,
                                                    n=img_as_array.shape))
                                                    
        self.dlg.Clfr_progressBar.setValue(60)

        class_prediction = rf.predict(img_as_array)

        class_prediction = class_prediction.reshape(img[:, :, 0].shape)
        print(class_prediction.shape)                               


        geotrans = img_ds.GetGeoTransform()
        proj = img_ds.GetProjection()

        fname = 'Delhi_classified.tif'

        #fname = self.dlg.input_img_box_3.filePath()
        
        #self.array2raster(fname, geotrans, proj, class_prediction)

        #To convert array to raster
        cols = class_prediction.shape[1]
        rows = class_prediction.shape[0]
        driver = gdal.GetDriverByName('GTiff')
        outRaster = driver.Create(fname, cols, rows, 1, gdal.GDT_Byte)
        outRaster.SetGeoTransform(geotrans)
        outRaster.SetProjection(proj)
        outband = outRaster.GetRasterBand(1)
        outband.WriteArray(class_prediction)
        outband.FlushCache()

        self.dlg.Clfr_progressBar.setValue(100)

    def UNET_Classifier(self):
        
        import gdal
        import numpy as np
        import matplotlib.pyplot as plt

        try:
            import tensorflow as tf
            from keras.models import Model, load_model, model_from_json
            from keras.preprocessing.image import img_to_array, load_img

        except ImportError:
            print("tensorflow not present\nInstalling...")
            import pip
            pip.main(["install", "--user", "tensorflow"])

            import tensorflow as tf
            from keras.models import Model, load_model, model_from_json
            from keras.preprocessing.image import img_to_array, load_img

        try:
            from skimage.transform import resize

        except ImportError:
            print("scikit-image package not present\nInstalling...")
            import pip
            pip.main(["install", "--user", "scikit-image"])

            from skimage.transform import resize

        self.dlg.Clfr_progressBar.setValue(30)

        THRESHOLD = 0.5

        IMG_ADD = self.dlg.input_img_box.filePath()
        MODEL_ADD = self.dlg.input_img_box_2.filePath()
        H5_ADD = self.dlg.input_img_box_6.filePath()

        img = load_img(IMG_ADD)
        x_img = img_to_array(img)
        x_img = resize(x_img, (512, 512), mode = 'constant', preserve_range = True)
        x_img.shape

        self.dlg.Clfr_progressBar.setValue(50)

        with open(MODEL_ADD, 'r') as json_file:
            loaded_nnet = json_file.read()

        save_model = tf.keras.models.model_from_json(loaded_nnet)
        save_model.load_weights(H5_ADD)

        self.dlg.Clfr_progressBar.setValue(70)

        pred = save_model.predict(np.expand_dims(x_img, axis=0), verbose=1)[0, :, :, 0] # [ : : ] to squeeze the image dimension equiv to pred[0].squeeze()

        pred_bin = np.where(pred > THRESHOLD, 1, 0)

        self.dlg.Clfr_progressBar.setValue(90)

        plt.imshow(pred_bin)
        plt.show()

        self.dlg.Clfr_progressBar.setValue(100)

#-------------------------------------------TRAIN---------------------------------------------------------
    def rfc_train(self):

        from osgeo import gdal, gdal_array
        import numpy as np
        import pickle

        try:
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import confusion_matrix, classification_report
        except ImportError:
            print("scikit-learn package not present\nInstalling...")
            import pip
            pip.main(["install", "--user", "scikit-learn"])

            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import confusion_matrix, classification_report
        
        self.dlg.train_progressBar.setValue(20)
        
        IMG_ADD = self.dlg.train_img_add.filePath()
        IMG_VLABEL_ADD = self.dlg.train_img_label.filePath()

        IMG_LABEL_ADD = self.vector2raster(IMG_VLABEL_ADD)
        RESAMPLED_IMG_LABEL1 = self.resampler(IMG_ADD, IMG_LABEL_ADD)
        
        VALIDATION_SPLIT = self.dlg.train_valRatio.currentText()  # TO BE TAKEN FROM USER (Train Val Ratio)

        if (not VALIDATION_SPLIT):
            VALIDATION_SPLIT = 0.2
        else:
            VALIDATION_SPLIT = float(VALIDATION_SPLIT)

        if (not IMG_ADD):
            print("Enter Input")
            QMessageBox.critical(self.dlg, 'No Input', 'Please select the image to be Trained.')
            return
        if (not IMG_LABEL_ADD):
            print("Enter Input")
            QMessageBox.critical(self.dlg, 'No Input', 'Please input the Label.')
            return

        num_trees = self.dlg.train_RFC_trees.value()
        if (not num_trees):
            num_trees = int(self.dlg.train_RFC_trees.defaultValue())
        else:
            num_trees = int(num_trees)

        Depth = self.dlg.train_RFC_Depth.value()
        if (not Depth):
            Depth = None
        else:
            Depth = int(Depth)


        self.dlg.train_progressBar.setValue(40)


        img_ds = gdal.Open(IMG_ADD, gdal.GA_ReadOnly)

        img = np.zeros((img_ds.RasterYSize, img_ds.RasterXSize, img_ds.RasterCount),
                       gdal_array.GDALTypeCodeToNumericTypeCode(img_ds.GetRasterBand(1).DataType))

        for b in range(img.shape[2]):
            img[:, :, b] = img_ds.GetRasterBand(b + 1).ReadAsArray()

        print(img.shape)

        roi_ds = gdal.Open(RESAMPLED_IMG_LABEL1, gdal.GA_ReadOnly)

        roi = roi_ds.GetRasterBand(1).ReadAsArray().astype(np.uint8)

        print(roi.shape)

        self.dlg.train_progressBar.setValue(50)

        np.vstack(np.unique(roi, return_counts=True)).T

        features = img[roi > 0, :]
        labels = roi[roi > 0]
        print(features.shape)
        print(labels.shape)

        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=VALIDATION_SPLIT, random_state=0)

        print(X_train.shape)
        print(X_test.shape)

        self.dlg.train_progressBar.setValue(60)


        rf = RandomForestClassifier(n_estimators=num_trees, max_depth=None, n_jobs=-1,
                                    oob_score=True)  # n_estim = Trees, max_depth = Depth

        rf.fit(X_train, y_train)

        self.dlg.train_progressBar.setValue(80)

        
        filename = 'model_5band_plugged.sav'
        pickle.dump(rf, open(filename, 'wb'))

        self.dlg.train_progressBar.setValue(100)

#-------------------------------------------WORKFLOW---------------------------------------------------------
    def parameterenabling(self):
        i = self.dlg.Method_comboBox.currentIndex()
        print(i)
        if i == 0:
            self.dlg.LearningRate_Filed.setEnabled(True)
            self.dlg.Iteration_comboBox.setEnabled(True)
            #self.dlg.HiddenLayer_comboBox.setEnabled(True)
        if i == 1:
            self.dlg.LearningRate_Filed.setEnabled(False)
            self.dlg.Iteration_comboBox.setEnabled(False)
            #self.dlg.HiddenLayer_comboBox.setEnabled(True)
        if i == 2:
            self.dlg.LearningRate_Filed.setEnabled(True)
            self.dlg.Iteration_comboBox.setEnabled(True)
            #self.dlg.HiddenLayer_comboBox.setEnabled(True)

    def parameterenabling1(self):
        i = self.dlg.Classifier_comboBox.currentIndex()
        print(i)
        if i == 0:
            self.dlg.input_img_box_6.setEnabled(True)

        if i == 1:
            self.dlg.input_img_box_6.setEnabled(False)

        return i

    def Run_Classifier(self):
        i = self.dlg.Classifier_comboBox.currentIndex()
        print(i)
        if i == 0:
            self.UNET_Classifier()

        if i == 1:
            #self.print_check()
            self.randomForest()
        
#---------------------------------------------------------------------------------------------------------------
    
    def print_check(self):
        print("check")

    def merge_tiles(self):
        input_path = self.dlg.input_img_box.filePath()
        # output_path = self.dlg.input_img_box_3.filePath()
        output_path = 'C:/Users/HP/Desktop/Tile'            # Output location needs to be looked at
        tiles = list()
        for tile in glob.glob(input_path + "/" + "*.tif"):
            tiles.append(tile)

        processing.run("gdal:merge", {'INPUT': tiles, 'PCT': 'False',
                                      'SEPERATE': 'False', 'DATA_TYPE': 1, 'NODATA_INPUT': None, 'NODATA_OUTPUT': None,
                                      'OPTIONS': 'High Compression', 'EXTRA': 'None',
                                      'OUTPUT': str(output_path) + '/' + 'Merge' + '.tif'})
        print("All Done !!")

#--------------------------------------------DIALOG BOX------------------------------------------------------

    def run(self):
        """Run method that performs all the real work"""

        # Create the dialog with elements (after translation) and keep reference
        # Only create GUI ONCE in callback, so that it will only load when the plugin is started
        if self.first_start == True:
            self.first_start = False
            self.dlg = RandomForestClassifierDialog()
        
        #Progress bar
        self.dlg.Tile_progressBar.setValue(0)
        self.dlg.train_progressBar.setValue(0)
        self.dlg.Clfr_progressBar.setValue(0)

        #Temporary
        self.dlg.input_img_box_3.setEnabled(False)
        self.dlg.train_output.setEnabled(False)
        self.dlg.Output_Field_2.setEnabled(False)
        self.dlg.Tiles_Output.setEnabled(False)

        # show the dialog
        self.dlg.show()

    #----------------------------CLASSIFIER TAB----------------------------------------

        #self.dlg.testButton.clicked.connect(QMessageBox(self.iface.mainWindow(), 'Reverse Geocoding Error', 'Wrong Format!\nExiting...'))
        #print(IMG_ADD)
        
        self.dlg.Classifier_comboBox.activated.connect(self.parameterenabling1)
        self.dlg.RunClassifier_Button.clicked.connect(self.Run_Classifier)

        #self.dlg.RunClassifier_Button.clicked.connect(self.merge)
    #--------------------Tiles Generation TAB----------------------------------------------
        
        #Stores entries from the input boxes
        #tr_IMG_ADD = self.dlg.ImageInput_Field.filePath()

        #Calls the function to split image after the button is pressed
        self.dlg.Tiles_Button.clicked.connect(self.tiles)


    #---------------------- Train TAB (NN based)-------------------------------------------------

        #for enabeling parameter input widgets
        self.dlg.Method_comboBox.activated.connect(self.parameterenabling)
        self.dlg.Train_Button.clicked.connect(self.vector2raster)

    #----------------------------Train TAB-------------------------------------------------------

        self.dlg.train_button.clicked.connect(self.rfc_train)


    #--------------------------------------------------------------------------------------------

        # Run the dialog event loop
        result = self.dlg.exec_()
        # See if OK was pressed
        if result:
            # Do something useful here - delete the line containing pass and
            # substitute with your code.
            pass
