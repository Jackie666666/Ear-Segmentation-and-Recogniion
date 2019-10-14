import sys 
import os
import logging
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
sys.path.append ("..")
from PIL import Image




def read_images ( path , sz = None ):
 c = 0
 X ,y = [] , []
 for dirname , dirnames , filenames in os.walk(path):
   for subdirname in dirnames:
     subject_path = os.path.join(dirname, subdirname)
     for filename in os.listdir(subject_path):
       try :
         im = Image.open(os.path.join(subject_path , filename ))
         
         im = im.convert ("L")
         # resize to given size (if given )
         if ( sz is not None ):
           im = im.resize(sz, Image.ANTIALIAS )
           
         X.append(np.asarray(im, dtype = np.uint8))
         tmp=subdirname.split("_")
         y.append(int(tmp[0]))
         
         #y.append(c)
       except IOError as e:
          errno, strerror = e.args
          print ("I/O error({0}): {1}".format(errno, strerror))
       except :
         print ("Unexpected error:", sys.exc_info()[0])
         raise
     c = c+1
     #print (X)
 return [X , y]





'''def read_images ( path , sz = None ):
 c = 0
 X ,y = [] , []
 for dirname , dirnames , filenames in os.walk(path):
   for subdirname in dirnames:
     subject_path = os.path.join(dirname, subdirname)
     for filename in os.listdir(subject_path):
       try :
         im = Image.open(os.path.join(subject_path , filename ))
         
         im = im.convert ("L")
         # resize to given size (if given )
         if ( sz is not None ):
           im = im.resize(sz, Image.ANTIALIAS )
           
         X.append(np.asarray(im, dtype = np.uint8))
         y.append(c)
       except IOError :
        
          print ("I/O error({0}): {1}".format(errno, strerror))
       except :
         print ("Unexpected error:", sys.exc_info()[0])
         raise
     c = c+1
     #print (X)
 return [X , y]
'''


def asRowMatrix (X) :
    if len (X) == 0:
        return np . array ([])
    mat = np.empty((0 , X [0]. size ), dtype = X [0].dtype)
    for row in X:
        mat = np.vstack((mat,np.asarray(row).reshape (1 , -1)))
    return mat


def asColumnMatrix (X):
    
  if len (X) == 0:
    return np . array ([])
  mat = np.empty((X[0]. size,0), dtype = X[0]. dtype)
  for col in X:
    mat = np.hstack((mat, np . asarray ( col ). reshape ( -1 ,1) ))  
  return mat



def project (W , X , mu = None ):
 if mu is None :
  return np . dot (X ,W)
 return np . dot (X - mu , W)



def reconstruct (W , Y , mu = None ) :
 if mu is None :
   return np . dot (Y ,W.T)
 return np . dot (Y , W .T) + mu



def normalize (X , low , high , dtype = None ):
   X = np . asarray (X)
   minX , maxX = np . min (X ) , np . max (X)
   X = X - float ( minX )
   X = X / float (( maxX - minX ) )
   X = X * ( high - low )
   X = X + low
   if dtype is None :
    return np . asarray (X)
   return np . asarray (X , dtype = dtype) 



def create_font(fontname ='Tahoma ', fontsize =10) :
   return { 'fontname': fontname , 'fontsize': fontsize }

def subplot ( title , images , rows , cols , sptitle =" subplot ", sptitles =[] , colormap = cm .
    gray , ticks_visible = True , filename = None ):
    fig = plt . figure ()
    # main title
    fig.text(.5 , .95 , title , horizontalalignment ='center')
    for i in range ( len ( images )):
      ax0 = fig . add_subplot ( rows , cols ,( i +1) )
      plt . setp ( ax0 . get_xticklabels () , visible = False )
      plt . setp ( ax0 . get_yticklabels () , visible = False )

      if len ( sptitles ) == len ( images ):
        plt . title ("%s #%s" % ( sptitle , str (sptitles [i ]) ) , create_font('Tahoma ',10) )
      else :
        plt . title ("%s #%d" % ( sptitle , (i +1) ) , create_font('Tahoma',10) )
      plt . imshow ( np . asarray ( images [ i ]) , cmap = colormap )
    if filename is None:
        plt . show ()
    else :
        fig.savefig(filename)


class AbstractDistance ( object ):
     def __init__ ( self , name ):
       self . _name = name
     def __call__ ( self ,p ,q):
       raise NotImplementedError (" Every AbstractDistance must implement the __call__method .")
     @property
     def name ( self ) :
       return self . _name


     def __repr__ ( self ) :
       return self . _name

class EuclideanDistance ( AbstractDistance ) :
     
    def __init__ ( self ) :
        AbstractDistance . __init__ ( self ," EuclideanDistance ")

    def __call__ ( self , p , q):
        p = np . asarray (p). flatten ()
        q = np . asarray (q). flatten ()
        return np . sqrt ( np . sum ( np . power (( p - q) ,2) ))


class CosineDistance ( AbstractDistance ):

    def __init__ ( self ) :
        AbstractDistance . __init__ ( self ," CosineDistance ")

    def __call__ ( self , p , q):
        p = np . asarray (p). flatten ()
        q = np . asarray (q). flatten ()
        return -np . dot ( p.T ,q ) / ( np . sqrt ( np . dot (p ,p.T )* np . dot (q ,q.T )))



class BaseModel ( object ):
  def __init__ ( self , X= None , y= None , dist_metric = EuclideanDistance () , num_components=0) :
      self . dist_metric = dist_metric
      self . num_components = 0
      self . projections = []
      self .W = []
      self . mu = []
      if (X is not None ) and (y is not None ):
        self.compute (X ,y )
  def compute ( self , X , y):
      raise NotImplementedError (" Every BaseModel must implement the compute method .")
  def predict ( self , X):
       count = 0
       no_classes =27
       
       minDist = np . finfo ('float') . max
       minClass = -1
       Q = project ( self .W , X. reshape (1 , -1) , self . mu )
       for i in range (no_classes):
         dist = 0 
         for j in range(len(self.projections)):
            if(self.y[j]==i):
              count = count+1
              dist = dist + self . dist_metric (self.projections[j], Q)
         dist = dist/count
         count=0
         #print("Distance for class " + str(i)+" = "+str(dist)) 
         if dist < minDist :
           minDist = dist
           minClass = i
       
       if(minDist>400):
         return 100
       else:
         print(minDist)
         return minClass





class EigenfacesModel ( BaseModel ):
    def __init__ ( self , X= None , y= None , dist_metric = EuclideanDistance () , num_components=74):
        
       super ( EigenfacesModel , self ). __init__ (X=X ,y=y , dist_metric = dist_metric ,
       num_components = num_components )
    def compute ( self , X , y):
       [D , self .W , self . mu ] = pca ( asRowMatrix ( X) ,y , self . num_components )
       #print(self . num_components )
# store labels
       self .y = y
# store projections
       for xi in X :
           self . projections . append ( project ( self .W , xi . reshape (1 , -1) , self . mu ))



def pca (X , y , num_components =0) :
  [n , d] = X . shape

  if ( num_components <= 0) or ( num_components >n) :
     num_components = n
  mu = X. mean ( axis =0)
  
  X = X - mu
  if n > d:
     C = np . dot (X.T ,X)
     [eigenvalues , eigenvectors ] = np . linalg . eigh (C)
  else :
     C = np . dot (X ,X .T)
     [eigenvalues , eigenvectors ] = np . linalg . eigh (C) 
     eigenvectors = np . dot (X .T , eigenvectors ) 
     for i in range (n):
       eigenvectors [: , i ] = eigenvectors [: , i ]/ np . linalg . norm ( eigenvectors [: , i ])
  idx = np.argsort(-eigenvalues)
  #print(eigenvectors[:,1])
  eigenvalues = eigenvalues [ idx ]
  eigenvectors = eigenvectors [: , idx ]
  eigenvalues = eigenvalues [0: num_components ]. copy ()
  eigenvectors = eigenvectors [: ,0: num_components ].copy()
  #print(np.shape(eigenvectors))
  return [ eigenvalues , eigenvectors , mu ]



def lda (X , y , num_components =0) :
    y = np . asarray (y)
    [n , d] = X . shape
    c = np . unique ( y)
    if ( num_components <= 0) or ( num_component >( len (c) -1) ):
        num_components = ( len (c) -1)
    meanTotal = X. mean ( axis =0)
    Sw = np . zeros ((d , d) , dtype = np . float32 )
    Sb = np . zeros ((d , d) , dtype = np . float32 )
    for i in c:
        Xi = X[ np . where (y == i) [0] ,:]
        meanClass = Xi . mean ( axis =0)
        Sw = Sw + np . dot (( Xi - meanClass ).T , ( Xi - meanClass ))
        Sb = Sb + n * np . dot (( meanClass - meanTotal ).T , ( meanClass - meanTotal ))
    eigenvalues , eigenvectors = np . linalg . eig ( np . linalg . inv ( Sw )* Sb )
    idx = np . argsort ( - eigenvalues . real )
    eigenvalues , eigenvectors = eigenvalues [ idx ] , eigenvectors [: , idx ]
    eigenvalues = np . array ( eigenvalues [0: num_components ]. real , dtype = np . float32 , copy =
                                                    True )
    eigenvectors = np . array ( eigenvectors [0: ,0: num_components ]. real , dtype = np . float32 ,
                                                        copy = True )
    return [ eigenvalues , eigenvectors ]



def fisherfaces (X ,y , num_components =0) :
    y = np . asarray (y)
    [n , d] = X . shape
    c = len ( np . unique (y ))
    [ eigenvalues_pca , eigenvectors_pca , mu_pca ] = pca (X , y , (n -c ))
    [ eigenvalues_lda , eigenvectors_lda ] = lda ( project ( eigenvectors_pca , X , mu_pca ) , y ,
                                                                                        num_components )
    eigenvectors = np . dot ( eigenvectors_pca , eigenvectors_lda )
    return [ eigenvalues_lda , eigenvectors , mu_pca ]








class FisherfacesModel ( BaseModel ):
    
    def __init__ ( self , X= None , y= None , dist_metric = EuclideanDistance () , num_components
                        =0) :
        super ( FisherfacesModel , self ). __init__ (X=X ,y=y , dist_metric = dist_metric ,
                    num_components = num_components )
    def compute ( self , X , y):
        [D , self .W , self . mu ] = fisherfaces ( asRowMatrix (X ) ,y , self . num_components )
        # store labels
        self .y = y
        # store projections
        for xi in X :
            self . projections . append ( project ( self .W , xi . reshape (1 , -1) , self . mu ))




def LdaAnalysis(CropImg) :
    

    [X , y] = read_images (r"C:\Users\kshit\Anaconda3\envs\tf_gpu\Lib\site-packages\tensorflow\models\research\deeplab\facerec-master\py\facerec\data\images",(150,150))
    [D , W , mu ] = fisherfaces ( asRowMatrix (X) , y)

    namelist =['AAKASH', 'KSHITIJ', 'MIHIR', 'SHREYAS', 'UJWAL', 'SHUBHANGI', 'AAYUSH', 'ADITYA', 'AKSHAY', 'AMAN', 'ANKIT', 'KUMAR', 'GANESH', 
               'HARSHAL', 'MANISH', 'NIKHIL','NISHANT','PRANAV','PRANCHIT','PRASAD','PUSHKAR','RAMTEKE','RAUT','SANKET','SAURABH','SHUBHAM','SITESH']



    model = FisherfacesModel ( X [:] , y [:])






    #[X1 , y1] = read_images(r"C:\Users\kshit\Desktop\facerec-master\py\facerec\data\imagestest",(150,150))
    CropImg = CropImg.convert("L")
    CropImg = CropImg.resize((150,150), Image.ANTIALIAS)
    
    CropImg = np.asarray(CropImg,dtype = np.uint8)
    
    predic = model.predict(CropImg)
    if predic == 100 :
        return "The Person is not in the database"
        
    else :
        return "The person in image is {}".format(namelist[predic])
        