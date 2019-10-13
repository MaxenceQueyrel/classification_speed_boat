
from skimage import io
from skimage import filters
from skimage import morphology
import numpy as np

def MoyEqT(img, mask):
    """
    Prend en entrée un objet image et en retourne la moyenne colorimétrique et
    son écart-type  dans chaque canal de couleur pour une image RVB
    
    :param type: Image
    :param type : np.array().bool
    :rtype: ( List[0,0,0] , List[0,0,0] )
    """
    moy=[0,0,0]
    var=[0,0,0]

    moy = img[mask].mean(axis=0)
    var = img[mask].var(axis=0)
    var = np.sqrt(var)
    return(moy,var)
    
def seuillage(image):
    """
    Prend en entrée un objet image en niveaux de gris pour ressortir une image binaire après 
    ouverture (érosion puis dilatation)
    
    :param type: Image
    :rtype: Image
    """
    seuil = filters.threshold_otsu(image)
    image[ image > seuil ] = 1
    image[ image < seuil ] = 0
    image = morphology.erosion(image, selem=[[1,1,1],[1,1,1],[1,1,1],[1,1,1]])
    image = morphology.dilation(image)

    return image

def mask(img,xmin,xmax,ymin,ymax):
    mask = np.ones((img.shape[0],img.shape[1])).astype(bool)
    mask[xmin:xmax,ymin:ymax] = False
    return(mask)
    
def mask_global(img,xmin,xmax,ymin,ymax,marge=5):
    """
    Applique un masque qui cache le bateau et les zones situées trop loin du bateau, connaissant ses coordo
    
    :param type: Image
    :param type : int
    :param type : int
    :param type : int
    :param type : int
    :param type : int
    :rtype: np.array().bool
    
    """
    maxi = max(xmax-xmin,ymax-ymin)
    mask = np.zeros((img.shape[0],img.shape[1])).astype(bool)
    mask[int(xmin - 1 * maxi) : int( xmax + 1 * maxi) , int( ymin - 1 * maxi) : int( ymax + 1 * maxi) ] = True
    mask[int( xmin - marge ) : int( xmax + marge ) , int(  ymin - marge ) : int( ymax + marge) ] = False

    return(mask)

def dessine_noir(img,mask):
    """
    noircit les cases d'une image qui appartiennent au masque (purement visuel, pas d'utilité)
    """
    img[mask == 0] = 0
    return img    

def compteBlanc(binaire,aire_bateau):
    """
    Renvoie la proportion de pixels blancs par l'aire du bateau dans une image binaire
    
    :param type: Image
    :param type: int
    :rtype: float
    """
    return(binaire.sum()/aire_bateau)

def calcul_attributs(filename, marge=5): 
    """
    renvoie des attributs tels que la moyenne clorimétrique, la variance et la proportion de piels blanc après binérisation de l'image en entrée
    La marge permet de masquer intégralement le bateau
    
    :param type: string
    :param type: int
    :rtype: Tuple( List[], List[], int)
    """
    
    image = io.imread(filename)
    image_grey = io.imread(filename, as_grey=True)
    
    echelle_img = int(filename.split("_")[-1].split(".")[0])/2 #echelle_img peut valoir 100, 150 ou 200 px
    xmin = echelle_img    
    xmax = len(image)-echelle_img
    ymin = echelle_img
    ymax = len(image[0]) - echelle_img
    
    mask = mask_global(image,xmin,xmax,ymin,ymax)
    (moyRVB, EcartRVB) = MoyEqT(image,mask)
    binaire = seuillage(image_grey)
    proportion_blancs = compteBlanc(binaire[mask == 1], (xmax - xmin) * (ymax - ymin))
    
    return (moyRVB, EcartRVB, proportion_blancs)


if __name__=="__main__":
    cheminImage1="I/0b4e2ccc-dd3e-4a28-b3c9-e447742e6c3b/db0e8df6227fbe462e102a8024c2729b16881e0e/1aaa6005dd4a9b29e9be294a43d3ecc4.jpg"
    cheminBlock="boat_box_200.jpg"
    #print(calcul_attributs(cheminBlock, marge=5))
    
    img = io.imread("jeu_test/0aadde32-e057-11e9-a936-4e46ad189ba4_100.jpg")
    img_grey = io.imread(cheminBlock,as_grey=True)
    
    #io.imshow(seuillage(img_grey))
    #io.imshow(mask(img,100,115,100,115))
    #io.imshow(mask_global(img,100,115,100,115))
    #io.imshow(dessine_noir(img,mask_global(img,100,115,100,115)))
    #print(MoyEqT(img,95,mask(img,100,115,100,115)))
    #print(compteBlanc(seuillage(img_grey)))
    #print(seuillage(img_grey).sum())
    print(calcul_attributs(cheminBlock))
    
#    imtest=io.imread(cheminImage1, as_grey=True)
#    plt.figure(1)
#    io.imshow (imtest)
#    print(MoyEqT(cheminImage1))
#    plt.figure(2)
#    imgauss=ndimage.gaussian_filter(imtest, sigma=10)
#    io.imshow(imgauss)
    #print(MoyEqT2("boat_box_200.jpg"))
    #print(MoyEqT("boat_box_200.jpg"))

    #imblock = io.imread(cheminBlock,as_grey=True)
    
    #BetW = seuillage("boat_box_200.jpg")
    #io.imsave("BetW_200.jpg",BetW[0])
    
    #mask = mask("BetW_200.jpg")
    #io.imsave("BetW_mask_200.jpg",mask)
    #io.imshow(mask_img(seuillage(imblock)[0],200))
    
