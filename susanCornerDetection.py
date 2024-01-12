import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

#definesc kernelul care va fi folosit pentru plasarea unei masti circulare in jurul pixelului studiat(7x7)
def kernel_mask():
    mask=np.ones((7,7))
    mask[0,0]=0
    mask[0,1]=0
    mask[0,5]=0
    mask[0,6]=0
    mask[1,0]=0
    mask[1,6]=0
    mask[5,0]=0
    mask[5,6]=0
    mask[6,0]=0
    mask[6,1]=0
    mask[6,5]=0
    mask[6,6]=0
    return mask

#functie de afisare a imaginii
def plot_image(image,title):
	plt.figure()
	plt.title(title)
	plt.imshow(image,cmap = 'gray')
	plt.show()

#functie pentru afisarea imaginilor in paralel
def plot_images_side_by_side(img1, title1, img2, title2):
    plt.figure(figsize=(10, 5))
    # Subplot pentru prima imagine
    plt.subplot(1, 2, 1)
    plt.imshow(img1, cmap='gray')
    plt.title(title1)
    # Subplot pentru a doua imagine
    plt.subplot(1, 2, 2)
    plt.imshow(img2, cmap='gray')
    plt.title(title2)

    plt.show()


#functie de detectare a colturilor folosind algoritmul SUSAN
def susan_corner_detection(img):
    #initializare
    img = img.astype(np.float64) #convertesc imaginea in float pentru a avea o precizie mai mare
    g=37/2 #g este setat la jumatate din dimensionarea mastii
    circularMask = kernel_mask() #definesc masca circulara
    #print (circularMask) #afisez masca circulara pentru verificare
    output=np.zeros(img.shape) #definesc matricea de output
    val=np.ones((7,7)) #definesc o matrice de 7x7 cu valori de 1

    #parcurgerea imaginii pixel cu pixel
    for i in range(3,img.shape[0]-3):
        for j in range(3,img.shape[1]-3):  #se parcurge imaginea excluzand o margine de 3 pixeli pentru a asigura aplicarea mastii in jurul fiecarui pixel
            ir=np.array(img[i-3:i+4, j-3:j+4]) #definesc o matrice de 7x7 in jurul fiecarui pixel
            ir =  ir[circularMask==1] #calculez intensitatea pixelilor din matricea de 7x7 care se afla in jurul pixelului curent
            ir0 = img[i,j] #definesc intensitatea pixelului curent
            a=np.sum(np.exp(-((ir-ir0)/10)**6)) #aplic formula de calcul a lui c(r,r0) - formula (2) din curs 4 -pagina 13 – prag de intensitate (cantitativ)
            if a<=g:  #daca suma este mai mica sau egala cu g, se atribuie unei noi variabile a diferența g - a, in caz contrar, a este setat la 0
                a=g-a
            else:
                a=0
            output[i,j]=a #atribui valoarea lui a pixelului curent din matricea de output
    return output


#filtrarea raspunsului initial 
def filter_response(output):
    #1.Calcularea centrului masei al USAN-ului (Cmu) și compararea distanței D la centrul nucleului (Cn)
    Cmu = np.zeros_like(output)
    Cn = np.zeros_like(output)
    a = 3 #poate fi ajustat în functie de situatie
    for i in range(3, output.shape[0]-3):
        for j in range(3, output.shape[1]-3):
            usan = output[i-3:i+4, j-3:j+4]
            usan_mask = kernel_mask()
            #calculul centrului de masa al USAN
            Cmu[i, j] = np.sum(usan * usan_mask) / np.sum(usan_mask)
            #calculul distantei D de la centrul de masa la centrul nucleului
            Cn[i, j] = np.sqrt((i - (i-3 + 3))**2 + (j - (j-3 + 3))**2)

    #2.Continuitatea USAN-ului
    continuity_threshold = 0.5  #poate fi ajustat in functie de situatie
    for i in range(3, output.shape[0]-3):
        for j in range(3, output.shape[1]-3):
            if Cn[i, j] == np.max(Cn[i-3:i+4, j-3:j+4]):
                #daca distanta D este maxima, centrul de masa este mai departe de centrul nucleului (colt real)
                continue

            #verific continuitatea USAN-ului
            continuity_mask = (Cn[i-3:i+4, j-3:j+4] - Cmu[i, j]) <= continuity_threshold
            if np.all(continuity_mask):
                #toti pixelii de pe segmentul D fac parte din USAN
                output[i, j] = 0  #elimin coltul nevalid
    return output


#suprimarea non-maximelor
def non_max_suppression(output, neighborhood_size=3):
    h, w = output.shape
    half_size = neighborhood_size // 2
    for i in range(half_size, h-half_size):
        for j in range(half_size, w-half_size):
            local_max = np.max(output[i-half_size:i+half_size+1, j-half_size:j+half_size+1])
            if output[i, j] < local_max:
                output[i, j] = 0  #suprim non-maximul
    return output


#primul test - rezultatul initial
img1 = cv.imread("susan_input1.png", 0)
output1 = susan_corner_detection(img1)
finaloutput1 = cv.cvtColor(img1, cv.COLOR_GRAY2RGB)
finaloutput1[output1 != 0] = [255, 0, 0]

#primul test - rezultatul filtrat si suprimat non-maxima
img2 = cv.imread("susan_input1.png", 0)
output2 = susan_corner_detection(img2)
output2_filtered = filter_response(output2)
output2_suppressed = non_max_suppression(output2_filtered)
finaloutput2 = cv.cvtColor(img2, cv.COLOR_GRAY2RGB)
finaloutput2[output2_suppressed != 0] = [255, 0, 0]

#afisarea rezultatelor
plot_images_side_by_side(finaloutput1, "Rezultat Test 1 - initial", finaloutput2, "Rezultat Test 1 - filtrat si suprimat non-maxima")

#Al doilea test
img1 = cv.imread("susan_input3.png", 0)
output3 = susan_corner_detection(img1)
finaloutput3 = cv.cvtColor(img1, cv.COLOR_GRAY2RGB)
finaloutput3[output3 != 0] = [255, 0, 0]

#al doilea test - rezultatul filtrat si suprimat non-maxima
img2 = cv.imread("susan_input3.png", 0)
output4 = susan_corner_detection(img2)
output4_filtered = filter_response(output4)
output4_suppressed = non_max_suppression(output4_filtered)
finaloutput4 = cv.cvtColor(img2, cv.COLOR_GRAY2RGB)
finaloutput4[output4_suppressed != 0] = [255, 0, 0]

#afisarea rezultatelor
plot_images_side_by_side(finaloutput3, "Rezultat Test 2 - initial", finaloutput4, "Rezultat Test 2 - filtrat si suprimat non-maxima")



