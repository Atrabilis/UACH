import numpy as np

def bwareaopen(imgBin, small_area):

    num_fil = imgBin.shape[0]
    num_col = imgBin.shape[1]

    imgLabel = np.zeros((num_fil,num_col), dtype = int)
    imgBinFiltered = imgBin.copy()

    object_index = 0
    col_actual = 0
    area = np.array([])

    while col_actual <= num_col:
        object = np.zeros((1, 2), int)
        temp = np.zeros((1, 2), int)
        
        flag = False
        for col in range (col_actual, num_col):
            for fil in range (0, num_fil):
            
                if imgBin[fil,col] == 0:  
                        object_index += 1
                        object[0,:] = [fil,col]
                        imgBin[fil,col] = 1
                        flag = True
                        col_actual = col
                        break      
            if flag == True: break
                    
        if flag == False: col_actual = num_col + 1
        k = 0

        while k < len(object) and flag == True:
            if object[k,0] > 0 and object[k,1] > 0: 
                if imgBin[object[k,0] - 1, object[k,1] - 1] == 0: 
                    temp[0,:] = [object[k,0] - 1, object[k,1] - 1]
                    object = np.append(object, temp, axis = 0)
                    imgBin[object[k,0] - 1, object[k,1] - 1] = 1
            if object[k,0] > 0: 
                if imgBin[object[k,0] - 1, object[k,1]    ] == 0: 
                    temp[0,:] = [object[k,0] - 1, object[k,1] ]
                    object = np.append(object, temp, axis = 0)
                    imgBin[object[k,0] - 1, object[k,1]    ] = 1
            if object[k,0] > 0 and object[k,1] < num_col - 1: 
                if imgBin[object[k,0] - 1, object[k,1] + 1] == 0: 
                    temp[0,:] = [object[k,0] - 1, object[k,1] + 1]
                    object = np.append(object, temp, axis = 0)
                    imgBin[object[k,0] - 1, object[k,1] + 1] = 1
            if object[k,1] > 0: 
                if imgBin[object[k,0]    , object[k,1] - 1] == 0: 
                    temp[0,:] = [object[k,0] , object[k,1] - 1]
                    object = np.append(object, temp, axis = 0)
                    imgBin[object[k,0]    , object[k,1] - 1] = 1
            if object[k,1] < num_col - 1: 
                if imgBin[object[k,0]    , object[k,1] + 1] == 0: 
                    temp[0,:] = [object[k,0] , object[k,1] + 1]
                    object = np.append(object, temp, axis = 0)
                    imgBin[object[k,0]    , object[k,1] + 1] = 1
            if object[k,0] < num_fil - 1 and object[k,1] > 0:    
                if imgBin[object[k,0] + 1, object[k,1] - 1] == 0: 
                    temp[0,:] = [object[k,0] + 1, object[k,1] - 1]
                    object = np.append(object, temp, axis = 0)
                    imgBin[object[k,0] + 1, object[k,1] - 1] = 1
            if object[k,0] < num_fil - 1: 
                if imgBin[object[k,0] + 1, object[k,1]    ] == 0: 
                    temp[0,:] = [object[k,0] + 1, object[k,1]]
                    object = np.append(object, temp, axis = 0)
                    imgBin[object[k,0] + 1, object[k,1]    ] = 1
            if object[k,0] < num_fil - 1 and object[k,1] < num_col - 1: 
                if imgBin[object[k,0] + 1, object[k,1] + 1] == 0: 
                    temp[0,:] = [object[k,0] + 1, object[k,1] + 1]
                    object = np.append(object, temp, axis = 0)
                    imgBin[object[k,0] + 1, object[k,1] + 1] = 1
            k += 1
        if flag == True:
            area = np.append(area, 0)
            for i in range(len(object)):
                imgLabel[object[i,0], object[i,1]] = object_index
                area[object_index - 1] += 1

    for fil in range (num_fil):
        for col in range (num_col):
            if area[imgLabel[fil,col] - 1] < small_area: imgBinFiltered[fil,col] = 1

    return imgBinFiltered



def bwlabel(imgBin):

    num_fil = imgBin.shape[0]
    num_col = imgBin.shape[1]

    imgLabel = np.zeros((num_fil,num_col), dtype = int)

    object_index = 0
    col_actual = 0

    while col_actual <= num_col:
        object = np.zeros((1, 2), int)
        temp = np.zeros((1, 2), int)
        
        flag = False
        for col in range (col_actual, num_col):
            for fil in range (0, num_fil):
            
                if imgBin[fil,col] == 0:  
                        object_index += 1
                        object[0,:] = [fil,col]
                        imgBin[fil,col] = 1
                        flag = True
                        col_actual = col
                        break      
            if flag == True: break
                    
        if flag == False: col_actual = num_col + 1
        k = 0

        while k < len(object) and flag == True:
            if object[k,0] > 0 and object[k,1] > 0: 
                if imgBin[object[k,0] - 1, object[k,1] - 1] == 0: 
                    temp[0,:] = [object[k,0] - 1, object[k,1] - 1]
                    object = np.append(object, temp, axis = 0)
                    imgBin[object[k,0] - 1, object[k,1] - 1] = 1
            if object[k,0] > 0: 
                if imgBin[object[k,0] - 1, object[k,1]    ] == 0: 
                    temp[0,:] = [object[k,0] - 1, object[k,1] ]
                    object = np.append(object, temp, axis = 0)
                    imgBin[object[k,0] - 1, object[k,1]    ] = 1
            if object[k,0] > 0 and object[k,1] < num_col - 1: 
                if imgBin[object[k,0] - 1, object[k,1] + 1] == 0: 
                    temp[0,:] = [object[k,0] - 1, object[k,1] + 1]
                    object = np.append(object, temp, axis = 0)
                    imgBin[object[k,0] - 1, object[k,1] + 1] = 1
            if object[k,1] > 0: 
                if imgBin[object[k,0]    , object[k,1] - 1] == 0: 
                    temp[0,:] = [object[k,0] , object[k,1] - 1]
                    object = np.append(object, temp, axis = 0)
                    imgBin[object[k,0]    , object[k,1] - 1] = 1
            if object[k,1] < num_col - 1: 
                if imgBin[object[k,0]    , object[k,1] + 1] == 0: 
                    temp[0,:] = [object[k,0] , object[k,1] + 1]
                    object = np.append(object, temp, axis = 0)
                    imgBin[object[k,0]    , object[k,1] + 1] = 1
            if object[k,0] < num_fil - 1 and object[k,1] > 0:    
                if imgBin[object[k,0] + 1, object[k,1] - 1] == 0: 
                    temp[0,:] = [object[k,0] + 1, object[k,1] - 1]
                    object = np.append(object, temp, axis = 0)
                    imgBin[object[k,0] + 1, object[k,1] - 1] = 1
            if object[k,0] < num_fil - 1: 
                if imgBin[object[k,0] + 1, object[k,1]    ] == 0: 
                    temp[0,:] = [object[k,0] + 1, object[k,1]]
                    object = np.append(object, temp, axis = 0)
                    imgBin[object[k,0] + 1, object[k,1]    ] = 1
            if object[k,0] < num_fil - 1 and object[k,1] < num_col - 1: 
                if imgBin[object[k,0] + 1, object[k,1] + 1] == 0: 
                    temp[0,:] = [object[k,0] + 1, object[k,1] + 1]
                    object = np.append(object, temp, axis = 0)
                    imgBin[object[k,0] + 1, object[k,1] + 1] = 1
            k += 1
        if flag == True:
            for i in range(len(object)):
                imgLabel[object[i,0], object[i,1]] = object_index

    return imgLabel