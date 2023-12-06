#checking random images
from keras.preprocessing import image 
  
#Input image 
test_image = image.load_img('1.jpg',target_size=(200,200)) 
  
#For show image 
plt.imshow(test_image) 
test_image = image.img_to_array(test_image) 
test_image = np.expand_dims(test_image,axis=0) 
  
# Result array 
result = model.predict(test_image) 
  
#Mapping result array with the main name list 
i=0
if(result>=0.5): 
  print("Dog") 
else: 
  print("Cat")
  
test_image = image.load_img('test/2.jpg', target_size=(200, 200)) 
  
# For show image 
plt.imshow(test_image) 
test_image = image.img_to_array(test_image) 
test_image = np.expand_dims(test_image, axis=0) 
  
# Result array 
result = model.predict(test_image) 
# Mapping result array with the main name list 
i = 0
if(result >= 0.5): 
    print("Dog") 
else: 
    print("Cat") 

