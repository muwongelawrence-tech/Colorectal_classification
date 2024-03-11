# Create an instance of your service class
bento_svc = VGG19ImageClassifier()

# Pack the model with your service
bento_svc.pack('model', 'vgg19_bs100_e10.h5')

# Save the BentoML service
saved_path = bento_svc.save()
print("Model saved at:", saved_path)