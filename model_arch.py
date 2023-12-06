#Model Architecture
model = tf.keras.models.Sequential([ 
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)), 
    layers.MaxPooling2D(2, 2), 
    layers.Conv2D(64, (3, 3), activation='relu'), 
    layers.MaxPooling2D(2, 2), 
    layers.Conv2D(64, (3, 3), activation='relu'), 
    layers.MaxPooling2D(2, 2), 
    layers.Conv2D(64, (3, 3), activation='relu'), 
    layers.MaxPooling2D(2, 2), 
  
    layers.Flatten(), 
    layers.Dense(512, activation='relu'), 
    layers.BatchNormalization(), 
    layers.Dense(512, activation='relu'), 
    layers.Dropout(0.1), 
    layers.BatchNormalization(), 
    layers.Dense(512, activation='relu'), 
    layers.Dropout(0.2), 
    layers.BatchNormalization(), 
    layers.Dense(1, activation='sigmoid') 
])

# print the summary of the model’s architecture
model.summary()
keras.utils.plot_model( 
    model, 
    show_shapes=True, 
    show_dtype=True, 
    show_layer_activations=True
)
model.compile( 
    loss='binary_crossentropy', 
    optimizer='adam', 
    metrics=['accuracy'] 
)

#Model Training
history = model.fit(train_datagen, 
          epochs=10, 
          validation_data=test_datagen)
history_df = pd.DataFrame(history.history) 
history_df.loc[:, ['loss', 'val_loss']].plot() 
history_df.loc[:, ['accuracy', 'val_accuracy']].plot() 
plt.show() 

