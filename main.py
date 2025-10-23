from model import *
from data import *
from tensorflow.keras.callbacks import EarlyStopping

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#settings
batch_size = 4
total_images = 457
validation_percentage = 0.2 #20%
epochs = 200

#we assume that when creating the data set we similarly round up the training set and round down the validation set
validation_set_size = int(total_images * validation_percentage)
training_set_size = total_images - validation_set_size

#resulting values MUST be integers
steps_per_epoch = training_set_size // batch_size 
validation_steps = validation_set_size // batch_size

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
myGene = trainGenerator(batch_size,'data/membrane/train','image','label',data_gen_args,save_to_dir = None)
validationGene = trainGenerator(batch_size,'data/membrane/validation','image','label',data_gen_args,save_to_dir=None)

earlystop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model = unet()
model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='val_loss',verbose=1, save_best_only=True)
model.fit(myGene,steps_per_epoch=steps_per_epoch,epochs=epochs,validation_data=validationGene, validation_steps=validation_steps, callbacks=[model_checkpoint, earlystop])

testGene = testGenerator("data/membrane/test")
results = model.predict(testGene,30,verbose=1)
saveResult("data/membrane/test",results)