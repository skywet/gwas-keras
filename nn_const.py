'''
Author: Tianyu Wu
Date: Dec 10, 2019
'''
# Import keras models
from keras.layers import Input, Dense, Dropout
from keras.models import Model, Sequential
from keras.optimizers import Adam, SGD
from keras.callbacks import TensorBoard
# Import test packages
import numpy as np

class SdA(object):
    '''
    Define Stacked Denoising Autoencoder class for gene expression matrix
    All expression value should be processed to log2FC
    '''
    def __init__(self, expr_matrix):
        '''
        Define the autoencoder model
        '''
        # Define layer units size
        self.expr_matrix = expr_matrix
        self.coder_layer_units = [self.expr_matrix.size, 128, 64, 32, 16, 12]
        self.feature_layer_units = 3
        self.ae_models = []
        self.feature_layer = Dense(units=self.feature_layer_units)
        # Define autoencoder models for training
        for unit_num in self.coder_layer_units:
            encoder = Dense(unit_num, activation = 'sigmoid')
            decoder = Dense(unit_num, activation = 'sigmoid')
            self.ae_models.append([encoder, decoder])
        # Define final model for we to fill in the weights
        self.final_encoder = Sequential()
        self.final_decoder = Sequential()

    def train(self, lr=0.001, ep = 30):
        '''
        Train the submodels and the final model
        '''
        trained_encoder_layers = []
        trained_decoder_layers = []
        encoded_data = []
        # Train subencoders
        count = 1
        print('[INFO] Submodels construced.')
        print("[WARNING] Submodels train starting. Please DON'T touch the computer or end the process")
        for encoder, decoder in self.ae_models:
            model = Sequential()
            model.add(encoder)
            model.add(self.feature_layer)
            model.add(decoder)
            opt = Adam(learning_rate=lr)
            input_data = self.expr_matrix if len(encoded_data) == 0 else encoded_data[-1]
            print('[INFO] Submodel layer {} building complete, printing the params......'.format(count))
            model.compile(loss='mse', optimizer=opt)
            print(model.summary())
            print('[INFO] Training for submodels {} of 5'.format(count))
            model.fit(input_data, input_data, epochs = ep)
            trained_encoder_layers.append(encoder)
            trained_decoder_layers.append(decoder)
            encoded_data.append(encoder.predict(input_data))
            count += 1
        # Construct final model for fine-tuning
        print('[INFO] Training for submodels complete')
        print('[WARNING] Constructing the final SdA model')
        trained_decoder_layers.reverse()
        self.final_model = Sequential()
        for subencoder in trained_encoder_layers:
            self.final_encoder.add(subencoder)
        self.final_model.add(self.final_encoder)
        self.final_model.add(self.feature_layer)
        for subdecoder in trained_decoder_layers:
            self.final_decoder.add(subdecoder)
        self.final_model.add(self.final_decoder)
        # Start fine-tuning
        print('[INFO] Final SdA model building complete, printing the params......')
        opt = SGD(learning_rate=0.001)
        self.final_model.compile(optimizer=opt, loss='mse')
        print(self.final_model.summary())
        print('[WARNING] Fine-tuning Starting......')
        print("[WARNING] UNLESS THERE IS AN ERROR MESSAGE OR INFORMATION MESSAGE")
        print("[WARNING] PLEASE DON'T TOUCH THE COMPUTER FOR ANY REASON")
        self.final_model.fit(self.expr_matrix, self.expr_matrix, epochs=100, callbacks=[TensorBoard(log_dir='log/')])
        print('[INFO] Fine-tuning completed, saving the model to models/......')
        self.final_encoder.save('models/final_encoder.h5')
        self.final_decoder.save('models/final_decoder.h5')
        print('[INFO] Model saved.')
        return self.final_encoder, self.final_decoder

if __name__ == '__main__':
    SdA_object = SdA(np.array(([1000,1000])))
    SdA_object.train()