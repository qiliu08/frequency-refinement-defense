
import numpy as np
import utils

class FrequencyRefinement(object):

    def __init__(self, input_preprocessing_type='padding', PD_band = 30, AC_band = 5, GD_band = 80):
    
        # input_preprocessing_type = 'padding' or 'crop'
        self.input_preprocessing_type = input_preprocessing_type
        self.PD_band = PD_band
        self.AC_band = AC_band
        self.GD_band = GD_band
        self.block_size = 8
        self.q_table_setup()
    
    def q_table_setup(self):   
    
        q_table = np.ones((self.block_size, self.block_size)) * self.GD_band
        q_table[0:4, 0:4] = self.AC_band
        q_table[0:2, 0:2] = self.PD_band
        self.q_table = q_table
    
    def input_preprocessing(self,input_matrix):
    
        if self.input_preprocessing_type == 'padding':
            length_h_padding = (-input_matrix.shape[1]) % self.block_size
            length_w_padding = (-input_matrix.shape[2]) % self.block_size
            input_matrix = np.pad(input_matrix, ((0, 0), (0, length_h_padding), (0, length_w_padding), (0, 0)), 'constant',
                              constant_values=(0))
        else:
            length_h_crop = input_matrix.shape[1] % self.block_size
            length_w_crop = input_matrix.shape[2] % self.block_size
            input_matrix = input_matrix[:, :input_matrix.shape[1] - length_h_crop, :input_matrix.shape[2] - length_w_crop,:]
            
        return input_matrix   
          
    def refinement_processing(self, input_matrix=None):
    
        input_matrix = self.input_preprocessing(input_matrix)
        n, h, w, c = input_matrix.shape[0], input_matrix.shape[1], input_matrix.shape[2], input_matrix.shape[3]
        horizontal_blocks_num = w / self.block_size
        output2 = np.zeros((c, h, w))
        output3 = np.zeros((n, 3, h, w))
        vertical_blocks_num = h / self.block_size
        n_block = np.split(input_matrix, n, axis=0)
        # n_block = input_matrix
        for i in range(n):
            c_block = np.split(n_block[i], c, axis=3)
            j = 0
            for ch_block in c_block:
                vertical_blocks = np.split(ch_block, vertical_blocks_num, axis=1)
                k = 0
                for block_ver in vertical_blocks:
                    hor_blocks = np.split(block_ver, horizontal_blocks_num, axis=2)
                    m = 0
                    for block in hor_blocks:
                        block = np.reshape(block, (self.block_size, self.block_size))
                        block = utils.dct2(block)
                        # quantization
                        table_quantized = np.matrix.round(np.divide(block, self.q_table))
                        table_quantized = np.squeeze(np.asarray(table_quantized))
                        # de-quantization
                        table_unquantized = table_quantized * self.q_table
                        IDCT_table = utils.idct2(table_unquantized)
                        if m == 0:
                            output = IDCT_table
                        else:
                            output = np.concatenate((output, IDCT_table), axis=1)
                        m = m + 1
                    if k == 0:
                        output1 = output
                    else:
                        output1 = np.concatenate((output1, output), axis=0)
                    k = k + 1
                output2[j] = output1
                j = j + 1
            output3[i] = output2

        output3 = np.transpose(output3, (0, 2, 1, 3))
        output3 = np.transpose(output3, (0, 1, 3, 2))
        return output3    