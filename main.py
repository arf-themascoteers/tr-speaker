import torch
import savee
import preprocessor

savee.prepare_if_needed()

train_x,train_y = savee.get_mel_data("train")
test_x,test_y = savee.get_mel_data("test")

train_x = preprocessor.preprocess_mel(train_x)
test_x = preprocessor.preprocess_mel(test_x)

nn = None # cnn.CNN(train_x, train_y, test_x, test_y)
nn.train()

accuracy, loss = nn.test()
print(f"Test Accuracy {accuracy}. Test Loss {loss}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
