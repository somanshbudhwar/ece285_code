import torch
import torch.nn as nn


# Define the Encoder
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2)
        self.bn4 = nn.BatchNorm2d(512)
        self.fc_mean = nn.Linear(262656, latent_dim)
        self.fc_logvar = nn.Linear(262656, latent_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.bn2(self.conv2(x)))
        x = self.leaky_relu(self.bn3(self.conv3(x)))
        x = self.leaky_relu(self.bn4(self.conv4(x)))
        x = x.view(x.size(0), -1)
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        return mean, logvar


# Define the Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()

        self.fc = nn.Linear(latent_dim, 262656)
        self.conv1 = nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2, padding=2, output_padding=(1))
        self.conv2 = nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=(1, 0))
        self.conv3 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=(1, 1))
        self.conv4 = nn.ConvTranspose2d(64, 1, kernel_size=5, stride=2, padding=2, output_padding=(0, 1))
        
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 512, 27, 19)
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.conv2(x))[:,:,:-1,:]
        x = self.leaky_relu(self.conv3(x))
        x = torch.tanh(self.conv4(x))[:,:,:78,:].squeeze(1)
        return x


# Define the VAE
class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z

import torch
import torch.nn as nn

class Seq2SeqLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Seq2SeqLSTM, self).__init__()
        self.hidden_size = hidden_size

        # Encoder LSTM
        self.encoder_lstm = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=1)

        # Decoder LSTM
        self.decoder_lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True, num_layers=1)

        # Output layer
        self.fc1 = nn.Linear(hidden_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_size)



        # self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, input_sequence, target_sequence):
        # Encoder
        encoder_outputs, (encoder_hidden, encoder_cell) = self.encoder_lstm(input_sequence)

        # Decoder
        decoder_hidden = encoder_hidden
        decoder_cell = encoder_cell
        decoder_outputs = []
        decoder_output = decoder_hidden.permute(1, 0, 2)

        #decoder_outputs, (decoder_hidden, decoder_cell) = self.decoder_lstm(
        #    target_sequence, (decoder_hidden, decoder_cell))


        for i in range(target_sequence.size(1)):
            decoder_output, (decoder_hidden, decoder_cell) = self.decoder_lstm(
                 target_sequence[:, i, :].unsqueeze(1), (decoder_hidden, decoder_cell))
# torch.tanh(target_sequence[:, i, :].unsqueeze(1))
            decoder_output = torch.relu(self.fc1(decoder_output))
            decoder_output = torch.relu(self.fc2(decoder_output))
            decoder_output = self.fc3(decoder_output)

            decoder_outputs.append(decoder_output)

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        return decoder_outputs

    def predict(self, input_sequence):
        encoder_outputs, (encoder_hidden, encoder_cell) = self.encoder_lstm(input_sequence)

        decoder_hidden = encoder_hidden
        decoder_cell = encoder_cell
        decoder_output = decoder_hidden.permute(1,0,2)
        decoder_outputs = []

        #decoder_outputs, (decoder_hidden, decoder_cell) = self.decoder_lstm(
            #decoder_output, (decoder_hidden, decoder_cell))

        for i in range(input_sequence.size(1)):
            decoder_output, (decoder_hidden, decoder_cell) = self.decoder_lstm(
                decoder_output, (decoder_hidden, decoder_cell))
            decoder_output = torch.relu(self.fc1(decoder_output))
            decoder_output = torch.relu(self.fc2(decoder_output))
            decoder_output = self.fc3(decoder_output)

            decoder_outputs.append(decoder_output)
        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        return decoder_outputs

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(78, 128)  # Input layer
        self.fc2 = nn.Linear(128, 256)  # Hidden layer
        self.fc3 = nn.Linear(256, 78)  # Output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x



