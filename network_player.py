import numpy as np
import pygame
import ui_controls

class FeedforwardNetworkRunOnly:
    """
    A run-only version of the Feedforward Network.
    This class can perform a forward pass (inference) but all training
    functionality (backpropagation, etc.) has been removed.
    Weights must be loaded from a file or will be used in their
    randomly initialized state.
    """
    def __init__(self, input_size=3, hidden_sizes=None, output_size=2):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes if hidden_sizes is not None else [6]
        self.output_size = output_size
        self._init_weights()
        
        # Initialize input to a default state (all zeros)
        self.last_input = np.zeros((self.input_size, 1))
        self.last_hidden = None
        self.last_output = None
        # Perform an initial forward pass to populate outputs
        self.forward(self.last_input)


    def _init_weights(self):
        """Initializes weights with random values."""
        self.weights = []
        self.biases = []
        layer_sizes = [self.input_size] + self.hidden_sizes + [self.output_size]
        for i in range(len(layer_sizes)-1):
            # Use Xavier/Glorot initialization for weights
            w = np.random.randn(layer_sizes[i+1], layer_sizes[i]) * np.sqrt(1/layer_sizes[i])
            b = np.zeros((layer_sizes[i+1], 1))
            self.weights.append(w)
            self.biases.append(b)

    def sigmoid(self, x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def forward(self, x):
        """
        Performs a forward pass through the network and returns the output.
        """
        activations = [x]
        self.last_input = x

        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activations[-1]) + b
            a = self.sigmoid(z)
            activations.append(a)
        
        self.last_hidden = activations[1:-1]
        self.last_output = activations[-1]
        return self.last_output

    def set_input_pattern(self, pattern):
        """Sets the input pattern and runs a forward pass."""
        self.last_input = np.array(pattern).reshape(-1, 1)
        self.forward(self.last_input)

    def get_output_pattern(self):
        """Returns the rounded binary output of the last forward pass."""
        if self.last_output is not None:
            return [int(round(v)) for v in self.last_output.flatten()]
        return [0] * self.output_size

    def reset_network(self):
        """Resets the network's state and re-initializes weights."""
        self._init_weights()
        self.last_input = np.zeros((self.input_size, 1))
        self.last_hidden = None
        self.last_output = None
        self.forward(self.last_input)

    def draw(self, screen, selected_layer=None):
        """Draws the network visualization on the screen."""
        num_layers = 1 + len(self.hidden_sizes) + 1
        layer_xs = np.linspace(200, 900, num_layers)
        layer_sizes = [self.input_size] + self.hidden_sizes + [self.output_size]
        
        neurons = []
        for l, (x, n) in enumerate(zip(layer_xs, layer_sizes)):
            layer = []
            # Calculate the starting y position to center the neurons
            total_height = (n - 1) * 80
            start_y = (screen.get_height() - total_height) / 2 - 100

            for i in range(n):
                y = start_y + i * 80
                
                # Determine neuron color based on its activation
                activation = 0
                if l == 0 and self.last_input is not None:
                    activation = self.last_input[i, 0]
                elif l == num_layers - 1 and self.last_output is not None:
                    activation = self.last_output[i, 0]
                elif l > 0 and self.last_hidden and len(self.last_hidden) > l-1:
                    activation = self.last_hidden[l-1][i,0]

                color = (int(255 * activation), int(255 * activation), int(255 * activation))


                pygame.draw.circle(screen, color, (int(x), y), 20)
                pygame.draw.circle(screen, ui_controls.WHITE, (int(x), y), 20, 2)
                
                text_val = f"{activation:.2f}"
                text_surf = ui_controls.small_font.render(text_val, True, ui_controls.RED if activation < 0.5 else ui_controls.GREEN)
                screen.blit(text_surf, (int(x) - 15, y - 8))
                
                layer.append((int(x), y))
            neurons.append(layer)

        # Highlight the selected layer
        if selected_layer is not None and 0 < selected_layer < num_layers-1:
            x = int(layer_xs[selected_layer])
            y_top = neurons[selected_layer][0][1] - 30
            y_bottom = neurons[selected_layer][-1][1] + 30
            pygame.draw.rect(screen, ui_controls.WHITE, (x-40, y_top, 80, y_bottom-y_top), 2, border_radius=5)

        # Draw weights as lines between neurons
        for l in range(len(layer_sizes)-1):
            for i, (x0, y0) in enumerate(neurons[l]):
                for j, (x1, y1) in enumerate(neurons[l+1]):
                    w = self.weights[l][j, i]
                    color = ui_controls.GREEN if w > 0 else ui_controls.RED
                    thickness = max(1, min(int(abs(w) * 4), 10))
                    pygame.draw.line(screen, color, (x0, y0), (x1, y1), thickness)

        # Display informational text
        info_text = "Feedforward Network (Inference Only)"
        text = ui_controls.font.render(info_text, True, ui_controls.WHITE)
        screen.blit(text, (20, 20))

        # Display current I/O
        input_text = f"Input: {[int(v) for v in self.last_input.flatten()]}"
        actual_text = f"Output: {self.get_output_pattern()}"
        text1 = ui_controls.small_font.render(input_text, True, ui_controls.WHITE)
        text3 = ui_controls.small_font.render(actual_text, True, ui_controls.YELLOW)
        screen.blit(text1, (20, 50))
        screen.blit(text3, (20, 70))