import pygame
import numpy as np
import ui_controls

class RecurrentNetwork:
    """
    A simple Recurrent Neural Network (RNN) implementation.
    This network processes sequences of data and uses Backpropagation Through Time (BPTT)
    for training. It features a single hidden layer with a recurrent connection.
    """
    def __init__(self, input_size=3, hidden_size=6, output_size=2, learning_rate=0.01, bptt_truncate=5):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.bptt_truncate = bptt_truncate # How many steps to backpropagate through
        self.title = "Recurrent Network (BPTT)"
        self._init_weights()

        # State and history
        self.last_input = None
        self.last_output = None
        self.h_prev = np.zeros((self.hidden_size, 1))
        self.history = []
        self.training_epochs = 0

        # Simulation cycle control (matches Feedforward network structure)
        self.cycle_count = 0
        self.max_cycles = 10 
        self.current_pattern = 0
        self.training_data = []
        self.generate_training_patterns()

    def _init_weights(self):
        """Initializes weights using standardized random values."""
        self.W_ih = np.random.randn(self.hidden_size, self.input_size) * 0.1
        self.W_hh = np.random.randn(self.hidden_size, self.hidden_size) * 0.1
        self.W_ho = np.random.randn(self.output_size, self.hidden_size) * 0.1
        self.b_h = np.zeros((self.hidden_size, 1))
        self.b_o = np.zeros((self.output_size, 1))
        self.h_prev = np.zeros((self.hidden_size, 1)) # Reset hidden state

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def forward(self, x):
        """Performs a single forward step for one time-step."""
        self.last_input = x
        self.h_prev = np.tanh(np.dot(self.W_ih, x) + np.dot(self.W_hh, self.h_prev) + self.b_h)
        self.last_output = self.sigmoid(np.dot(self.W_ho, self.h_prev) + self.b_o)
        return self.last_output

    def train_network(self):
        """
        Performs training using Backpropagation Through Time (BPTT).
        The 'history' collected from forward passes serves as the sequence.
        """
        if not self.history:
            return

        # Initialize gradients
        dW_ih = np.zeros_like(self.W_ih)
        dW_hh = np.zeros_like(self.W_hh)
        dW_ho = np.zeros_like(self.W_ho)
        db_h = np.zeros_like(self.b_h)
        db_o = np.zeros_like(self.b_o)
        dh_next = np.zeros_like(self.h_prev)

        # Process sequence in reverse for BPTT
        for t in reversed(range(len(self.history))):
            entry = self.history[t]
            x, h, y_pred, y_target = entry['x'], entry['h'], entry['y_pred'], entry['y_target']

            # 1. Output layer error
            d_output = y_pred - y_target
            dW_ho += np.dot(d_output, h.T)
            db_o += d_output

            # 2. Backpropagate to hidden layer
            d_hidden = np.dot(self.W_ho.T, d_output) + dh_next
            d_raw_hidden = (1 - h**2) * d_hidden # tanh derivative
            db_h += d_raw_hidden

            # 3. Backpropagate to input and recurrent weights
            dW_ih += np.dot(d_raw_hidden, x.T)
            
            # Use hidden state from previous time step (t-1)
            h_prev_t = self.history[t-1]['h'] if t > 0 else np.zeros_like(self.h_prev)
            dW_hh += np.dot(d_raw_hidden, h_prev_t.T)

            # Pass gradient to the next (previous in time) step
            dh_next = np.dot(self.W_hh.T, d_raw_hidden)

        # Clip gradients to prevent exploding gradients
        for dparam in [dW_ih, dW_hh, dW_ho, db_h, db_o]:
            np.clip(dparam, -5, 5, out=dparam)

        # Update weights
        self.W_ih -= self.learning_rate * dW_ih
        self.W_hh -= self.learning_rate * dW_hh
        self.W_ho -= self.learning_rate * dW_ho
        self.b_h -= self.learning_rate * db_h
        self.b_o -= self.learning_rate * db_o
        
        self.training_epochs += 1
        self.history = [] # Clear history after training
    
    def get_output_pattern(self):
        if self.last_output is not None:
            return [int(round(v)) for v in self.last_output.flatten()]
        return [0] * self.output_size

    def set_input_pattern(self, pattern):
        self.last_input = np.array(pattern).reshape(-1, 1)

    def update_network(self):
        """
        Runs one forward pass and stores the result for BPTT.
        This is called on each simulation cycle.
        """
        if self.last_input is not None:
            x = self.last_input
            y_target = np.array(self.training_data[self.current_pattern][1]).reshape(-1, 1)
            
            y_pred = self.forward(x)
            
            # Store the state for BPTT
            self.history.append({'x': x.copy(), 'h': self.h_prev.copy(), 'y_pred': y_pred, 'y_target': y_target})

            # Truncate history if it gets too long
            if len(self.history) > self.bptt_truncate:
                self.history.pop(0)

        self.cycle_count += 1

    def reset_network(self):
        """Resets the network's hidden state and history."""
        self.h_prev = np.zeros((self.hidden_size, 1))
        self.cycle_count = 0
        self.history = []
        if self.training_data:
            current_input, _ = self.training_data[self.current_pattern]
            self.set_input_pattern(current_input)
            self.forward(self.last_input)

    def generate_training_patterns(self):
        import random
        self.training_data = []
        for _ in range(4):
            input_pattern = [random.choice([0, 1]) for _ in range(self.input_size)]
            output_pattern = [random.choice([0, 1]) for _ in range(self.output_size)]
            self.training_data.append((input_pattern, output_pattern))
    
    def draw(self, screen, selected_layer=None):
        num_layers = 3 # Input, Hidden, Output
        layer_xs = [250, 550, 850]
        layer_sizes = [self.input_size, self.hidden_size, self.output_size]
        
        neurons = []
        for l, (x, n) in enumerate(zip(layer_xs, layer_sizes)):
            layer = []
            for i in range(n):
                y = 200 + i * 80
                color = ui_controls.GRAY
                if l == 0 and self.last_input is not None and self.last_input[i, 0] > 0.5:
                    color = ui_controls.GREEN
                elif l == 1 and self.h_prev is not None:
                    # Color based on hidden state activation
                    activation = (self.h_prev[i, 0] + 1) / 2 # Tanh is -1 to 1
                    color = (0, int(255 * activation), int(255 * activation))
                elif l == 2 and self.last_output is not None and self.last_output[i, 0] > 0.5:
                    color = ui_controls.ORANGE

                pygame.draw.circle(screen, color, (int(x), y), 20)
                pygame.draw.circle(screen, ui_controls.WHITE, (int(x), y), 20, 2)
                layer.append((int(x), y))
            neurons.append(layer)

        # Draw weights
        # Input -> Hidden
        for i, (x0, y0) in enumerate(neurons[0]):
            for j, (x1, y1) in enumerate(neurons[1]):
                w = self.W_ih[j, i]
                color = ui_controls.GREEN if w > 0 else ui_controls.RED
                thickness = max(1, int(abs(w) * 10))
                pygame.draw.line(screen, color, (x0, y0), (x1, y1), thickness)
        
        # Hidden -> Output
        for i, (x0, y0) in enumerate(neurons[1]):
            for j, (x1, y1) in enumerate(neurons[2]):
                w = self.W_ho[j, i]
                color = ui_controls.GREEN if w > 0 else ui_controls.RED
                thickness = max(1, int(abs(w) * 10))
                pygame.draw.line(screen, color, (x0, y0), (x1, y1), thickness)

        # Recurrent connection (W_hh)
        hidden_x = layer_xs[1]
        top_neuron_y = neurons[1][0][1]
        pygame.draw.arc(screen, ui_controls.CYAN, [hidden_x - 60, top_neuron_y - 20, 40, 40], np.pi/2, np.pi * 1.5, 3)
        pygame.draw.polygon(screen, ui_controls.CYAN, [(hidden_x-40, top_neuron_y+25), (hidden_x-40, top_neuron_y+15), (hidden_x-35, top_neuron_y+20)])
        
        # Draw text info
        text = ui_controls.font.render(self.title, True, ui_controls.WHITE)
        screen.blit(text, (20, 20))
        
        if self.training_data:
            current_input, desired_output = self.training_data[self.current_pattern]
            text1 = ui_controls.small_font.render(f"Input: {current_input}", True, ui_controls.WHITE)
            text2 = ui_controls.small_font.render(f"Desired: {desired_output}", True, ui_controls.GREEN)
            text3 = ui_controls.small_font.render(f"Actual: {self.get_output_pattern()}", True, ui_controls.YELLOW)
            screen.blit(text1, (20, 50))
            screen.blit(text2, (20, 70))
            screen.blit(text3, (20, 90))
        
        text = ui_controls.small_font.render(f"Cycles: {self.cycle_count}/{self.max_cycles}", True, ui_controls.WHITE)
        screen.blit(text, (20, 110))
        text2 = ui_controls.small_font.render(f"Training Epochs: {self.training_epochs}", True, ui_controls.WHITE)
        screen.blit(text2, (20, 130))